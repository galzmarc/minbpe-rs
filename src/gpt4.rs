use base64::{engine::general_purpose, Engine as _};
use fancy_regex::Regex;
use indexmap::IndexMap;
use lazy_static::lazy_static;

use crate::base::{Token, Tokenizer};

const GPT4_SPLIT_PATTERN: &str = r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+";

lazy_static! {
    static ref GPT4_SPLIT_COMPILED_PATTERN: Regex = Regex::new(GPT4_SPLIT_PATTERN).unwrap();
}

lazy_static! {
    static ref GPT4_MERGEABLE_RANKS: IndexMap<Vec<u8>, Token> = {
        // https://github.com/zurawiki/tiktoken-rs/blob/main/tiktoken-rs/assets/cl100k_base.tiktoken
        let cl100k_base: &str = include_str!("../assets/cl100k_base.tiktoken");

        let mut encoder = IndexMap::default();
        for line in cl100k_base.lines() {
            let mut parts = line.split(' ');
            let raw = parts.next().unwrap();
            let token = &general_purpose::STANDARD.decode(raw).unwrap();
            let rank: Token = parts.next().unwrap().parse().unwrap();
            if rank < 0 {
                panic!("Rank {} for token {:?} is negative", rank, token);
            }
            encoder.insert(token.clone(), rank);
        }
        encoder
    };
}

fn bpe(
    mergeable_ranks: &IndexMap<Vec<u8>, Token>,
    token: &[u8],
    max_rank: Option<Token>,
) -> Vec<Vec<u8>> {
    let mut parts: Vec<Vec<u8>> = Vec::with_capacity(token.len());
    for &b in token {
        parts.push(vec![b]);
    }

    loop {
        let mut min_idx = None;
        let mut min_rank = None;
        for (i, pair) in parts.windows(2).enumerate() {
            let rank = mergeable_ranks.get(&[pair[0].clone(), pair[1].clone()].concat());
            if let Some(rank) = rank {
                if min_rank.is_none() || rank < min_rank.unwrap() {
                    min_idx = Some(i);
                    min_rank = Some(rank);
                }
            }
        }
        if min_rank.is_none() || (max_rank.is_some() && *min_rank.unwrap() >= max_rank.unwrap()) {
            break;
        }
        let min_idx = min_idx.unwrap();
        parts[min_idx] = [parts[min_idx].clone(), parts[min_idx + 1].clone()].concat();
        parts.remove(min_idx + 1);
    }
    parts
}

fn recover_merges(mergeable_ranks: &IndexMap<Vec<u8>, Token>) -> IndexMap<(Token, Token), Token> {
    let mut merges = IndexMap::new();
    for (token, &rank) in mergeable_ranks {
        if token.len() == 1 {
            continue;
        }
        let pair = bpe(mergeable_ranks, token, Some(rank));
        assert_eq!(pair.len(), 2);
        // recover the integer ranks of the pair
        let ix0 = mergeable_ranks[&pair[0]];
        let ix1 = mergeable_ranks[&pair[1]];
        merges.insert((ix0, ix1), rank);
    }
    merges
}

pub struct GPT4Tokenizer {
    // Lightweight wrapper on RegexTokenizer that matches GPT-4's tokenizer
    merges: IndexMap<(Token, Token), Token>,
    vocab: IndexMap<Token, Vec<u8>>,

    byte_shuffle: IndexMap<u8, u8>,
    inverse_byte_shuffle: IndexMap<u8, u8>,
}

impl GPT4Tokenizer {
    pub fn new() -> Self {
        // let enc = cl100k_base().unwrap();
        let mergeable_ranks = &GPT4_MERGEABLE_RANKS;
        let merges = recover_merges(mergeable_ranks);
        let mut vocab: IndexMap<Token, Vec<u8>> =
            (0..=255).map(|i| (i as Token, vec![i])).collect();
        for (&(p0, p1), &idx) in &merges {
            let mut token = vocab[&p0].clone();
            token.extend_from_slice(&vocab[&p1]);
            vocab.insert(idx, token);
        }

        let byte_shuffle: IndexMap<u8, u8> = (0..=255)
            .map(|i| {
                let value = mergeable_ranks[&vec![i]];
                if value < 0 || value > u8::MAX as Token {
                    panic!(
                        "Value {} for key {} in mergeable_ranks does not fit in u8",
                        value, i
                    );
                }
                (i, value as u8)
            })
            .collect();

        let inverse_byte_shuffle: IndexMap<u8, u8> =
            byte_shuffle.iter().map(|(&k, &v)| (v, k)).collect();

        GPT4Tokenizer {
            merges,
            vocab,
            byte_shuffle,
            inverse_byte_shuffle,
        }
    }

    fn get_stats(&self, ids: &[Token]) -> IndexMap<(Token, Token), Token> {
        let mut counts = IndexMap::new();
        for pair in ids.windows(2) {
            // `windows(2)` creates pairs efficiently
            let pair = (pair[0], pair[1]);
            *counts.entry(pair).or_insert(0) += 1;
        }
        counts
    }

    fn merge(&self, ids: &[Token], pair: (Token, Token), new_token: Token) -> Vec<Token> {
        // in the slice of ints (ids), replace all consecutive occurences of pair with the new token
        let mut new_ids = Vec::with_capacity(ids.len());
        let mut i = 0;
        while i < ids.len() {
            // if we are not at the very last position and the pair matches, replace it
            if i < ids.len() - 1 && ids[i] == pair.0 && ids[i + 1] == pair.1 {
                new_ids.push(new_token);
                i += 2;
            } else {
                new_ids.push(ids[i]);
                i += 1;
            }
        }
        new_ids
    }

    fn encode_chunk_inner(&self, text_bytes: &[u8]) -> Vec<Token> {
        let merges = &self.merges;
        let mut ids: Vec<Token> = text_bytes.iter().map(|&b| b as Token).collect();
        while ids.len() >= 2 {
            // Find the pair with the lowest merge index
            let stats = self.get_stats(&ids);

            let pair_opt = stats
                .keys()
                .filter_map(|&pair| merges.get(&pair).map(|_| pair))
                .min_by_key(|&pair| merges[&pair]);

            match pair_opt {
                None => break, // If there are no more merges available, break
                Some(pair) => {
                    // Otherwise, merge the best pair (lowest merge index)
                    let idx = merges[&pair];
                    ids = self.merge(&ids, pair, idx);
                }
            };
        }
        ids
    }

    fn encode_chunk(&self, text_bytes: &[u8]) -> Vec<Token> {
        let text_bytes: Vec<u8> = text_bytes.iter().map(|&b| self.byte_shuffle[&b]).collect();
        self.encode_chunk_inner(&text_bytes)
    }
}

impl Tokenizer for GPT4Tokenizer {
    fn encode(&mut self, text: &str) -> Vec<Token> {
        let re = &GPT4_SPLIT_COMPILED_PATTERN;
        let text_chunks: Vec<_> = re
            .find_iter(text)
            .map(|m| {
                let matched = m.unwrap();
                &text[matched.start()..matched.end()]
            })
            .collect();
        let mut ids = Vec::new();
        for chunk in text_chunks {
            let chunk_bytes = chunk.as_bytes();
            let chunk_ids = self.encode_chunk(chunk_bytes);
            ids.extend(chunk_ids);
        }
        ids
    }

    fn decode(&self, ids: &[Token]) -> String {
        let text_bytes: Vec<u8> = ids
            .iter()
            .flat_map(|&idx| self.vocab[&idx].clone())
            .collect();
        let text_bytes: Vec<u8> = text_bytes
            .into_iter()
            .map(|b| self.inverse_byte_shuffle[&b])
            .collect();
        String::from_utf8_lossy(&text_bytes).to_string()
    }
}
