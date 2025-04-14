use crate::base::{Token, Tokenizer};
use fancy_regex::Regex;
use std::collections::HashMap;

const GPT4_SPLIT_PATTERN: &str = r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+";

pub struct RegexTokenizer {
    merges: HashMap<(Token, Token), Token>,
    vocab: HashMap<Token, Vec<u8>>,
    pattern: String,
    cache: HashMap<String, Vec<Token>>,
}

impl RegexTokenizer {
    pub fn new() -> Self {
        let mut tokenizer = RegexTokenizer {
            merges: HashMap::new(),
            vocab: HashMap::new(),
            pattern: GPT4_SPLIT_PATTERN.to_string(),
            cache: HashMap::new(),
        };
        tokenizer.build_vocab();
        tokenizer
    }

    fn get_stats(&self, ids: &[Token]) -> HashMap<(Token, Token), Token> {
        let mut counts = HashMap::new();
        for pair in ids.windows(2) {
            // `windows(2)` creates pairs efficiently
            let pair = (pair[0], pair[1]);
            *counts.entry(pair).or_insert(0) += 1;
        }
        counts
    }

    fn sorted_stats(&self, stats: HashMap<(Token, Token), i32>) -> Vec<((Token, Token), i32)> {
        let mut sorted_pairs: Vec<_> = stats.into_iter().collect();
        sorted_pairs.sort_by(|a, b| a.1.cmp(&b.1)); // Sort in ascending order
        sorted_pairs
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

    /// Train a vocabulary of size `vocab_size` in distinct Tokens from `text`.
    pub fn train(&mut self, text: &str, vocab_size: i32) {
        self.cache.clear();

        assert!(vocab_size >= 256, "Vocab size must be at least 256");
        let num_merges = vocab_size - 256;

        let text_bytes = text.as_bytes();
        let mut ids: Vec<Token> = text_bytes.iter().map(|t| *t as Token).collect();

        for i in 0..num_merges {
            let stats = self.get_stats(&ids);
            let mut sorted = self.sorted_stats(stats);
            let idx = 256 + i;
            if let Some((top_pair, _count)) = sorted.pop() {
                ids = self.merge(&ids, top_pair, idx);
                self.merges.insert(top_pair, idx);
                self.vocab.insert(
                    idx,
                    [
                        self.vocab[&top_pair.0].clone(),
                        self.vocab[&top_pair.1].clone(),
                    ]
                    .concat(),
                );
            }
        }
    }

    // Given a string, return a list of integers (tokens)
    fn bpe(&mut self, text: &str) -> Vec<Token> {
        if let Some(cached) = self.cache.get(text) {
            return cached.clone();
        }
        // Convert all bytes to integers in range 0..255
        let text_bytes = text.as_bytes();
        let mut ids: Vec<i32> = text_bytes.into_iter().map(|&t| t as Token).collect();

        while ids.len() >= 2 {
            let stats = self.get_stats(&ids);
            // Find the pair with the lowest merge index
            let pair = stats
                .keys()
                .min_by_key(|&&p| self.merges.get(&p).unwrap_or(&i32::MAX));
            // If no valid merge is found, stop
            if let Some(&pair) = pair {
                if !self.merges.contains_key(&pair) {
                    break;
                }
                // Merge the best pair
                let idx = self.merges[&pair];
                ids = self.merge(&ids, pair, idx);
            } else {
                break;
            }
        }
        self.cache.insert(text.to_string(), ids.clone());
        ids
    }

    fn build_vocab(&mut self) {
        self.vocab = (0..256).map(|idx| (idx, vec![idx as u8])).collect();
        // Reconstruct the vocab
        for ((p0, p1), idx) in &self.merges {
            if let (Some(v0), Some(v1)) = (self.vocab.get(&p0), self.vocab.get(&p1)) {
                let mut merged = v0.clone();
                merged.extend(v1);
                self.vocab.insert(*idx, merged);
            }
        }
    }
}

impl Tokenizer for RegexTokenizer {
    /// A Tokenizer can encode a string into a list of integers.
    fn encode(&mut self, text: &str) -> Vec<Token> {
        // split text into chunks of text by categories defined in regex pattern
        let re = Regex::new(&self.pattern).unwrap();
        let text_chunks: Vec<_> = re.find_iter(text).map(|m| m.unwrap().as_str()).collect();
        // all chunks of text are encoded separately, then results are joined
        let mut ids: Vec<Token> = Vec::new();
        for chunk in text_chunks {
            let chunk_ids = self.bpe(chunk);
            ids.extend(chunk_ids);
        }
        ids
    }

    /// A Tokenizer can decode a list of integers into a string.
    fn decode(&self, ids: &[Token]) -> String {
        // Decode the ids into bytes
        let mut text_bytes = Vec::new();
        for &id in ids {
            if let Some(bytes) = self.vocab.get(&id) {
                text_bytes.extend(bytes);
            }
        }
        // Convert bytes to String
        String::from_utf8(text_bytes).unwrap_or_else(|_| "�".to_string())
    }
}
