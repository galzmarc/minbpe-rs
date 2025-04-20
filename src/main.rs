mod base;
mod gpt4;

use base::Tokenizer;
use gpt4::GPT4Tokenizer;

fn main() {
    let sample_text = "Hello've world12345 how's are you!!!?";

    let mut tokenizer = GPT4Tokenizer::new();
    let enc = tokenizer.encode(sample_text);
    let dec = tokenizer.decode(&enc);

    let cl100k_base = [9906, 3077, 1917, 4513, 1774, 1268, 596, 527, 499, 12340, 30];
    assert_eq!(enc, cl100k_base);

    println!("{:?}", enc);
    // cl100k_base = 9906, 3077, 1917, 4513, 1774, 1268, 596, 527, 499, 12340, 30;
    // gpt-2: 15496, 1053, 995, 10163, 2231, 703, 338, 389, 345, 10185, 30
    println!("{:?}", dec)
}
