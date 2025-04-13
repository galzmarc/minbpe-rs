use std::fs::*;

use tokenizer::Tokenizer;

fn main() {
    let train_text =
        read_to_string("./taylorswift.txt").expect("Should have been able to read the file");

    let sample_text = "Hello've world12345 how's are you!!!?";

    let mut tokenizer = Tokenizer::new();
    tokenizer.train(&train_text, 32768);
    let enc = tokenizer.encode(sample_text);
    let dec = tokenizer.decode(&enc);

    println!("{:?}", enc);
    // cl100k_base: 9906, 3077, 1917, 4513, 1774, 1268, 596, 527, 499, 12340, 30
    // gpt-2: 15496, 1053, 995, 10163, 2231, 703, 338, 389, 345, 10185, 30
    println!("{:?}", dec);
}
