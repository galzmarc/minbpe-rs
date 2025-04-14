pub type Token = i32;

pub trait Tokenizer {
    fn encode(&mut self, text: &str) -> Vec<Token>;
    fn decode(&self, ids: &[Token]) -> String;
}
