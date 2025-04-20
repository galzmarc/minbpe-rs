# minbpe-rs

**minbpe-rs** is a minimal, fast, and accurate Rust implementation of Byte Pair Encoding (BPE) compatible with GPT-4's tokenizer. It mirrors the functionality of [Andrej Karpathy's `minbpe`](https://github.com/karpathy/minbpe), with careful attention to match OpenAI's `cl100k_base` vocabulary.

---

## ✨ Features

- 🧠 **Accurate** — Reproduces GPT-4's tokenization behavior
- ⚡ **Fast** — Built in Rust for high performance
- 🪶 **Minimal** — No unnecessary dependencies or abstractions
- 📦 **Self-contained** — Embeds `cl100k_base.tiktoken` vocabulary directly
- 🔄 **Encode/Decode** — Full round-trip support between text and tokens

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/galzmarc/minbpe-rs
cd minbpe-rs
```
Then build with Cargo:
```
cargo build --release
```

---

## 🙏 Acknowledgements

- Inspired by Andrej Karpathy
- GPT-4 vocabulary sourced from tiktoken

## 🔮 Future Improvements

Here are some features and enhancements planned or under consideration:

- 🐍 **PyO3 Bindings**  
  Expose `GPT4Tokenizer` to Python via [PyO3](https://github.com/PyO3/pyo3), allowing you to `pip install` a high-performance BPE tokenizer in Python with the same behavior as `tiktoken`

- 📜 **CLI Tool**  
  Provide a simple command-line interface for tokenizing or detokenizing files or strings from the terminal