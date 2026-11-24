"""Local experiments illustrating Byte-level BPE behavior and trade-offs."""

from __future__ import annotations

import time
from pathlib import Path

from bpe import BPETokenizer, GPT4_SPLIT_PATTERN, render_bytes


FALLBACK_CORPUS = (
    "First Citizen:\n"
    "Before we proceed any further, hear me speak.\n\n"
    "All:\n"
    "Speak, speak.\n\n"
    "First Citizen:\n"
    "You are all resolved rather to die than to famish?\n\n"
    "All:\n"
    "Resolved. resolved.\n\n"
    "First Citizen:\n"
    "First, you know Caius Marcius is chief enemy to the people.\n"
) * 300


def load_shakespeare_corpus() -> tuple[str, str]:
    repo_root = Path(__file__).resolve().parent.parent
    corpus_path = repo_root / "nanoGPT" / "data" / "shakespeare_char" / "input.txt"
    if corpus_path.exists():
        return corpus_path.read_text(encoding="utf-8"), str(corpus_path)
    return FALLBACK_CORPUS, "<built-in fallback sample>"


def compression_ratio(text: str, token_ids: list[int]) -> float:
    raw_bytes = len(text.encode("utf-8"))
    return raw_bytes / max(len(token_ids), 1)


def run_experiment_1_toy() -> None:
    print("=== Experiment 1: Step-by-Step Merges on Toy Corpus ===")
    toy_text = "low lower lowest newer wider " * 5
    tok = BPETokenizer(pattern=GPT4_SPLIT_PATTERN)
    history = tok.train(toy_text, vocab_size=264)

    for step, (pair, new_id, freq) in enumerate(history[:8], start=1):
        left = render_bytes(tok.vocab[pair[0]])
        right = render_bytes(tok.vocab[pair[1]])
        merged = render_bytes(tok.vocab[new_id])
        print(
            f"  step {step:2d}: {pair[0]:3d} ({left!r}) + {pair[1]:3d} ({right!r}) "
            f"-> {new_id:3d} ({merged!r}) | freq={freq}"
        )

    probe = "lowest newer"
    probe_ids = tok.encode(probe)
    probe_pieces = [render_bytes(tok.vocab[i]) for i in probe_ids]
    print(f"  encode({probe!r}) -> ids={probe_ids} pieces={probe_pieces}")


def run_experiment_2_basic_vs_regex(corpus: str, source: str) -> BPETokenizer:
    print("\n=== Experiment 2: Basic BPE (no regex) vs GPT-4 Regex BPE ===")
    print(f"Corpus source: {source} (total {len(corpus.encode('utf-8')):,} bytes)")
    slice_text = corpus[:100_000]
    slice_bytes = len(slice_text.encode("utf-8"))

    t0 = time.perf_counter()
    basic_tok = BPETokenizer(pattern=None)
    basic_tok.train(slice_text, vocab_size=512)
    basic_time = time.perf_counter() - t0
    basic_ids = basic_tok.encode(slice_text)

    t0 = time.perf_counter()
    regex_tok = BPETokenizer(pattern=GPT4_SPLIT_PATTERN)
    regex_tok.train(slice_text, vocab_size=512)
    regex_time = time.perf_counter() - t0
    regex_ids = regex_tok.encode(slice_text)

    print(
        f"  [100KB slice, vocab=512] Basic BPE : {basic_time:.2f}s | "
        f"{len(basic_ids):,} tokens | {slice_bytes / len(basic_ids):.2f} bytes/token"
    )
    print(
        f"  [100KB slice, vocab=512] Regex BPE : {regex_time:.2f}s | "
        f"{len(regex_ids):,} tokens | {slice_bytes / len(regex_ids):.2f} bytes/token"
    )

    basic_cross = [
        render_bytes(basic_tok.vocab[i])
        for i in range(256, basic_tok.vocab_size)
        if any(c in basic_tok.vocab[i] for c in (b"\n", b":", b",", b".", b"?", b"!"))
        and any(b"a" <= bytes([c]) <= b"z" or b"A" <= bytes([c]) <= b"Z" for c in basic_tok.vocab[i])
    ]
    regex_sample = [render_bytes(regex_tok.vocab[i]) for i in range(256, 276)]
    print(f"  Basic BPE cross-boundary tokens (letters+punct/newline): {basic_cross[:10]}")
    print(f"  Regex BPE first 20 merged tokens: {regex_sample}")

    specials = ["<|endoftext|>", "<|user|>", "<|assistant|>"]
    t0 = time.perf_counter()
    full_tok = BPETokenizer(pattern=GPT4_SPLIT_PATTERN)
    full_tok.train(corpus, vocab_size=1024, special_tokens=specials)
    full_time = time.perf_counter() - t0
    full_ids = full_tok.encode(corpus)
    full_bytes = len(corpus.encode("utf-8"))
    print(
        f"  [Full corpus, vocab=1024] Regex BPE: {full_time:.2f}s | "
        f"{len(full_ids):,} tokens | {full_bytes / len(full_ids):.2f} bytes/token"
    )
    return full_tok


def _try_load_tiktoken(name: str) -> object | None:
    try:
        import tiktoken

        return tiktoken.get_encoding(name)
    except Exception as exc:
        print(f"  [tiktoken] skipped {name!r}: {exc}")
        return None


def run_experiment_3_cross_domain(custom_tok: BPETokenizer) -> None:
    print("\n=== Experiment 3: Cross-Lingual / Code / Digits & Special Tokens ===")
    samples = [
        (
            "English prose",
            "Attention is all you need, and Byte-level BPE keeps token boundaries clean.",
        ),
        (
            "Chinese tech",
            "大语言模型使用 Byte-level BPE 分词，中文未高频合并时通常按 UTF-8 字节切分。",
        ),
        (
            "Python code",
            "def forward(self, x: torch.Tensor) -> torch.Tensor:\n    return self.proj(x) * 0.5",
        ),
        ("Long digits", "1234567890"),
    ]

    gpt2_enc = _try_load_tiktoken("gpt2")
    cl100k_enc = _try_load_tiktoken("cl100k_base")

    for label, text in samples:
        byte_len = len(text.encode("utf-8"))
        custom_ids = custom_tok.encode(text)
        line = (
            f"  {label:<14} ({byte_len:2d}B) -> "
            f"Our-1024: {len(custom_ids):2d} tok ({byte_len / len(custom_ids):.2f} B/tok)"
        )
        if gpt2_enc is not None:
            gpt2_ids = gpt2_enc.encode(text)
            line += f" | gpt2: {len(gpt2_ids):2d} tok ({byte_len / len(gpt2_ids):.2f} B/tok)"
        if cl100k_enc is not None:
            cl_ids = cl100k_enc.encode(text)
            line += f" | cl100k: {len(cl_ids):2d} tok ({byte_len / len(cl_ids):.2f} B/tok)"
        print(line)

    if cl100k_enc is not None:
        digit_pieces = [render_bytes(cl100k_enc.decode_single_token_bytes(i)) for i in cl100k_enc.encode("1234567890")]
        print(f"  cl100k_base digit split for '1234567890': {digit_pieces}")

    chat_sample = "<|user|>Speak, speak!<|assistant|>Before we proceed.<|endoftext|>"
    chat_ids = custom_tok.encode(chat_sample, allowed_special="all")
    chat_pieces = [render_bytes(custom_tok.vocab[i]) for i in chat_ids]
    print(f"  Chat with special tokens -> {len(chat_ids)} tokens: {chat_pieces}")


def main() -> None:
    corpus, source = load_shakespeare_corpus()
    run_experiment_1_toy()
    custom_tok = run_experiment_2_basic_vs_regex(corpus, source)
    run_experiment_3_cross_domain(custom_tok)


if __name__ == "__main__":
    main()
