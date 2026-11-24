"""Unit tests for the from-scratch Byte-level BPE tokenizer."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest
import regex

from bpe import BPETokenizer, GPT4_SPLIT_PATTERN, merge_ids


def _bruteforce_bpe_merges(
    text: str,
    num_merges: int,
    pattern: str | None,
) -> dict[tuple[int, int], int]:
    """Simple reference trainer that recomputes all pair frequencies every step."""
    if pattern is None:
        sequences = [list(text.encode("utf-8"))] if text else []
    else:
        sequences = [list(chunk.encode("utf-8")) for chunk in regex.findall(pattern, text) if chunk]

    merges: dict[tuple[int, int], int] = {}
    for step in range(num_merges):
        counts: Counter[tuple[int, int]] = Counter()
        for seq in sequences:
            counts.update(zip(seq, seq[1:]))
        if not counts:
            break
        best_pair = max(counts.items(), key=lambda kv: (kv[1], -kv[0][0], -kv[0][1]))[0]
        new_id = 256 + step
        merges[best_pair] = new_id
        sequences = [merge_ids(seq, best_pair, new_id) for seq in sequences]
    return merges


def test_merge_ids_overlapping() -> None:
    assert merge_ids([1, 1, 1], (1, 1), 256) == [256, 1]
    assert merge_ids([1, 1, 1, 1], (1, 1), 256) == [256, 256]
    assert merge_ids([1, 2, 3], (4, 5), 256) == [1, 2, 3]
    assert merge_ids([], (1, 2), 256) == []
    assert merge_ids([42], (42, 42), 256) == [42]


@pytest.mark.parametrize("pattern", [None, GPT4_SPLIT_PATTERN])
def test_roundtrip_unicode_and_edge_cases(pattern: str | None) -> None:
    corpus = (
        "Hello, world! Hello, LLM! 你好，大模型！\n"
        "Tabs\tand\nnewlines\r\nand emojis 🚀🔥🧠 repeated words: code code code.\n"
    ) * 4

    tok = BPETokenizer(pattern=pattern)
    tok.train(corpus, vocab_size=320)

    samples = [
        "",
        "a",
        "   \n\t  ",
        "Hello, world!",
        "大模型分词测试：Unicode 与 Emoji 🚀✨",
        "1234567890 + 42 = 1234567932",
        corpus[:120],
    ]
    for sample in samples:
        ids = tok.encode(sample)
        assert tok.decode(ids) == sample


@pytest.mark.parametrize("pattern", [None, GPT4_SPLIT_PATTERN])
def test_incremental_trainer_matches_bruteforce_reference(pattern: str | None) -> None:
    corpus = (
        "aaaaaaa bbbb aaaa low lower lowest newer wider! "
        "aaaaaaa bbbb aaaa low lower lowest!!?? 中文测试中文测试 1234567 1234567\n"
    )
    target_vocab = 285
    expected_merges = _bruteforce_bpe_merges(corpus, num_merges=target_vocab - 256, pattern=pattern)

    tok = BPETokenizer(pattern=pattern)
    tok.train(corpus, vocab_size=target_vocab)

    assert tok.merges == expected_merges


def test_regex_prevents_cross_boundary_merges() -> None:
    corpus = "dog. dog. dog! dog! dog? " * 6

    basic = BPETokenizer(pattern=None)
    basic.train(corpus, vocab_size=280)

    regex_tok = BPETokenizer(pattern=GPT4_SPLIT_PATTERN)
    regex_tok.train(corpus, vocab_size=280)

    basic_tokens = set(basic.vocab.values())
    regex_tokens = set(regex_tok.vocab.values())

    assert any(b"." in tok or b"!" in tok or b"?" in tok for tok in basic_tokens if b"dog" in tok)
    assert not any(
        b"." in tok or b"!" in tok or b"?" in tok for tok in regex_tokens if b"dog" in tok
    )


def test_special_tokens_modes() -> None:
    corpus = "hello world hello world user assistant " * 5
    specials = ["<|end|>", "<|endoftext|>", "<|user|>"]

    tok = BPETokenizer(pattern=GPT4_SPLIT_PATTERN)
    tok.train(corpus, vocab_size=280, special_tokens=specials)

    end_id = tok.special_tokens["<|end|>"]
    eot_id = tok.special_tokens["<|endoftext|>"]
    user_id = tok.special_tokens["<|user|>"]

    sample = "<|user|>hello<|endoftext|><|end|>"

    with pytest.raises(ValueError):
        tok.encode(sample, allowed_special="none_raise")

    ids_all = tok.encode(sample, allowed_special="all")
    assert ids_all[0] == user_id
    assert ids_all[-2:] == [eot_id, end_id]
    assert tok.decode(ids_all) == sample

    ids_none = tok.encode(sample, allowed_special="none")
    assert user_id not in ids_none
    assert eot_id not in ids_none
    assert end_id not in ids_none
    assert tok.decode(ids_none) == sample

    ids_subset = tok.encode(sample, allowed_special={"<|user|>"})
    assert ids_subset[0] == user_id
    assert eot_id not in ids_subset
    assert end_id not in ids_subset
    assert tok.decode(ids_subset) == sample


def test_decode_invalid_utf8_replaces_safely() -> None:
    tok = BPETokenizer()
    # First 2 bytes of a 3-byte UTF-8 sequence for '中' (0xE4 0xB8 0xAD)
    decoded = tok.decode([0xE4, 0xB8])
    assert "\ufffd" in decoded


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    corpus = "Transformers use Byte-level BPE. 你好世界！\n" * 8
    tok = BPETokenizer(pattern=GPT4_SPLIT_PATTERN)
    tok.train(corpus, vocab_size=300, special_tokens=["<|bos|>", "<|eos|>"])

    # Verify contiguous IDs up to vocab_size - 1
    assert sorted(tok.vocab.keys()) == list(range(tok.vocab_size))

    save_path = tmp_path / "tokenizer.json"
    tok.save(save_path)
    loaded = BPETokenizer.load(save_path)

    assert loaded.vocab_size == tok.vocab_size
    assert loaded.pattern == tok.pattern
    assert loaded.merges == tok.merges
    assert loaded.special_tokens == tok.special_tokens

    sample = "<|bos|>Transformers & 你好世界！<|eos|>"
    assert loaded.encode(sample, allowed_special="all") == tok.encode(sample, allowed_special="all")
    assert loaded.decode(loaded.encode(sample, allowed_special="all")) == sample
