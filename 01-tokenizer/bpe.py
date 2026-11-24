"""Byte-level BPE tokenizer implemented from scratch.

Covers the core pipeline used by modern LLMs (GPT-2 / GPT-4 / Llama 3 style):
1. UTF-8 byte base vocabulary (256 byte tokens, zero UNK).
2. Regex pre-tokenization so merges never cross word/punctuation/number boundaries.
3. Deduplicated incremental BPE training.
4. Rank-ordered greedy encoding + safe UTF-8 decoding + special tokens.
"""

from __future__ import annotations

import json
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Collection

import regex

# Classic GPT-2 pre-tokenization regex (single contractions, optional leading space + letters/digits/punct)
GPT2_SPLIT_PATTERN = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)

# GPT-4 (cl100k_base) pre-tokenization regex:
# - Case-insensitive English contractions
# - Numbers chunked in groups of 1-3 digits to keep numerical structure crisp
# - Possessive quantifiers (?+ / ++) to avoid catastrophic backtracking on long punctuation runs
GPT4_SPLIT_PATTERN = (
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
)


def merge_ids(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Greedily replace non-overlapping adjacent occurrences of `pair` with `new_id`."""
    if len(ids) < 2:
        return list(ids)

    p0, p1 = pair
    out: list[int] = []
    i = 0
    n = len(ids)
    while i < n:
        if i + 1 < n and ids[i] == p0 and ids[i + 1] == p1:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


def render_bytes(token_bytes: bytes) -> str:
    """Decode bytes into a printable string with control characters escaped."""
    text = token_bytes.decode("utf-8", errors="replace")
    chars: list[str] = []
    for ch in text:
        if ch == "\n":
            chars.append("\\n")
        elif ch == "\r":
            chars.append("\\r")
        elif ch == "\t":
            chars.append("\\t")
        elif unicodedata.category(ch).startswith("C"):
            chars.append(f"\\u{ord(ch):04x}")
        else:
            chars.append(ch)
    return "".join(chars)


class BPETokenizer:
    """Byte-level Byte Pair Encoding tokenizer with regex pre-tokenization and special tokens."""

    def __init__(self, pattern: str | None = GPT4_SPLIT_PATTERN) -> None:
        self.pattern = pattern
        self._compiled_pattern = regex.compile(pattern) if pattern else None
        # Maps (left_id, right_id) -> merged_id; merged_id >= 256 also defines merge priority (rank).
        self.merges: dict[tuple[int, int], int] = {}
        self.special_tokens: dict[str, int] = {}
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def _split_text(self, text: str) -> list[str]:
        if not text:
            return []
        if self._compiled_pattern is None:
            return [text]
        return regex.findall(self._compiled_pattern, text)

    def train(
        self,
        text: str,
        vocab_size: int,
        special_tokens: list[str] | None = None,
        verbose: bool = False,
    ) -> list[tuple[tuple[int, int], int, int]]:
        """Train BPE merges on `text` until `vocab_size` tokens (including 256 bytes + specials)."""
        specials = list(special_tokens or [])
        if len(set(specials)) != len(specials):
            raise ValueError("special_tokens must not contain duplicates.")
        if any(not tok for tok in specials):
            raise ValueError("special_tokens must not contain empty strings.")

        min_vocab = 256 + len(specials)
        if vocab_size < min_vocab:
            raise ValueError(
                f"vocab_size ({vocab_size}) must be >= 256 + len(special_tokens) ({min_vocab})."
            )

        self.merges = {}
        self.special_tokens = {}
        self.vocab = {i: bytes([i]) for i in range(256)}

        # Deduplicate pre-tokenized byte chunks so training scales with unique chunks, not raw text length.
        chunk_counts: Counter[bytes] = Counter()
        for chunk in self._split_text(text):
            chunk_bytes = chunk.encode("utf-8")
            if chunk_bytes:
                chunk_counts[chunk_bytes] += 1

        words: list[list[int]] = []
        freqs: list[int] = []
        for chunk_bytes, freq in chunk_counts.items():
            words.append(list(chunk_bytes))
            freqs.append(freq)

        pair_counts: dict[tuple[int, int], int] = defaultdict(int)
        pair_to_words: dict[tuple[int, int], set[int]] = defaultdict(set)
        for idx, word in enumerate(words):
            if len(word) < 2:
                continue
            freq = freqs[idx]
            for pair in zip(word, word[1:]):
                pair_counts[pair] += freq
                pair_to_words[pair].add(idx)

        target_merges = vocab_size - min_vocab
        history: list[tuple[tuple[int, int], int, int]] = []

        for step in range(target_merges):
            if not pair_counts:
                break

            # Deterministic tie-breaking: highest count first, then lexicographically smallest pair.
            best_pair, best_freq = max(
                pair_counts.items(),
                key=lambda kv: (kv[1], -kv[0][0], -kv[0][1]),
            )
            new_id = 256 + step
            self.merges[best_pair] = new_id
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            history.append((best_pair, new_id, best_freq))

            if verbose:
                left = render_bytes(self.vocab[best_pair[0]])
                right = render_bytes(self.vocab[best_pair[1]])
                merged = render_bytes(self.vocab[new_id])
                print(
                    f"merge {step + 1}/{target_merges}: "
                    f"{best_pair} -> {new_id} ({left!r} + {right!r} -> {merged!r}) freq={best_freq}"
                )

            affected_indices = list(pair_to_words.get(best_pair, ()))
            for idx in affected_indices:
                old_word = words[idx]
                freq = freqs[idx]

                for pair in zip(old_word, old_word[1:]):
                    new_count = pair_counts[pair] - freq
                    if new_count <= 0:
                        pair_counts.pop(pair, None)
                    else:
                        pair_counts[pair] = new_count
                    word_set = pair_to_words.get(pair)
                    if word_set is not None:
                        word_set.discard(idx)
                        if not word_set:
                            pair_to_words.pop(pair, None)

                new_word = merge_ids(old_word, best_pair, new_id)
                words[idx] = new_word

                for pair in zip(new_word, new_word[1:]):
                    pair_counts[pair] += freq
                    pair_to_words[pair].add(idx)

        # Assign contiguous IDs right after actual merges so nn.Embedding has no unused gaps.
        next_id = 256 + len(self.merges)
        for tok in specials:
            self.special_tokens[tok] = next_id
            self.vocab[next_id] = tok.encode("utf-8")
            next_id += 1

        return history

    def _encode_chunk(self, chunk_bytes: bytes) -> list[int]:
        ids = list(chunk_bytes)
        while len(ids) >= 2:
            best_pair: tuple[int, int] | None = None
            best_rank = float("inf")
            for pair in zip(ids, ids[1:]):
                rank = self.merges.get(pair)
                if rank is not None and rank < best_rank:
                    best_rank = rank
                    best_pair = pair
            if best_pair is None:
                break
            ids = merge_ids(ids, best_pair, self.merges[best_pair])
        return ids

    def _encode_ordinary(self, text: str) -> list[int]:
        if not text:
            return []
        ids: list[int] = []
        for chunk in self._split_text(text):
            ids.extend(self._encode_chunk(chunk.encode("utf-8")))
        return ids

    def encode(
        self,
        text: str,
        allowed_special: str | Collection[str] = "none_raise",
    ) -> list[int]:
        """Encode `text` into token IDs according to `allowed_special` policy."""
        if not text:
            return []

        if allowed_special == "all":
            special = self.special_tokens
        elif allowed_special == "none":
            special = {}
        elif allowed_special == "none_raise":
            special = {}
            for tok in self.special_tokens:
                if tok in text:
                    raise ValueError(
                        f"Encountered special token {tok!r} with allowed_special='none_raise'."
                    )
        elif isinstance(allowed_special, str):
            raise ValueError(f"Unsupported allowed_special value: {allowed_special!r}")
        else:
            allowed_set = set(allowed_special)
            special = {tok: idx for tok, idx in self.special_tokens.items() if tok in allowed_set}

        if not special:
            return self._encode_ordinary(text)

        # Sort longest-first so prefix special tokens (e.g. "<|end|>" vs "<|endoftext|>") match greedily.
        special_pattern = "(" + "|".join(
            regex.escape(tok) for tok in sorted(special, key=len, reverse=True)
        ) + ")"
        parts = regex.split(special_pattern, text)

        ids: list[int] = []
        for part in parts:
            if not part:
                continue
            if part in special:
                ids.append(special[part])
            else:
                ids.extend(self._encode_ordinary(part))
        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs back into a UTF-8 string (invalid sequences become U+FFFD)."""
        byte_parts: list[bytes] = []
        for token_id in ids:
            if token_id not in self.vocab:
                raise KeyError(f"Unknown token id: {token_id}")
            byte_parts.append(self.vocab[token_id])
        return b"".join(byte_parts).decode("utf-8", errors="replace")

    def save(self, path: str | Path) -> None:
        """Persist tokenizer configuration, merges, and special tokens to a JSON file."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "pattern": self.pattern,
            "merges": [
                [p0, p1, new_id]
                for (p0, p1), new_id in sorted(self.merges.items(), key=lambda kv: kv[1])
            ],
            "special_tokens": self.special_tokens,
        }
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> BPETokenizer:
        """Load a tokenizer previously saved with `BPETokenizer.save`."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        tokenizer = cls(pattern=data["pattern"])
        for p0, p1, new_id in sorted(data["merges"], key=lambda item: item[2]):
            pair = (int(p0), int(p1))
            nid = int(new_id)
            tokenizer.merges[pair] = nid
            tokenizer.vocab[nid] = tokenizer.vocab[pair[0]] + tokenizer.vocab[pair[1]]
        for tok, sid in data.get("special_tokens", {}).items():
            sid_int = int(sid)
            tokenizer.special_tokens[tok] = sid_int
            tokenizer.vocab[sid_int] = tok.encode("utf-8")
        return tokenizer
