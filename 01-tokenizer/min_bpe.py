"""Minimal pure-Python Byte-level BPE (Karpathy minbpe style)."""

from collections import Counter


def get_stats(ids: list[int]) -> Counter[tuple[int, int]]:
    return Counter(zip(ids, ids[1:]))


def merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    out, i = [], 0
    while i < len(ids):
        if i + 1 < len(ids) and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class MinBPE:
    def __init__(self) -> None:
        self.merges: dict[tuple[int, int], int] = {}
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    def train(self, text: str, vocab_size: int, verbose: bool = True) -> None:
        ids = list(text.encode("utf-8"))
        for i in range(vocab_size - 256):
            stats = get_stats(ids)
            if not stats:
                break
            pair = max(stats, key=stats.get)
            new_id = 256 + i
            ids = merge(ids, pair, new_id)
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            if verbose:
                b0, b1, bm = self.vocab[pair[0]], self.vocab[pair[1]], self.vocab[new_id]
                print(
                    f"  step {i + 1:2d}: ({pair[0]:3d}, {pair[1]:3d}) -> {new_id} | "
                    f"[{b0.hex(' ')}] + [{b1.hex(' ')}] -> [{bm.hex(' ')}] "
                    f"({bm.decode('utf-8', errors='replace')!r}) freq={stats[pair]}"
                )

    def encode(self, text: str) -> list[int]:
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = merge(ids, pair, self.merges[pair])
        return ids

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")


def show_encoding(bpe: MinBPE, label: str, text: str) -> list[int]:
    raw = text.encode("utf-8")
    ids = bpe.encode(text)
    decoded = bpe.decode(ids)
    assert decoded == text, f"Roundtrip mismatch: {decoded!r} != {text!r}"

    ratio = len(raw) / len(ids) if ids else 0.0
    print(f"\n[{label}] {text!r}")
    print(f"  原始 UTF-8: {len(raw)} 字节 [{raw.hex(' ')}]")
    print(f"  编码结果  : {len(ids)} tokens {ids} (压缩比 {ratio:.2f} B/token)")
    for tid in ids:
        tok_bytes = bpe.vocab[tid]
        kind = "基础字节" if tid < 256 else "合并token"
        preview = tok_bytes.decode("utf-8", errors="replace")
        print(f"    - id={tid:3d} ({kind}) | bytes=[{tok_bytes.hex(' ')}] | {preview!r}")
    print(f"  解码还原  : {decoded!r} (无损还原 OK)")
    return ids


if __name__ == "__main__":
    # 语料故意不含 '龘' (E9 BE 98) 和 '🚀' (F0 9F 9A 80)，
    # 但包含共享前缀 E9 BE 的 '龍' (E9 BE 8D) 与 '龜' (E9 BE 9C)
    corpus = (
        "hello token," * 5
        + "token hello." * 4
        + "Byte-level BPE," * 2
        + "你好大模型," * 4
        + "大模型你好。" * 4
        + "你好大模型！" * 3
        + "龍龜," * 6
    )

    print("=== 1. 训练与合并过程 (vocab_size=285, 256 基础字节 + 29 次合并) ===")
    bpe = MinBPE()
    bpe.train(corpus, vocab_size=285, verbose=True)

    print("\n=== 2. 英文 / 中文 / 未见生僻字 / Emoji 编码效果 ===")
    show_encoding(bpe, "英文高频词", "hello token")
    show_encoding(bpe, "中文高频词", "你好大模型")
    rare_ids = show_encoding(bpe, "未见生僻字(共享前缀退化)", "龘")
    show_encoding(bpe, "未见Emoji(完全退化到基础字节)", "🚀")
    show_encoding(bpe, "中英符号混合", "hello 你好 龘 🚀")

    print("\n=== 3. 流式半截 UTF-8 解码演示 ('龘' = [E9 BE] + [98]) ===")
    print(f"  仅解码第 1 个 token {rare_ids[:1]} -> {bpe.decode(rare_ids[:1])!r} (不完整字节安全替换)")
    print(f"  拼齐全部 token  {rare_ids}     -> {bpe.decode(rare_ids)!r} (完整恢复)")
