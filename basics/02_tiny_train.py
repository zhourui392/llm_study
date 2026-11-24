"""极简 LLM 训练实验（实验 B）：看随机权重如何学会“猜下一个词”。

延续实验 A 的“苹果”双语境：
- 句子 A: ["拍照", "用", "苹果", "很", "清晰"]
- 句子 B: ["咬",   "一口", "苹果", "很", "脆甜"]

故意让两句话在最后都要从相同的 "苹果 很" 往下猜：
只有模型自己把 Attention 的 W_q / W_k / W_v 调教成“往回看前面的区分词”，
才能分清后面到底该接 "清晰" 还是 "脆甜"。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


SENT_A = ["拍照", "用", "苹果", "很", "清晰"]
SENT_B = ["咬", "一口", "苹果", "很", "脆甜"]
VOCAB = ["拍照", "用", "苹果", "很", "清晰", "咬", "一口", "脆甜"]

WORD2ID = {w: i for i, w in enumerate(VOCAB)}
ID2WORD = {i: w for i, w in enumerate(VOCAB)}


def get_device() -> torch.device:
    """按 AGENTS.md 约定自动选择设备：cuda -> mps -> cpu。"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def encode(words: list[str], device: torch.device) -> torch.Tensor:
    return torch.tensor([WORD2ID[w] for w in words], dtype=torch.long, device=device)


class TinyLM(nn.Module):
    """最简迷你语言模型：词向量表 + 1层因果 Attention + 残差 + 输出打分头。"""

    def __init__(self, vocab_size: int, d_model: int = 16) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.embedding(token_ids)  # (T, d_model)
        Q, K, V = self.W_q(x), self.W_k(x), self.W_v(x)

        d = Q.shape[-1]
        scores = (Q @ K.T) / math.sqrt(d)
        mask = torch.triu(torch.ones_like(scores, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)

        # 残差连接：保留词自身原始信息 + 加上 Attention 融合的上下文信息
        h = x + weights @ V
        logits = self.lm_head(h)  # (T, vocab_size)
        return logits, weights


@torch.no_grad()
def inspect_prediction(
    model: TinyLM,
    prefix_words: list[str],
    device: torch.device,
) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
    """查看给定前缀时，模型预测下一个词的 Top-3 概率，以及末尾词的注意力分配。"""
    ids = encode(prefix_words, device)
    logits, weights = model(ids)
    probs = torch.softmax(logits[-1], dim=-1)

    top_vals, top_idxs = torch.topk(probs, k=3)
    top3 = [(ID2WORD[idx.item()], val.item()) for val, idx in zip(top_vals, top_idxs)]

    last_attn_row = weights[-1].detach().cpu().tolist()
    last_attn = [(word, float(w)) for word, w in zip(prefix_words, last_attn_row)]
    return top3, last_attn


@torch.no_grad()
def _target_prob(
    model: TinyLM,
    prefix_words: list[str],
    target_word: str,
    device: torch.device,
) -> float:
    ids = encode(prefix_words, device)
    logits, _ = model(ids)
    probs = torch.softmax(logits[-1], dim=-1)
    return probs[WORD2ID[target_word]].item()


@torch.no_grad()
def generate(
    model: TinyLM,
    start_word: str,
    steps: int,
    device: torch.device,
) -> list[str]:
    """只给第 1 个词，让模型自回归循环猜词接龙。"""
    ids = [WORD2ID[start_word]]
    for _ in range(steps):
        x = torch.tensor(ids, dtype=torch.long, device=device)
        logits, _ = model(x)
        next_id = int(logits[-1].argmax().item())
        ids.append(next_id)
    return [ID2WORD[i] for i in ids]


def _fmt_pairs(pairs: list[tuple[str, float]]) -> str:
    return ", ".join(f"{name}: {val * 100:.1f}%" for name, val in pairs)


def main() -> None:
    device = get_device()
    print(f"运行设备: {device}")

    torch.manual_seed(42)
    model = TinyLM(len(VOCAB), d_model=16).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.03)

    ids_a = encode(SENT_A, device)
    ids_b = encode(SENT_B, device)
    prefix_a = SENT_A[:-1]  # ["拍照", "用", "苹果", "很"] -> 期望预测 "清晰"
    prefix_b = SENT_B[:-1]  # ["咬", "一口", "苹果", "很"] -> 期望预测 "脆甜"

    w_q_before = model.W_q.weight[0, :4].detach().cpu().tolist()

    # --- 阶段 1：训练前（全是随机小数） ---
    print("\n=== 阶段 1：训练前（Step 0，权重全是随机数）===")
    print("W_q 第一行前 4 个数:", [round(v, 4) for v in w_q_before])
    top3_a0, attn_a0 = inspect_prediction(model, prefix_a, device)
    top3_b0, attn_b0 = inspect_prediction(model, prefix_b, device)
    print(f"输入 {prefix_a} -> Top3 猜测: [{_fmt_pairs(top3_a0)}]")
    print(f"  '很' 的注意力权重: [{_fmt_pairs(attn_a0)}]")
    print(f"输入 {prefix_b} -> Top3 猜测: [{_fmt_pairs(top3_b0)}]")
    print(f"  '很' 的注意力权重: [{_fmt_pairs(attn_b0)}]")

    # --- 阶段 2：最小训练循环（200 步） ---
    print("\n=== 阶段 2：开始训练（猜 -> 算 Loss -> 反向传播 -> 更新矩阵）===")
    loss = torch.tensor(0.0, device=device)
    for step in range(1, 201):
        logits_a, _ = model(ids_a[:-1])
        logits_b, _ = model(ids_b[:-1])
        loss = (
            F.cross_entropy(logits_a, ids_a[1:])
            + F.cross_entropy(logits_b, ids_b[1:])
        ) / 2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step in (1, 50, 100, 150, 200):
            p_clear = _target_prob(model, prefix_a, "清晰", device)
            p_crisp = _target_prob(model, prefix_b, "脆甜", device)
            print(
                f"  Step {step:3d} | Loss = {loss.item():.4f} | "
                f"句子A猜'清晰'概率 = {p_clear * 100:5.1f}% | "
                f"句子B猜'脆甜'概率 = {p_crisp * 100:5.1f}%"
            )

    # --- 阶段 3：训练后（验证权重变化、注意力分工与接龙效果） ---
    print("\n=== 阶段 3：训练后（Step 200）===")
    w_q_after = model.W_q.weight[0, :4].detach().cpu().tolist()
    print("W_q 第一行前 4 个数对比:")
    print("  训练前:", [round(v, 4) for v in w_q_before])
    print("  训练后:", [round(v, 4) for v in w_q_after])

    top3_a1, attn_a1 = inspect_prediction(model, prefix_a, device)
    top3_b1, attn_b1 = inspect_prediction(model, prefix_b, device)
    print(f"\n输入 {prefix_a} -> Top3 猜测: [{_fmt_pairs(top3_a1)}]")
    print(f"  '很' 自动学到的注意力: [{_fmt_pairs(attn_a1)}]")
    print(f"输入 {prefix_b} -> Top3 猜测: [{_fmt_pairs(top3_b1)}]")
    print(f"  '很' 自动学到的注意力: [{_fmt_pairs(attn_b1)}]")

    gen_a = generate(model, "拍照", steps=4, device=device)
    gen_b = generate(model, "咬", steps=4, device=device)
    print("\n=== 自回归接龙测试（只给 1 个开头词，连续猜 4 步）===")
    print(f"输入 ['拍照'] -> 模型接龙: {gen_a}")
    print(f"输入 ['咬']   -> 模型接龙: {gen_b}")

    assert loss.item() < 0.05, f"Expected loss < 0.05, got {loss.item()}"
    assert gen_a == SENT_A, f"Expected {SENT_A}, got {gen_a}"
    assert gen_b == SENT_B, f"Expected {SENT_B}, got {gen_b}"
    print("\n实验 B 全部验证通过！")


if __name__ == "__main__":
    main()
