"""极简 Attention（注意力机制）教学实验。

用 3 个词、2 维语义向量 [科技/手机属性, 食物/水果属性]，亲眼看清：
1. Q / K / V 是如何算出注意力权重矩阵的；
2. Causal Mask（因果遮罩）如何防止模型“偷看后面的词”；
3. 同一个“苹果”，如何根据上下文融合出截然不同的新向量；
4. 真实 LLM 里的 Q/K/V 是怎么通过可学习矩阵变出来的。
"""

from __future__ import annotations

import math
import unicodedata

import torch
import torch.nn as nn


def get_device() -> torch.device:
    """按 AGENTS.md 约定自动选择设备：cuda -> mps -> cpu。"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    causal: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """核心 Scaled Dot-Product Attention（缩放点积注意力）。"""
    d = Q.shape[-1]
    scores = (Q @ K.T) / math.sqrt(d)
    if causal:
        mask = torch.triu(torch.ones_like(scores, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    output = weights @ V
    return output, weights


def _pad_display(text: str, width: int) -> str:
    """按终端显示宽度（中文占 2 格）右对齐/补齐空格，让表格对齐整齐。"""
    disp = sum(2 if unicodedata.east_asian_width(c) in ("W", "F") else 1 for c in text)
    return " " * max(width - disp, 0) + text


def print_weight_matrix(words: list[str], weights: torch.Tensor, title: str) -> None:
    """把 3x3 注意力权重矩阵整齐地打印出来。"""
    print(f"\n[{title}]")
    w_cpu = weights.detach().cpu()
    col_w = 9
    header = _pad_display("Q \\ K", 8) + " | " + " | ".join(_pad_display(w, col_w) for w in words)
    print(header)
    print("-" * len(header))
    for i, q_word in enumerate(words):
        row_cells = " | ".join(f"{w_cpu[i, j].item():>{col_w}.2f}" for j in range(len(words)))
        print(f"{_pad_display(q_word, 8)} | {row_cells}")


def demo_apple_in_two_contexts(device: torch.device) -> None:
    print("语义维度约定: [科技/手机, 食物/水果]")

    # --- 演示 1：句子 A ["拍照", "用", "苹果"] ---
    words_a = ["拍照", "用", "苹果"]
    K_a = torch.tensor([[2.0, 0.0], [0.0, 0.0], [0.5, 0.5]], device=device)
    Q_a = torch.tensor([[1.5, 0.0], [0.5, 0.5], [1.5, 1.5]], device=device)
    V_a = torch.tensor([[1.0, 0.0], [0.0, 0.0], [0.5, 0.5]], device=device)

    out_a_bi, w_a_bi = attention(Q_a, K_a, V_a, causal=False)
    out_a_causal, w_a_causal = attention(Q_a, K_a, V_a, causal=True)

    print("\n" + "=" * 54)
    print("演示 1：句子 A = ['拍照', '用', '苹果']")
    print_weight_matrix(words_a, w_a_bi, "双向注意力 (causal=False，能看到整句话)")
    print_weight_matrix(words_a, w_a_causal, "因果注意力 (causal=True，只能看当前和前面，右上角为 0)")

    apple_before = V_a[2].detach().cpu().tolist()
    apple_after_a = out_a_causal[2].detach().cpu().tolist()
    print(f"\n句子 A 中 '苹果' 初始值 V: {[round(x, 3) for x in apple_before]}")
    print(
        f"句子 A 中 '苹果' 融合后  : {[round(x, 3) for x in apple_after_a]}  "
        "<- 科技/手机维度明显升高！"
    )

    # --- 演示 2：句子 B ["咬", "一口", "苹果"] ---
    # 注意："苹果" 自身的 Q / K / V 和句子 A 完全一模一样！
    words_b = ["咬", "一口", "苹果"]
    K_b = torch.tensor([[0.0, 2.0], [0.0, 1.5], [0.5, 0.5]], device=device)
    Q_b = torch.tensor([[0.0, 1.5], [0.0, 1.5], [1.5, 1.5]], device=device)
    V_b = torch.tensor([[0.0, 1.0], [0.0, 0.8], [0.5, 0.5]], device=device)

    out_b_causal, w_b_causal = attention(Q_b, K_b, V_b, causal=True)

    print("\n" + "=" * 54)
    print("演示 2：句子 B = ['咬', '一口', '苹果']")
    print_weight_matrix(words_b, w_b_causal, "因果注意力 (causal=True)")

    apple_after_b = out_b_causal[2].detach().cpu().tolist()
    print(f"\n句子 B 中 '苹果' 初始值 V: {[round(x, 3) for x in apple_before]}  (与句子 A 完全相同)")
    print(
        f"句子 B 中 '苹果' 融合后  : {[round(x, 3) for x in apple_after_b]}  "
        "<- 食物/水果维度明显升高！"
    )

    assert torch.allclose(w_a_causal.sum(dim=-1), torch.ones(3, device=device), atol=1e-5)
    assert torch.allclose(w_b_causal.sum(dim=-1), torch.ones(3, device=device), atol=1e-5)
    assert out_a_causal[2, 0] > out_a_causal[2, 1]
    assert out_b_causal[2, 1] > out_b_causal[2, 0]


def demo_learned_qkv_projection(device: torch.device) -> None:
    print("\n" + "=" * 54)
    print("演示 3：真实 LLM 里 Q / K / V 从哪来？（不是人手写的，是训练学出来的）")
    torch.manual_seed(42)

    seq_len, d_model = 3, 4
    X = torch.randn(seq_len, d_model, device=device)

    W_q = nn.Linear(d_model, d_model, bias=False).to(device)
    W_k = nn.Linear(d_model, d_model, bias=False).to(device)
    W_v = nn.Linear(d_model, d_model, bias=False).to(device)

    Q = W_q(X)
    K = W_k(X)
    V = W_v(X)
    output, weights = attention(Q, K, V, causal=True)

    print(f"输入词向量 X shape: {tuple(X.shape)}  (3 个词，每个词 4 个小数)")
    print(f"Q / K / V  shape: {tuple(Q.shape)} / {tuple(K.shape)} / {tuple(V.shape)}")
    print(f"输出向量   shape: {tuple(output.shape)}  (融合上下文后的新词向量)")
    print(f"权重矩阵   shape: {tuple(weights.shape)}  (3x3 因果权重矩阵)")

    assert not torch.isnan(output).any()
    assert not torch.isnan(weights).any()


def main() -> None:
    device = get_device()
    print(f"运行设备: {device}")
    demo_apple_in_two_contexts(device)
    demo_learned_qkv_projection(device)
    print("\n实验完成！")


if __name__ == "__main__":
    main()
