"""完整 Transformer Block 教学实验：位置编码 + RMSNorm + 多头注意力 + FFN。

用 3 个直观演示验证每个零件缺一不可：
1. RMSNorm 稳压演示：把悬殊的向量 [100, -50, 2, 0.1] 拉回均匀尺度；
2. 位置编码生死对比："狗咬人很常见" vs "人咬狗很稀奇"
   - 关掉位置编码：两句话用词完全相同，模型彻底分不清谁咬谁（各猜 ~50%）；
   - 打开位置编码：立刻学会主宾语顺序，准确率 > 95%；
3. 多头注意力分工：看看 4 个头在 "很" 这个位置分别在关注前面哪些词。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


SENT_A = ["狗", "咬", "人", "很", "常见"]
SENT_B = ["人", "咬", "狗", "很", "稀奇"]
VOCAB = ["狗", "咬", "人", "很", "常见", "稀奇"]

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


class RMSNorm(nn.Module):
    """均方根归一化（稳压器）：把每个词向量的数值尺度拉稳，防止层数多了爆炸或归零。"""

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


class MultiHeadAttention(nn.Module):
    """多头因果注意力：把 d_model 拆成 n_heads 组小通道，各自独立关注不同线索再拼回。"""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        T, D = x.shape
        # (T, D) -> (T, n_heads, head_dim) -> (n_heads, T, head_dim)
        Q = self.W_q(x).view(T, self.n_heads, self.head_dim).transpose(0, 1)
        K = self.W_k(x).view(T, self.n_heads, self.head_dim).transpose(0, 1)
        V = self.W_v(x).view(T, self.n_heads, self.head_dim).transpose(0, 1)

        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)  # (n_heads, T, T)

        out = (weights @ V).transpose(0, 1).contiguous().view(T, D)
        return self.W_o(out), weights


class FeedForward(nn.Module):
    """前馈网络（独立思考层）：先升维展开思考，非线性激活筛选，再压缩回原维度。"""

    def __init__(self, d_model: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """一块完整现代 Pre-Norm Transformer 积木：Norm1+MHA+残差 -> Norm2+FFN+残差。"""

    def __init__(self, d_model: int, n_heads: int, hidden_dim: int) -> None:
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.norm2 = RMSNorm(d_model)
        self.ffn = FeedForward(d_model, hidden_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        attn_out, weights = self.attn(self.norm1(x))
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x, weights


class MiniTransformerLM(nn.Module):
    """包含词向量 + 位置编码 + 1 块 TransformerBlock + 最终归一化 + 输出层的迷你模型。"""

    def __init__(
        self,
        vocab_size: int,
        max_len: int = 8,
        d_model: int = 16,
        n_heads: int = 4,
        use_pos_emb: bool = True,
    ) -> None:
        super().__init__()
        self.use_pos_emb = use_pos_emb
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model) if use_pos_emb else None
        self.block = TransformerBlock(d_model=d_model, n_heads=n_heads, hidden_dim=d_model * 4)
        self.final_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.tok_emb(token_ids)
        if self.pos_emb is not None:
            positions = torch.arange(token_ids.shape[0], device=token_ids.device)
            x = x + self.pos_emb(positions)
        x, weights = self.block(x)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits, weights


def demo_1_rmsnorm(device: torch.device) -> None:
    print("=== 演示 1：RMSNorm 稳压器效果 ===")
    norm = RMSNorm(4).to(device)
    raw = torch.tensor([100.0, -50.0, 2.0, 0.1], device=device)
    out = norm(raw)

    raw_rms = torch.sqrt(raw.pow(2).mean()).item()
    out_rms = torch.sqrt(out.pow(2).mean()).item()
    print("归一化前向量:", [round(v, 3) for v in raw.detach().cpu().tolist()], f"| RMS = {raw_rms:.2f}")
    print("归一化后向量:", [round(v, 3) for v in out.detach().cpu().tolist()], f"| RMS = {out_rms:.2f}")
    assert abs(out_rms - 1.0) < 1e-4


def train_and_eval(use_pos_emb: bool, device: torch.device) -> tuple[MiniTransformerLM, float, torch.Tensor, torch.Tensor]:
    torch.manual_seed(42)
    model = MiniTransformerLM(
        vocab_size=len(VOCAB),
        max_len=8,
        d_model=16,
        n_heads=4,
        use_pos_emb=use_pos_emb,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)

    ids_a = encode(SENT_A, device)
    ids_b = encode(SENT_B, device)

    loss = torch.tensor(0.0, device=device)
    for _ in range(200):
        logits_a, _ = model(ids_a[:-1])
        logits_b, _ = model(ids_b[:-1])
        loss = (
            F.cross_entropy(logits_a, ids_a[1:])
            + F.cross_entropy(logits_b, ids_b[1:])
        ) / 2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        probs_a = torch.softmax(model(ids_a[:-1])[0][-1], dim=-1)
        probs_b = torch.softmax(model(ids_b[:-1])[0][-1], dim=-1)

    return model, float(loss.item()), probs_a, probs_b


def demo_2_positional_encoding(device: torch.device) -> MiniTransformerLM:
    print("\n=== 演示 2：位置编码生死对比（'狗咬人很常见' vs '人咬狗很稀奇'）===")
    p_idx_common = WORD2ID["常见"]
    p_idx_rare = WORD2ID["稀奇"]

    _, loss_no, pa_no, pb_no = train_and_eval(use_pos_emb=False, device=device)
    print(f"[关闭位置编码] 训练 200 步最终 Loss = {loss_no:.4f}")
    print(
        f"  输入 ['狗', '咬', '人', '很'] -> 猜'常见': {pa_no[p_idx_common] * 100:5.1f}%, "
        f"猜'稀奇': {pa_no[p_idx_rare] * 100:5.1f}%"
    )
    print(
        f"  输入 ['人', '咬', '狗', '很'] -> 猜'常见': {pb_no[p_idx_common] * 100:5.1f}%, "
        f"猜'稀奇': {pb_no[p_idx_rare] * 100:5.1f}%  <- 彻底分不清！"
    )
    assert torch.allclose(pa_no, pb_no, atol=1e-4)

    model_yes, loss_yes, pa_yes, pb_yes = train_and_eval(use_pos_emb=True, device=device)
    print(f"\n[开启位置编码] 训练 200 步最终 Loss = {loss_yes:.4f}")
    print(
        f"  输入 ['狗', '咬', '人', '很'] -> 猜'常见': {pa_yes[p_idx_common] * 100:5.1f}%, "
        f"猜'稀奇': {pa_yes[p_idx_rare] * 100:5.1f}%"
    )
    print(
        f"  输入 ['人', '咬', '狗', '很'] -> 猜'常见': {pb_yes[p_idx_common] * 100:5.1f}%, "
        f"猜'稀奇': {pb_yes[p_idx_rare] * 100:5.1f}%  <- 瞬间分清主宾语！"
    )
    assert loss_yes < 0.05, f"Expected loss < 0.05, got {loss_yes}"
    assert pa_yes[p_idx_common].item() > 0.95
    assert pb_yes[p_idx_rare].item() > 0.95
    return model_yes


@torch.no_grad()
def demo_3_multi_head_roles(model: MiniTransformerLM, device: torch.device) -> None:
    print("\n=== 演示 3：多头注意力（4 个头在 '很' 位置各自的分工）===")
    prefix = ["狗", "咬", "人", "很"]
    _, weights = model(encode(prefix, device))
    print(f"注意力权重张量 shape: {tuple(weights.shape)}  (4 个头 × 4 个Query × 4 个Key)")
    for head_idx in range(weights.shape[0]):
        last_row = weights[head_idx, -1].detach().cpu().tolist()
        pairs = ", ".join(f"{w}: {val * 100:5.1f}%" for w, val in zip(prefix, last_row))
        print(f"  Head {head_idx}: [{pairs}]")


def main() -> None:
    device = get_device()
    print(f"运行设备: {device}")
    demo_1_rmsnorm(device)
    trained_model = demo_2_positional_encoding(device)
    demo_3_multi_head_roles(trained_model, device)
    print("\n完整 Transformer Block 验证通过！")


if __name__ == "__main__":
    main()
