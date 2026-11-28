"""Embedding 层核心机制 PyTorch 验证实验。

包含 4 个小实验：
1. nn.Embedding 查表与 One-hot 矩阵乘法的数值等价性；
2. 稀疏梯度更新与上下游联合反向传播；
3. 随机初始化向量在 Next-Token Prediction 训练中自动聚拢语义；
4. Weight Tying（入口 Embedding 与出口 LM Head 权重共享）。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_device() -> torch.device:
    """按 AGENTS.md 约定自动选择设备：cuda -> mps -> cpu。"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def exp1_lookup_vs_onehot(device: torch.device) -> None:
    print("=== Experiment 1: nn.Embedding 查表 vs One-hot 矩阵乘法 ===")
    torch.manual_seed(42)
    vocab_size, d_model = 6, 4
    emb = nn.Embedding(vocab_size, d_model).to(device)
    ids = torch.tensor([0, 3, 1], device=device)

    lookup_out = emb(ids)
    onehot = F.one_hot(ids, num_classes=vocab_size).float()
    matmul_out = onehot @ emb.weight

    max_diff = (lookup_out - matmul_out).abs().max().item()
    assert torch.allclose(lookup_out, matmul_out, atol=1e-6), f"Mismatch: {max_diff}"
    print(f"  lookup_out shape={tuple(lookup_out.shape)}, matmul_out shape={tuple(matmul_out.shape)}")
    print(f"  max_abs_diff = {max_diff:.2e} (严格等价)")


def exp2_sparse_grad_and_joint_backprop(device: torch.device) -> None:
    print("\n=== Experiment 2: 稀疏梯度与 Embedding/线性层联合反向传播 ===")
    torch.manual_seed(42)
    vocab_size, d_model = 6, 4
    emb = nn.Embedding(vocab_size, d_model).to(device)
    proj = nn.Linear(d_model, vocab_size, bias=False).to(device)

    ids = torch.tensor([0, 2], device=device)
    targets = torch.tensor([2, 4], device=device)

    logits = proj(emb(ids))
    loss = F.cross_entropy(logits, targets)
    loss.backward()

    assert emb.weight.grad is not None and proj.weight.grad is not None
    row_norms = emb.weight.grad.norm(dim=-1).tolist()
    for unused_idx in (1, 3, 4, 5):
        assert torch.all(emb.weight.grad[unused_idx] == 0.0), f"Row {unused_idx} should be zero"
    for used_idx in (0, 2):
        assert row_norms[used_idx] > 0.0, f"Row {used_idx} should be non-zero"
    proj_grad_norm = proj.weight.grad.norm().item()
    assert proj_grad_norm > 0.0

    print(f"  输入 Token ID: {ids.tolist()}, loss = {loss.item():.4f}")
    for idx, norm in enumerate(row_norms):
        tag = "已使用 -> 收到梯度" if idx in (0, 2) else "未出现 -> 梯度严格为 0"
        print(f"    emb.weight.grad[{idx}] L2 norm = {norm:.6f} ({tag})")
    print(f"  proj.weight.grad L2 norm = {proj_grad_norm:.6f} (同步获得梯度)")


class TinyNextTokenModel(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, tie_weights: bool = False) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_weights:
            self.lm_head.weight = self.emb.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.emb(x))


def _cosine(weight: torch.Tensor, idx_a: int, idx_b: int) -> float:
    return F.cosine_similarity(weight[idx_a], weight[idx_b], dim=0).item()


def exp3_semantic_clustering(device: torch.device) -> None:
    print("\n=== Experiment 3: 随机初始化在 Next-Token 训练中自动聚拢相似词 ===")
    torch.manual_seed(42)
    sentences = [
        ["我", "养", "猫", "很", "可爱"],
        ["我", "养", "狗", "很", "可爱"],
        ["小", "猫", "爱", "玩", "球"],
        ["小", "狗", "爱", "玩", "球"],
        ["他", "开", "汽车", "飞", "快"],
        ["他", "开", "火箭", "飞", "快"],
        ["大", "汽车", "能", "跑", "远"],
        ["大", "火箭", "能", "跑", "远"],
    ]
    vocab = sorted({tok for sent in sentences for tok in sent})
    stoi = {tok: i for i, tok in enumerate(vocab)}

    inputs_list: list[int] = []
    targets_list: list[int] = []
    for sent in sentences:
        for a, b in zip(sent, sent[1:]):
            inputs_list.append(stoi[a])
            targets_list.append(stoi[b])

    x = torch.tensor(inputs_list, device=device)
    y = torch.tensor(targets_list, device=device)

    model = TinyNextTokenModel(vocab_size=len(vocab), d_model=4, tie_weights=False).to(device)
    pairs = [("猫", "狗"), ("汽车", "火箭"), ("猫", "汽车"), ("狗", "火箭")]

    before_cos = {pair: _cosine(model.emb.weight.detach(), stoi[pair[0]], stoi[pair[1]]) for pair in pairs}

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.05, weight_decay=0.05)
    init_loss = 0.0
    final_loss = 0.0
    for step in range(500):
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        if step == 0:
            init_loss = loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        final_loss = loss.item()

    after_cos = {pair: _cosine(model.emb.weight.detach(), stoi[pair[0]], stoi[pair[1]]) for pair in pairs}

    print(f"  词表大小 V={len(vocab)}, 训练样本对={len(inputs_list)}, loss: {init_loss:.4f} -> {final_loss:.4f}")
    print("  余弦相似度对比 (训练前 -> 训练后):")
    for a, b in pairs:
        print(f"    cos({a}, {b}) : {before_cos[(a, b)]:+.4f} -> {after_cos[(a, b)]:+.4f}")

    assert after_cos[("猫", "狗")] > 0.8
    assert after_cos[("汽车", "火箭")] > 0.8
    assert after_cos[("猫", "狗")] > max(after_cos[("猫", "汽车")], after_cos[("狗", "火箭")]) + 0.5
    assert after_cos[("汽车", "火箭")] > max(after_cos[("猫", "汽车")], after_cos[("狗", "火箭")]) + 0.5


def exp4_weight_tying(device: torch.device) -> None:
    print("\n=== Experiment 4: Weight Tying 权重共享验证 ===")
    torch.manual_seed(42)
    vocab_size, d_model = 1000, 128
    untied = TinyNextTokenModel(vocab_size=vocab_size, d_model=d_model, tie_weights=False).to(device)
    tied = TinyNextTokenModel(vocab_size=vocab_size, d_model=d_model, tie_weights=True).to(device)

    assert tied.emb.weight.data_ptr() == tied.lm_head.weight.data_ptr()
    assert untied.emb.weight.data_ptr() != untied.lm_head.weight.data_ptr()

    untied_params = sum(p.numel() for p in untied.parameters())
    # nn.Module.parameters() deduplicates shared Parameter objects automatically
    tied_params = sum(p.numel() for p in tied.parameters())
    saved = untied_params - tied_params
    assert saved == vocab_size * d_model

    print(f"  配置: V={vocab_size}, d={d_model}")
    print(f"  不共享权重参数量 : {untied_params:,} (data_ptr 不同)")
    print(f"  Weight Tying 参数量: {tied_params:,} (emb 与 lm_head 共用同一底层存储)")
    print(f"  节省参数量         : {saved:,} (= V × d = {vocab_size} × {d_model})")


def main() -> None:
    device = get_device()
    print(f"Running Embedding experiments on device: {device}")
    exp1_lookup_vs_onehot(device)
    exp2_sparse_grad_and_joint_backprop(device)
    exp3_semantic_clustering(device)
    exp4_weight_tying(device)
    print("\nAll Embedding assertions passed!")


if __name__ == "__main__":
    main()
