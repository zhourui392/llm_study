# LLM 训练学习

## 技术栈

- **PyTorch（MPS 后端）**：主力训练框架，设备用 `torch.device("mps")`，代码后续可零改动迁移到 CUDA。
