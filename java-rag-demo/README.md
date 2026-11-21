# Java 最小 RAG 全链路 Demo

这是一个纯离线、零云端 API 依赖的 Java 最小 RAG 示例项目，使用 **LangChain4j + 本地 ONNX 中文向量模型 + PGVector + Ollama** 跑通完整的检索增强生成流程。

## 组件架构

| 角色 | 技术选型 | 说明 |
| :--- | :--- | :--- |
| 编排框架 | LangChain4j `1.23.0` | 文档切分、向量存储接口、Ollama 对接 |
| 向量模型 (Embedding) | `BgeSmallZhV15QuantizedEmbeddingModel` | 进程内 ONNX 中文模型（512 维） |
| 向量数据库 | PostgreSQL + `pgvector` (Docker) | 存储文本片段及向量，支持余弦相似度检索 |
| 对话大模型 (Chat LLM) | Ollama (`qwen2.5:0.5b`) | 本机运行的开源中文轻量模型（可替换为更大参数版本） |

## 启动与运行步骤

1. **启动 PGVector 数据库**
   ```bash
   docker compose up -d
   ```
2. **启动本地 Ollama 并确保模型已拉取**
   ```bash
   ollama serve
   ollama pull qwen2.5:0.5b
   ```
3. **使用 JDK 21 运行 Demo**
   ```bash
   JAVA_HOME=/Users/vv/jdk/jdk-21.0.12+8/Contents/Home mvn compile exec:java
   ```
