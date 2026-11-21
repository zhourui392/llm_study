package com.study.llm;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.bgesmallzhv15q.BgeSmallZhV15QuantizedEmbeddingModel;
import dev.langchain4j.model.ollama.OllamaChatModel;
import dev.langchain4j.store.embedding.EmbeddingMatch;
import dev.langchain4j.store.embedding.EmbeddingSearchRequest;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.pgvector.PgVectorEmbeddingStore;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.List;
import java.util.stream.Collectors;

/**
 * 最小 Java RAG 全链路演示：
 * 本地知识文档 -> 切分 -> 进程内 ONNX 中文向量模型 -> PGVector 入库
 * 用户提问 -> 向量化 -> PGVector 相似度检索 -> 拼装 Prompt -> 本地 Ollama 大模型回答
 */
public class MinimalRagDemo {

    private static final String OLLAMA_BASE_URL = "http://localhost:11434";
    private static final String OLLAMA_MODEL = System.getProperty("ollama.model", "qwen2.5:0.5b");

    public static void main(String[] args) throws IOException {
        boolean retrieveOnly = Boolean.getBoolean("rag.retrieveOnly");
        System.out.println("=== 1. 初始化组件（Embedding 模型 + PGVector + Ollama Chat 模型） ===");

        // 进程内 ONNX 中文向量模型（bge-small-zh-v1.5 量化版，维度 512）
        EmbeddingModel embeddingModel = new BgeSmallZhV15QuantizedEmbeddingModel();
        System.out.println("Embedding 模型维度: " + embeddingModel.dimension());

        // PGVector 向量数据库（首次运行自动建表，dropTableFirst=true 方便重复演示不产生脏数据）
        EmbeddingStore<TextSegment> embeddingStore = PgVectorEmbeddingStore.builder()
                .host("localhost")
                .port(5432)
                .database("rag_demo")
                .user("postgres")
                .password("postgres")
                .table("company_handbook_vectors")
                .dimension(embeddingModel.dimension())
                // 数据量少时关闭 IVFFlat 索引做精确检索；生产环境数据量大时可改为 .useIndex(true).indexListSize(100)
                .useIndex(false)
                .createTable(true)
                .dropTableFirst(true)
                .build();

        // 本地 Ollama 对话模型（加入 repeatPenalty 和 numPredict 防止小参数模型复读循环）
        ChatModel chatModel = retrieveOnly ? null : OllamaChatModel.builder()
                .baseUrl(OLLAMA_BASE_URL)
                .modelName(OLLAMA_MODEL)
                .temperature(0.3)
                .repeatPenalty(1.2)
                .numPredict(256)
                .timeout(Duration.ofSeconds(120))
                .build();

        System.out.println("\n=== 2. 离线入库阶段：读取文档 -> 切分 Chunk -> 计算向量 -> 存入 PGVector ===");
        String rawText = loadResourceText("/company-handbook.txt");
        Document document = Document.from(rawText);

        // 递归切分文档：每块最多 180 字符，块间重叠 30 字符（避免一句话从中间切断丢失语义）
        List<TextSegment> segments = DocumentSplitters.recursive(180, 30).split(document);
        System.out.println("文档切分完成，共 " + segments.size() + " 个片段：");
        for (int i = 0; i < segments.size(); i++) {
            System.out.printf("  [Chunk %d] %s%n", i + 1, segments.get(i).text().replace("\n", " "));
        }

        // 批量计算所有片段的向量，并写入 PGVector
        List<Embedding> embeddings = embeddingModel.embedAll(segments).content();
        embeddingStore.addAll(embeddings, segments);
        System.out.println("已将 " + embeddings.size() + " 条向量及原文写入 PGVector。");

        System.out.println("\n=== 3. 在线问答阶段：语义检索 Top-2 + 拼装 Prompt + 调用 Ollama ===");
        List<String> questions = List.of(
                "我想请 5 天年假，需要提前几天申请？谁来审批？",
                "晚上加班到 9 点多，有餐补吗？怎么报销？",
                "周五下午 4 点可以发布普通业务系统吗？"
        );

        for (String question : questions) {
            System.out.println("\n--------------------------------------------------");
            System.out.println("【用户提问】: " + question);

            // 3.1 用同一个 Embedding 模型把问题转成向量
            Embedding queryEmbedding = embeddingModel.embed(question).content();

            // 3.2 去 PGVector 检索相似度最高的 Top-2 片段（过滤掉相关性较低的噪声片段）
            EmbeddingSearchRequest searchRequest = EmbeddingSearchRequest.builder()
                    .queryEmbedding(queryEmbedding)
                    .maxResults(2)
                    .minScore(0.78)
                    .build();
            List<EmbeddingMatch<TextSegment>> matches = embeddingStore.search(searchRequest).matches();

            System.out.println("【PGVector 召回片段】:");
            for (int i = 0; i < matches.size(); i++) {
                EmbeddingMatch<TextSegment> match = matches.get(i);
                System.out.printf("  Top-%d (相似度=%.4f): %s%n",
                        i + 1,
                        match.score(),
                        match.embedded().text().replace("\n", " "));
            }

            // 3.3 把召回的参考片段和用户问题拼成最终 Prompt
            String context = matches.stream()
                    .map(m -> m.embedded().text())
                    .collect(Collectors.joining("\n---\n"));

            String systemPrompt = "你是星云科技内部助手，请严格根据提供的参考资料用中文简明回答问题。";
            String userPrompt = """
                    参考资料：
                    %s

                    问题：%s
                    请直接给出答案并说明依据。""".formatted(context, question);

            // 3.4 调用本地 Ollama 模型生成最终回答
            if (chatModel == null) {
                System.out.println("【检索验证模式】已跳过 Ollama 调用，组装好的 Prompt 长度: " + userPrompt.length());
            } else {
                String answer = chatModel.chat(
                        SystemMessage.from(systemPrompt),
                        UserMessage.from(userPrompt)
                ).aiMessage().text();
                System.out.println("【大模型最终回答】:\n" + answer);
            }
        }
    }

    private static String loadResourceText(String resourcePath) throws IOException {
        try (InputStream is = MinimalRagDemo.class.getResourceAsStream(resourcePath)) {
            if (is == null) {
                throw new IOException("找不到资源文件: " + resourcePath);
            }
            return new String(is.readAllBytes(), StandardCharsets.UTF_8);
        }
    }
}
