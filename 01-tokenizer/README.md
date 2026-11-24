# 01 - 从零手写 Byte-level BPE 分词器

现代 LLM（GPT-2 / GPT-4 / Llama 3 / Qwen）几乎统一采用 **Byte-level BPE（字节级字节对编码）**。本阶段全程纯本地实现并验证一个完整的 Byte-level BPE 分词器，为后续的架构搭建与预训练准备输入管线。

---

## 1. 为什么是 Byte-level BPE？

| 方案 | 词表规模 | 序列长度 | 核心缺陷 / 优势 |
|---|---|---|---|
| **词级（Word-level）** | 极大（数十万+） | 最短 | 无法处理未登录词（`<UNK>`）、词形变化与拼写错误；Embedding 矩阵爆炸 |
| **字符级（Char-level）** | 极小（几十～几百） | 极长 | 注意力计算是 $O(T^2)$，极长序列严重压缩上下文窗口；跨语言字符集依然膨胀 |
| **Byte-level BPE** | **可控（32k～128k）** | **高压缩比** | 以 **256 个原始字节**作基表：**永远没有 `<UNK>`**，任何 UTF-8 文本/代码/Emoji 都能无损编码 |

---

## 2. 核心机制拆解

### 2.1 256 字节基表
任何字符串先通过 `.encode("utf-8")` 转为 `0..255` 的字节序列，直接对应 ID `0..255`。例如英文 ASCII 占 1 字节，常见汉字占 3 字节，Emoji 占 4 字节。

### 2.2 预分词正则（Regex Pre-tokenization）
如果不做预分词直接跑 BPE，高频标点或换行会和单词粘在一起（比如 `dog.`、`dog!`、`dog?` 被学成三个独立 token），浪费词表且破坏语义泛化。

GPT-4（`cl100k_base`）使用的预分词正则 `GPT4_SPLIT_PATTERN` 做了几件关键的事：
```text
'(?i:[sdmt]|ll|ve|re)         # 英文缩写后缀（大小写不敏感）：'s, 't, 'll, 've ...
| [^\r\n\p{L}\p{N}]?+\p{L}+   # 字母词（可选带 1 个前导非换行符号，如前导空格 " hello"）
| \p{N}{1,3}                  # 数字按 1~3 位切分（防止超大数字学出一堆稀疏整串 token）
|  ?[^\s\p{L}\p{N}]++[\r\n]*  # 标点符号块及紧随的换行
| \s*[\r\n]                   # 含前导空白的换行符
| \s+(?!\S)                   # 末尾或连续缩进空白（保留代码缩进结构）
| \s+                         # 其他空白
```
先用正则切块、**禁止跨块合并**，再把相同文本块去重计数后训练 BPE：
- 消除了跨词/标点粘连；
- 把训练复杂度从“全量语料长度”降到“去重后唯一词块数量”；
- 结合 `pair_to_words` 倒排索引做增量更新，避免每轮全量重扫。

### 2.3 按 Merge Rank 编码（Encoding）
训练阶段学到的是有序合并表 `(left_id, right_id) -> new_id`（`new_id` 从 `256` 递增，**ID 越小代表训练时合并越早、优先级 Rank 越高**）。
编码某个 chunk 时：
1. 初始拆成 UTF-8 字节列表；
2. 不断找出当前相邻 pair 中 **`rank` 最小** 的那一对执行合并；
3. 直到没有任何相邻 pair 出现在 `merges` 中为止。

### 2.4 安全解码（Decoding）与特殊标记（Special Tokens）
- **解码**：把每个 token ID 查表拼回 `bytes`，再用 `.decode("utf-8", errors="replace")` 转字符串。这样即使生成中途被截断（例如只输出了 3 字节汉字的前 2 个字节），也会安全渲染为替换符 ``（`U+FFFD`）而不会崩溃。
- **Special Tokens**：如 `<|endoftext|>`、`<|user|>`、`<|assistant|>`，它们**不走 BPE 字节合并**，而是在编解码前通过正则切分直接映射成固定高 ID；匹配时按长度降序排列，防止短前缀抢先截断长标记（如 `<|end|>` 截断 `<|endoftext|>`）。此外训练完成后连续分配 special token ID，避免词表出现空洞导致 `nn.Embedding` 浪费行。

---

## 3. 关键工程洞见

1. **词表大小权衡（Vocab Size Trade-off）**：
   - 词表越大 → 单条文本生成的 token 数越少（上下文更长、推理步数更少），但输入输出 Embedding 参数量 $V \times d$ 与最终 Softmax 计算开销线性增大。
   - 小模型（<100M）常用较小词表（如 1k～32k），现代大模型因计算瓶颈多在主干层且要兼容多语言/代码，逐步扩至 100k～150k+。
2. **跨语种“词元税”（Fertility）**：
   - 仅在英文语料上训练的 BPE 遇到中文时，因为中文 3 字节片段未被高频合并，1 个汉字往往退化成 3 个字节 token（压缩率约 1 B/token）；而在多语种充分训练的大词表（如 `cl100k_base`、Qwen）下，常用汉字和词组会被合并成 1～2 个 token，显著降低推理成本。
3. **数字切分影响算术能力**：
   - GPT-2 允许任意长度数字（`\p{N}+`）参与合并，导致 `1234` 与 `123` 被切得毫无规律；GPT-4 强制 `\p{N}{1,3}` 把连续数字规范地切成 1～3 位组，显著改善模型对多位数运算的泛化。

---

## 4. 文件说明与本地运行

- [bpe.py](./bpe.py)：`BPETokenizer` 核心实现（含 `GPT2_SPLIT_PATTERN`、`GPT4_SPLIT_PATTERN`、增量训练、Rank 编码、Special Tokens、JSON 持久化）。
- [test_bpe.py](./test_bpe.py)：单元测试（重叠合并、Unicode/Emoji 往返、增量训练与暴力参考实现严格等价性、边界隔离、特殊标记策略、非法 UTF-8 容错与存取往返）。
- [experiment.py](./experiment.py)：本地对比实验脚本（玩具示例逐步合并、Basic BPE vs Regex BPE、英文/中文/代码/数字与 `tiktoken` 对比）。

```bash
# 运行单元测试
uv run pytest 01-tokenizer/test_bpe.py -v

# 运行对比实验
uv run python 01-tokenizer/experiment.py
```
