# 基于通义千问的 Word 文档 RAG 检索问答机器人

## 项目简介
本项目是一个基于自己本地 Word 文档的 RAG（检索增强生成）智能问答机器人。通过 LangChain 框架，结合通义千问的文本嵌入模型（text-embedding-v3）和回答模型（qwen-plus），实现了对本地专利文档的精准检索与问答。

## 技术栈
- Python 3.10+
- LangChain
- 通义千问 (DashScope API)
- Chroma 向量数据库
- PyCharm

## 核心功能
1. 解析本地 Word 文档并自动切分片段。
2. 调用通义千问 Embedding API 向量化并存入 Chroma。
3. 基于用户提问，检索最相关的 3 个片段并生成回答。
4. 严格限制模型基于文档回答，防止幻觉。

## 如何运行
1. 安装依赖：`pip install langchain langchain-community langchain-chroma langchain-core
            pip install docx2txt python-dotenv dashscope openai chromadb` 
2. 在根目录创建 `KEY.env`，写入 `OPENAI_API_KEY=你的密钥`
3. 运行 `RAG检索本地文档.py`