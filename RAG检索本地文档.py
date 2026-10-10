import os
from dotenv import load_dotenv
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv("KEY.env")
# 1. 加载并切分文档
doc_path = r"C:\Users\FanChong\Desktop\专利申请文档\一种靶向TIMD-4抗体及其在制备治疗动脉粥样硬化药物中的应用-定稿.docx"
loader = Docx2txtLoader(doc_path)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=50,
    separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
)
splits = text_splitter.split_documents(docs)
# 2. 向量化并存储
embeddings = DashScopeEmbeddings(
    model="text-embedding-v3",
    dashscope_api_key=os.environ["OPENAI_API_KEY"]
)
vectorstore = Chroma.from_documents(
    documents=splits, embedding=embeddings,
    persist_directory="./chroma_db"
)
# 7. 创建检索器
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  # 检索最相似的3个片段
# 8. 初始化通义千问回答模型（通过OpenAI兼容接口）
llm = ChatOpenAI(
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen3.8-flash",
    api_key=os.environ["OPENAI_API_KEY"],
    temperature=0.1  # 降低随机性，让回答更贴合文档
)
prompt = ChatPromptTemplate.from_template("""
你是一个基于文档的问答助手。请严格根据以下提供的文档片段回答问题。
如果文档中没有相关信息，请直接回答“文档中未找到相关内容”，不要编造。

文档片段：
{context}

用户问题：{question}

请用中文回答：
""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt | llm | StrOutputParser()
)

# 4. 交互问答
while True:
    question = input("\n请输入你的问题（输入 q 退出）：")
    if question.lower() == 'q':
        break
    answer = rag_chain.invoke(question)
    print(f"\n回答：{answer}")
