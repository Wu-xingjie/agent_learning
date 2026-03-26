import os, sys
lib_path = os.path.realpath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(lib_path)
import common
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
# from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains.retrieval_qa.base import RetrievalQA

# 资料库路径
material_path = os.path.abspath("libmaterial")
# 数据库路径
libdata_path = os.path.abspath("libdata")

# 读取并存储PDF文件
all_docments = []
print("================== 开始加载资料 ==================")
for file in os.listdir(material_path):
    try:
        file_abspath = os.path.join(material_path, file)
        print("[INFO] 当前加载文档{}".format(file_abspath))
        loader = PyPDFLoader(file_abspath)
        document = loader.load()
        for doc in document:
            doc.metadata["source"] = file
        all_docments.extend(document)
        print("[INFO] {}文档加载完成".format(file))
    except:
        print("[ERRO] {}文档加载失败".format(file))
print("[INFO] 文档加载完毕, 共加载文档数：{}".format(len(all_docments)))
print("================== 文档加载结束 ==================\n")

# 文档分块
print("================== 开始文档分块 ==================")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=5000,          # 每块最大字符数
    chunk_overlap=50,        # 块之间重叠字符数，避免语义断裂
    separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]  # 优先按段落、句子切分
)
chunks = text_splitter.split_documents(all_docments)
for i in range(len(chunks)):
    print("chunk_{}:\n{}\n".format(i, chunks[i].page_content))
print("[INFO] 共生成{}个文本块".format(len(chunks)))
print("================== 文档分块结束 ==================\n")

# 向量化存储
print("================== 开始向量化存储 ==================\n")
embedding = OllamaEmbeddings(model="nomic-embed-text:latest")
vector_store = Chroma.from_documents(
    documents = chunks,
    embedding = embedding,
    persist_directory=libdata_path
)
vector_store.persist()
print("[INFO] 向量数据库已存储与{}".format(libdata_path))
print("================== 向量化存储结束 ==================\n")

# 构建检索链
print("================== 开始构建检索链 ==================")
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)
prompt_template = """你是一个专业的知识助手。请根据以下参考资料回答用户问题。
如果参考资料中没有相关信息，请诚实告知，不要编造。

参考资料：
{context}

用户问题：{question}
回答："""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)
print("================== 构建检索链结束 ==================\n")


qa_chain = RetrievalQA.from_chain_type(
    llm=common.llm_local,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": PROMPT},
    return_source_documents=True
)

# 问答交互
print("================== 问答交互 ==================")
print("\n知识库已就绪！输入 'exit' 退出。")
while True:
    query = input("\n请输入问题：")
    if query.lower() == "exit":
        break
    
    # 使用 .invoke() 方法替代直接调用
    result = qa_chain.invoke({"query": query})
    
    print("\n回答：", result["result"])
    print("\n参考来源：")
    for doc in result["source_documents"]:
        print(f"- {doc.metadata.get('source', '未知')} (第{doc.metadata.get('page', '?')}页)")