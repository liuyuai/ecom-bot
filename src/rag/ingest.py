"""建库脚本 - 使用 LangChain 的文档加载、分块、向量化"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from config import (
    CHROMA_PATH, COLLECTION_NAME, DOCS_DIR,
)
from rag.embeddings import SiliconFlowEmbeddings


def get_embeddings():
    """获取 Embedding 模型（自定义硅基流动封装）"""
    return SiliconFlowEmbeddings()


def load_documents():
    """加载 docs/ 目录下的所有 .md 文件"""
    loader = DirectoryLoader(
        DOCS_DIR,
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    docs = loader.load()
    print(f"加载了 {len(docs)} 个文档")
    for doc in docs:
        print(f"  - {doc.metadata['source']}: {len(doc.page_content)} 字")
    return docs


def split_documents(docs):
    """分块：先按 Markdown 标题切，再按字符长度兜底，标题路径拼进内容"""
    # 第1层：按标题层级切分（strip_headers=False 保留标题在内容中）
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "header_1"),
            ("##", "header_2"),
            ("###", "header_3"),
        ],
        strip_headers=False,
    )

    # 第2层：字符长度兜底（每块不超过500字，重叠50字）
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )

    chunks = []
    for doc in docs:
        # 先按标题切
        header_chunks = header_splitter.split_text(doc.page_content)
        # 再按字符长度切
        for hc in header_chunks:
            hc.metadata["source"] = doc.metadata["source"]
            # 把标题层级拼到内容前面，确保检索时能命中标题关键词
            header_parts = []
            for key in ["header_1", "header_2", "header_3"]:
                if key in hc.metadata:
                    header_parts.append(hc.metadata[key])
            if header_parts:
                hc.page_content = " > ".join(header_parts) + "\n" + hc.page_content
            smaller = char_splitter.split_documents([hc])
            chunks.extend(smaller)

    print(f"分块完成：{len(chunks)} 个块")
    return chunks


def main():
    print("=" * 50)
    print("  电商客服机器人 - 建库")
    print("=" * 50)

    # 1. 加载文档
    docs = load_documents()

    # 2. 分块
    chunks = split_documents(docs)

    # 3. 向量化并存入 Chroma
    print("\n正在向量化并存入 Chroma...")
    embeddings = get_embeddings()

    # 如果库已存在，删除重建
    if os.path.exists(CHROMA_PATH):
        import shutil
        shutil.rmtree(CHROMA_PATH)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH,
        collection_name=COLLECTION_NAME,
    )
    print(f"建库完成！共 {len(chunks)} 个块，存在 {CHROMA_PATH}/")

    # 4. 测试检索
    print("\n=== 测试检索 ===")
    query = "退货政策是什么？"
    results = vectorstore.similarity_search(query, k=2)
    print(f"查询: {query}")
    for i, r in enumerate(results):
        print(f"  [{i+1}] {r.metadata.get('source', '?')}: {r.page_content[:60]}...")


if __name__ == "__main__":
    main()
