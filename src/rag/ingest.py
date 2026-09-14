"""建库脚本 - 支持 .md / .txt / .pdf / .docx 多格式文档加载、分块、向量化"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_community.document_loaders import (
    DirectoryLoader, TextLoader, PyPDFLoader, Docx2txtLoader,
)
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
    """加载 docs/ 目录下所有支持的文档（.md / .txt / .pdf / .docx）"""
    all_docs = []

    def _load(glob_pattern, loader_cls, loader_kwargs=None, file_type=None):
        try:
            loader = DirectoryLoader(
                DOCS_DIR,
                glob=glob_pattern,
                loader_cls=loader_cls,
                loader_kwargs=loader_kwargs or {},
            )
            docs = loader.load()
            for doc in docs:
                doc.metadata["file_type"] = file_type or os.path.splitext(doc.metadata["source"])[1].lstrip(".")
            return docs
        except Exception as e:
            print(f"  加载 {glob_pattern} 失败: {e}")
            return []

    # .md
    md_docs = _load("**/*.md", TextLoader, {"encoding": "utf-8"}, "markdown")
    print(f"[Markdown] {len(md_docs)} 个")
    all_docs.extend(md_docs)

    # .txt
    txt_docs = _load("**/*.txt", TextLoader, {"encoding": "utf-8"}, "text")
    print(f"[文本] {len(txt_docs)} 个")
    all_docs.extend(txt_docs)

    # .pdf
    pdf_docs = _load("**/*.pdf", PyPDFLoader, file_type="pdf")
    print(f"[PDF] {len(pdf_docs)} 页")
    all_docs.extend(pdf_docs)

    # .docx
    docx_docs = _load("**/*.docx", Docx2txtLoader, file_type="docx")
    print(f"[Word] {len(docx_docs)} 个")
    all_docs.extend(docx_docs)

    print(f"\n共加载 {len(all_docs)} 个文档片段")
    for doc in all_docs:
        print(f"  - [{doc.metadata.get('file_type', '?')}] {os.path.basename(doc.metadata['source'])}: {len(doc.page_content)} 字")
    return all_docs


def split_documents(docs):
    """分块：Markdown 按标题切，其他格式按字符长度切"""
    # Markdown 标题切分器
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "header_1"),
            ("##", "header_2"),
            ("###", "header_3"),
        ],
        strip_headers=False,
    )

    # 通用字符长度切分器
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )

    chunks = []
    for doc in docs:
        file_type = doc.metadata.get("file_type", "text")

        if file_type == "markdown":
            # Markdown：先按标题切，再按长度兜底，标题路径拼进内容
            header_chunks = header_splitter.split_text(doc.page_content)
            for hc in header_chunks:
                hc.metadata["source"] = doc.metadata["source"]
                header_parts = []
                for key in ["header_1", "header_2", "header_3"]:
                    if key in hc.metadata:
                        header_parts.append(hc.metadata[key])
                if header_parts:
                    hc.page_content = " > ".join(header_parts) + "\n" + hc.page_content
                smaller = char_splitter.split_documents([hc])
                chunks.extend(smaller)
        else:
            # .txt / .pdf / .docx：直接按字符长度切
            smaller = char_splitter.split_documents([doc])
            chunks.extend(smaller)

    print(f"分块完成：{len(chunks)} 个块")
    return chunks


def main():
    print("=" * 50)
    print("  电商客服机器人 - 建库（支持 md/txt/pdf/docx）")
    print("=" * 50)

    # 1. 加载文档
    docs = load_documents()
    if not docs:
        print("没有找到任何文档，请检查 docs/ 目录")
        return

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
