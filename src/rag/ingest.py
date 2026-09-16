"""建库脚本 - 支持 .md / .txt / .pdf / .docx 多格式文档加载、清洗、分块、向量化"""
import sys
import os
import re
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


def clean_text(text: str) -> str:
    """清洗文档文本中的噪音，提升检索质量。

    处理内容：
    1. 去除零宽字符和控制字符（PDF 提取常见）
    2. 规范化空白：多个空格→单个，多个换行→最多两个
    3. 去除行首尾空白
    4. 合并重复标点（。。。→。，！！！→！）
    5. 规范化全角/半角空格
    """
    if not text:
        return text

    # 1. 去除零宽字符和常见控制字符（保留 \n \t）
    text = re.sub(r'[\u200b-\u200f\u2028\u2029\ufeff]', '', text)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

    # 2. 去除行首尾空白
    lines = text.split('\n')
    lines = [line.strip() for line in lines]

    # 3. 去除空行（连续空行压缩为一个）
    cleaned_lines = []
    prev_empty = False
    for line in lines:
        if not line:
            if not prev_empty:
                cleaned_lines.append('')
            prev_empty = True
        else:
            cleaned_lines.append(line)
            prev_empty = False
    text = '\n'.join(cleaned_lines)

    # 4. 多个空格→单个空格（不影响中文）
    text = re.sub(r'[ \t]+', ' ', text)

    # 5. 合并重复标点（中文标点）
    text = re.sub(r'([。！？，；：、])\1+', r'\1', text)

    # 6. 合并重复换行（最多保留 2 个连续换行 = 一个空行）
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


def clean_documents(docs):
    """对加载后的文档批量清洗 page_content"""
    cleaned = 0
    for doc in docs:
        original = doc.page_content
        doc.page_content = clean_text(original)
        if original != doc.page_content:
            cleaned += 1
    print(f"文本清洗完成：{len(docs)} 个文档，其中 {cleaned} 个有改动")
    return docs


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

    # 2. 文本清洗（去噪音：控制字符、多余空白、重复标点等）
    docs = clean_documents(docs)

    # 3. 分块
    chunks = split_documents(docs)

    # 4. 向量化并存入 Chroma
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

    # 5. 测试检索
    print("\n=== 测试检索 ===")
    query = "退货政策是什么？"
    results = vectorstore.similarity_search(query, k=2)
    print(f"查询: {query}")
    for i, r in enumerate(results):
        print(f"  [{i+1}] {r.metadata.get('source', '?')}: {r.page_content[:60]}...")


if __name__ == "__main__":
    main()
