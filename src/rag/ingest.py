"""建库脚本 - 增量更新向量库（默认增量，--force 全量重建）

支持 .md / .txt / .pdf / .docx 多格式
流程：扫描文件 → 对比哈希 → 只处理变化的文件 → 更新 manifest
"""
import sys
import os
import re
import json
import hashlib
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

# manifest 文件：记录每个已处理文件的哈希，存在向量库目录下
MANIFEST_PATH = os.path.join(CHROMA_PATH, "manifest.json")
SUPPORTED_EXTS = {".md", ".txt", ".pdf", ".docx"}


def get_embeddings():
    """获取 Embedding 模型（自定义硅基流动封装）"""
    return SiliconFlowEmbeddings()


# ──────────────────────────────────────────────
# 文本清洗
# ──────────────────────────────────────────────

def clean_text(text: str) -> str:
    """清洗文档文本中的噪音，提升检索质量。

    处理内容：
    1. 去除零宽字符和控制字符（PDF 提取常见）
    2. 规范化空白：多个空格→单个，多个换行→最多两个
    3. 去除行首尾空白
    4. 合并重复标点（。。。→。，！！！→！）
    """
    if not text:
        return text

    text = re.sub(r'[\u200b-\u200f\u2028\u2029\ufeff]', '', text)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

    lines = [line.strip() for line in text.split('\n')]
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

    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'([。！？，；：、])\1+', r'\1', text)
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
    print(f"  文本清洗：{len(docs)} 个文档，{cleaned} 个有改动")
    return docs


# ──────────────────────────────────────────────
# 文档加载（全量 + 单文件）
# ──────────────────────────────────────────────

def load_documents():
    """全量加载 docs/ 目录下所有支持的文档（--force 时用）"""
    all_docs = []

    def _load(glob_pattern, loader_cls, loader_kwargs=None, file_type=None):
        try:
            loader = DirectoryLoader(
                DOCS_DIR, glob=glob_pattern,
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

    all_docs.extend(_load("**/*.md", TextLoader, {"encoding": "utf-8"}, "markdown"))
    all_docs.extend(_load("**/*.txt", TextLoader, {"encoding": "utf-8"}, "text"))
    all_docs.extend(_load("**/*.pdf", PyPDFLoader, file_type="pdf"))
    all_docs.extend(_load("**/*.docx", Docx2txtLoader, file_type="docx"))

    print(f"共加载 {len(all_docs)} 个文档片段")
    return all_docs


def load_single_file(filepath):
    """加载单个文档文件，根据扩展名选择加载器（增量更新时用）"""
    ext = os.path.splitext(filepath)[1].lower()
    try:
        if ext == ".md":
            docs = TextLoader(filepath, encoding="utf-8").load()
            file_type = "markdown"
        elif ext == ".txt":
            docs = TextLoader(filepath, encoding="utf-8").load()
            file_type = "text"
        elif ext == ".pdf":
            docs = PyPDFLoader(filepath).load()
            file_type = "pdf"
        elif ext == ".docx":
            docs = Docx2txtLoader(filepath).load()
            file_type = "docx"
        else:
            return []

        for doc in docs:
            doc.metadata["file_type"] = file_type
        return docs
    except Exception as e:
        print(f"  加载失败 {os.path.basename(filepath)}: {e}")
        return []


# ──────────────────────────────────────────────
# 分块
# ──────────────────────────────────────────────

def split_documents(docs):
    """分块：Markdown 按标题切，其他格式按字符长度切"""
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "header_1"), ("##", "header_2"), ("###", "header_3")],
        strip_headers=False,
    )
    char_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40)

    chunks = []
    for doc in docs:
        file_type = doc.metadata.get("file_type", "text")
        if file_type == "markdown":
            header_chunks = header_splitter.split_text(doc.page_content)
            for hc in header_chunks:
                hc.metadata["source"] = doc.metadata["source"]
                hc.metadata["file_type"] = doc.metadata.get("file_type", "markdown")
                header_parts = [hc.metadata[k] for k in ["header_1", "header_2", "header_3"] if k in hc.metadata]
                if header_parts:
                    hc.page_content = " > ".join(header_parts) + "\n" + hc.page_content
                chunks.extend(char_splitter.split_documents([hc]))
        else:
            chunks.extend(char_splitter.split_documents([doc]))

    missing_source = sum(1 for c in chunks if not c.metadata.get("source"))
    if missing_source:
        print(f"  ⚠️ 警告：{missing_source} 个块缺少 source metadata")

    return chunks


# ──────────────────────────────────────────────
# 增量更新基础设施
# ──────────────────────────────────────────────

def file_hash(filepath):
    """计算文件内容的 MD5 哈希，用于判断文件是否变化"""
    with open(filepath, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def scan_docs_dir():
    """扫描 docs/ 目录，返回所有支持的文件绝对路径列表"""
    files = []
    for root, _, filenames in os.walk(DOCS_DIR):
        for fn in filenames:
            if os.path.splitext(fn)[1].lower() in SUPPORTED_EXTS:
                files.append(os.path.abspath(os.path.join(root, fn)))
    return sorted(files)


def load_manifest():
    """加载已处理文件清单 {filepath: hash}，不存在则返回空"""
    if os.path.exists(MANIFEST_PATH):
        try:
            with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_manifest(manifest):
    """保存已处理文件清单"""
    os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def delete_by_source(vectorstore, filepath):
    """从向量库中删除指定来源文件的所有块，返回删除的块数"""
    try:
        result = vectorstore.get(where={"source": filepath})
        if result["ids"]:
            vectorstore.delete(ids=result["ids"])
            return len(result["ids"])
    except Exception as e:
        print(f"  删除旧块失败 {os.path.basename(filepath)}: {e}")
    return 0


def get_vectorstore(embeddings):
    """加载或创建向量库"""
    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )


# ──────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────

def main(force=False):
    print("=" * 50)
    mode = "全量重建" if force else "增量更新"
    print(f"  电商客服机器人 - 建库（{mode}）")
    print("=" * 50)

    # 1. 扫描 docs/ 目录
    files = scan_docs_dir()
    if not files:
        print("没有找到任何文档，请检查 docs/ 目录")
        return
    print(f"扫描到 {len(files)} 个文档")

    # 2. 全量重建模式：删除旧库，重置 manifest
    if force:
        if os.path.exists(CHROMA_PATH):
            import shutil
            shutil.rmtree(CHROMA_PATH)
            print("已删除旧向量库")
        manifest = {}
        to_add = files
        to_delete = []
    else:
        # 3. 增量模式：对比 manifest，找出新增/修改/删除的文件
        manifest = load_manifest()
        to_add = []
        to_delete = []

        for f in files:
            if manifest.get(f) != file_hash(f):
                to_add.append(f)

        for f in manifest:
            if f not in files:
                to_delete.append(f)

        # 没有变化就提前退出
        if not to_add and not to_delete:
            print("\n✅ 知识库已是最新，无需更新")
            return

        print(f"\n变更检测：{len(to_add)} 个新增/修改，{len(to_delete)} 个删除")

    # 4. 加载向量库
    embeddings = get_embeddings()
    vectorstore = get_vectorstore(embeddings)

    # 5. 处理删除的文件
    for f in to_delete:
        n = delete_by_source(vectorstore, f)
        print(f"  [删除] {os.path.basename(f)}（{n} 个块）")
        manifest.pop(f, None)

    # 6. 处理新增/修改的文件
    total_chunks = 0
    for f in to_add:
        name = os.path.basename(f)
        # 先删旧块（修改的文件）
        old_n = delete_by_source(vectorstore, f)
        if old_n:
            print(f"  [更新] {name}（删除旧块 {old_n} 个）")
        else:
            print(f"  [新增] {name}")

        # 加载 → 清洗 → 分块
        docs = load_single_file(f)
        if not docs:
            continue
        docs = clean_documents(docs)
        chunks = split_documents(docs)
        if not chunks:
            continue

        # 写入向量库
        vectorstore.add_documents(chunks)
        manifest[f] = file_hash(f)
        total_chunks += len(chunks)
        print(f"         → 写入 {len(chunks)} 个块")

    # 7. 保存 manifest
    save_manifest(manifest)

    # 8. 统计
    all_docs = vectorstore.get()
    total = len(all_docs["ids"]) if all_docs.get("ids") else 0
    print(f"\n✅ 完成！向量库共 {total} 个块，存在 {CHROMA_PATH}/")

    # 9. 测试检索
    print("\n=== 测试检索 ===")
    query = "退货政策是什么？"
    results = vectorstore.similarity_search(query, k=2)
    print(f"查询: {query}")
    for i, r in enumerate(results):
        print(f"  [{i+1}] {os.path.basename(r.metadata.get('source', '?'))}: {r.page_content[:60]}...")


if __name__ == "__main__":
    force = "--force" in sys.argv
    main(force=force)
