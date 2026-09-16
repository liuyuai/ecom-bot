"""检索模块 - 混合检索（向量+BM25）+ 重排序（Reranker）"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from langchain_community.vectorstores import Chroma
from rank_bm25 import BM25Okapi
import jieba

from config import (
    CHROMA_PATH, COLLECTION_NAME, TOP_K,
    EMBED_API_KEY, EMBED_BASE_URL, RERANK_MODEL, RERANK_TIMEOUT,
)
from rag.embeddings import SiliconFlowEmbeddings
from common.logger import get_logger

logger = get_logger("retriever")

# 全局缓存：BM25 索引和文档列表，避免每次检索都重建
_bm25 = None
_all_docs = None
_bm25_doc_count = 0  # 构建 BM25 时 Chroma 的文档数，用于检测知识库变化


def _tokenize(text: str) -> list:
    """中文分词：用 jieba 分词，BM25 需要 token 列表"""
    return list(jieba.cut(text))


def _get_bm25_index():
    """懒加载 BM25 索引（从 Chroma 取所有文档构建）

    缓存失效机制：记录构建时的 Chroma 文档计数，
    检索前比对当前计数，变化则自动重建（好评入库/增量更新后自动生效）。
    """
    global _bm25, _all_docs, _bm25_doc_count

    vs = get_vectorstore()
    current_count = vs._collection.count() if hasattr(vs, "_collection") else len(vs.get()["documents"])

    # 缓存有效：BM25 已构建 且 文档计数未变化
    if _bm25 is not None and current_count == _bm25_doc_count:
        return _bm25, _all_docs

    # 缓存失效：重建 BM25 索引
    if _bm25 is not None:
        logger.info(f"BM25 缓存失效 | 旧计数={_bm25_doc_count} → 新计数={current_count}，重建索引")

    result = vs.get()
    _all_docs = []
    for i, doc_text in enumerate(result["documents"]):
        metadata = result["metadatas"][i] if result["metadatas"] else {}
        from langchain_core.documents import Document
        _all_docs.append(Document(page_content=doc_text, metadata=metadata))

    tokenized_corpus = [_tokenize(doc.page_content) for doc in _all_docs]
    _bm25 = BM25Okapi(tokenized_corpus)
    _bm25_doc_count = current_count
    logger.info(f"BM25 索引已构建 | 文档数={current_count}")
    return _bm25, _all_docs


def get_vectorstore():
    """加载已有的 Chroma 向量库"""
    embeddings = SiliconFlowEmbeddings()
    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )


def _vector_search(query: str, k: int) -> list:
    """向量检索：语义相似"""
    vs = get_vectorstore()
    return vs.similarity_search(query, k=k)


def _bm25_search(query: str, k: int) -> list:
    """BM25 检索：关键词匹配"""
    bm25, all_docs = _get_bm25_index()
    tokenized_query = _tokenize(query)
    scores = bm25.get_scores(tokenized_query)
    # 按分数降序，取 top k
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [all_docs[i] for i in top_indices if scores[i] > 0]


def _merge_dedup(docs_list: list) -> list:
    """合并多个检索结果，按内容去重"""
    seen = set()
    merged = []
    for docs in docs_list:
        for doc in docs:
            key = doc.page_content[:100]  # 用前100字作为去重键
            if key not in seen:
                seen.add(key)
                merged.append(doc)
    return merged


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.TimeoutException, httpx.ConnectError)),
    reraise=True,
)
def _rerank_api(query: str, documents: list) -> list:
    """调用硅基流动 Reranker API（带重试）"""
    resp = httpx.post(
        f"{EMBED_BASE_URL}/rerank",
        headers={"Authorization": f"Bearer {EMBED_API_KEY}"},
        json={
            "model": RERANK_MODEL,
            "query": query,
            "documents": documents,
        },
        timeout=RERANK_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json().get("results", [])


def _rerank(query: str, docs: list, top_k: int) -> list:
    """
    重排序：调用硅基流动 Reranker API，按 query-doc 相关性重新打分。
    API 失败时自动重试3次，仍失败则回退到原顺序。
    """
    if not docs:
        return []

    documents = [doc.page_content for doc in docs]
    try:
        results = _rerank_api(query, documents)
        # 按分数降序，取 top_k
        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        return [docs[r["index"]] for r in results[:top_k]]
    except Exception as e:
        logger.error(f"Reranker 失败（重试3次后）| {e}")
        return docs[:top_k]


def search_knowledge(query: str, k: int = TOP_K):
    """
    混合检索：向量 + BM25 → 合并去重 → Reranker 重排序
    """
    # 1. 两路检索（各取 k*2，留出去重和重排序的空间）
    vector_docs = _vector_search(query, k=k * 2)
    bm25_docs = _bm25_search(query, k=k * 2)

    # 2. 合并去重
    merged = _merge_dedup([vector_docs, bm25_docs])

    # 3. 重排序，取 top k
    reranked = _rerank(query, merged, top_k=k)

    return reranked


def format_docs(docs):
    """把检索结果格式化成文本，拼到 prompt 里"""
    return "\n\n".join(
        f"[来自 {os.path.basename(doc.metadata.get('source', '未知'))}]\n{doc.page_content}"
        for doc in docs
    )
