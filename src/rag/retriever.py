"""检索模块 - 混合检索（向量+BM25）+ 重排序（Reranker）"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import hashlib
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
_bm25_fingerprint = None  # 构建 BM25 时的内容指纹（全部文档文本 md5），用于检测知识库变化


def _tokenize(text: str) -> list:
    """中文分词：用 jieba 分词，BM25 需要 token 列表"""
    return list(jieba.cut(text))


def _content_fingerprint(vs) -> str:
    """计算 Chroma 全部文档的内容指纹：md5(排序拼接全部文本)。

    用内容哈希而非文档计数判断 BM25 缓存失效：
    - 计数方案漏洞：知识库"改一删一、总数不变"时不触发重建，检索仍用旧 BM25 索引
    - 内容哈希：任何文本变化（新增/删除/修改，含同数量变更）都会改变指纹，必然触发重建
    - 先排序再拼接：防止 Chroma 内部返回顺序变化导致误判重建
    """
    docs = (vs.get(include=["documents"]) or {}).get("documents") or []
    texts = sorted((d or "") for d in docs)
    return hashlib.md5("\x1f".join(texts).encode("utf-8")).hexdigest()


def _get_bm25_index():
    """懒加载 BM25 索引（从 Chroma 取所有文档构建）

    缓存失效机制：记录构建时的内容指纹，
    检索前比对当前指纹，任何内容变化（增/删/改，含同数量变更）自动重建
    （好评入库/增量更新后自动生效）。
    """
    global _bm25, _all_docs, _bm25_fingerprint

    vs = get_vectorstore()
    current_fp = _content_fingerprint(vs)

    # 缓存有效：BM25 已构建 且 内容指纹未变化
    if _bm25 is not None and current_fp == _bm25_fingerprint:
        return _bm25, _all_docs

    # 缓存失效：重建 BM25 索引
    if _bm25 is not None:
        logger.info("BM25 缓存失效 | 内容指纹变化（增/删/改），重建索引")

    result = vs.get()
    _all_docs = []
    for i, doc_text in enumerate(result["documents"]):
        metadata = result["metadatas"][i] if result["metadatas"] else {}
        from langchain_core.documents import Document
        _all_docs.append(Document(page_content=doc_text, metadata=metadata))

    tokenized_corpus = [_tokenize(doc.page_content) for doc in _all_docs]
    _bm25 = BM25Okapi(tokenized_corpus)
    _bm25_fingerprint = current_fp
    logger.info(f"BM25 索引已构建 | 文档数={len(_all_docs)}")
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
