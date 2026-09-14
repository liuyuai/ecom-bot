"""企业级 Embedding 封装 - 重试 + 缓存 + 批量 + 异步"""
from typing import List
import json
import os
import hashlib
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from langchain_core.embeddings import Embeddings

from config import EMBED_API_KEY, EMBED_BASE_URL, EMBED_MODEL, PROJECT_ROOT, EMBED_TIMEOUT
from common.logger import get_logger

logger = get_logger("embedding")


# 缓存文件路径
CACHE_FILE = os.path.join(PROJECT_ROOT, "embed_cache.json")


class SiliconFlowEmbeddings(Embeddings):
    """
    硅基流动 Embedding，企业级封装：
    - tenacity 指数退避重试（3次）
    - 本地文件缓存（相同文本不重复调 API）
    - 批量调用（embed_documents 一次传多条）
    - 异步支持
    """

    def __init__(self):
        self._cache = self._load_cache()

    # ========== 缓存 ==========
    def _load_cache(self) -> dict:
        """从文件加载缓存"""
        if os.path.exists(CACHE_FILE):
            try:
                with open(CACHE_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_cache(self):
        """保存缓存到文件"""
        try:
            with open(CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"缓存保存失败: {e}")

    def _cache_key(self, text: str) -> str:
        """用文本的 MD5 作为缓存键"""
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    # ========== 核心调用（带重试）==========
    @retry(
        stop=stop_after_attempt(3),               # 最多重试3次
        wait=wait_exponential(multiplier=1, min=1, max=10),  # 1s → 2s → 4s 指数退避
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.TimeoutException, httpx.ConnectError)),
        reraise=True,
    )
    def _embed_api(self, texts: List[str]) -> List[List[float]]:
        """调用硅基流动 Embedding API（带重试）"""
        resp = httpx.post(
            f"{EMBED_BASE_URL}/embeddings",
            headers={"Authorization": f"Bearer {EMBED_API_KEY}"},
            json={"model": EMBED_MODEL, "input": texts},
            timeout=EMBED_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        return [item["embedding"] for item in data["data"]]

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.TimeoutException, httpx.ConnectError)),
        reraise=True,
    )
    async def _aembed_api(self, texts: List[str]) -> List[List[float]]:
        """异步调用 Embedding API（带重试）"""
        async with httpx.AsyncClient(timeout=EMBED_TIMEOUT) as client:
            resp = await client.post(
                f"{EMBED_BASE_URL}/embeddings",
                headers={"Authorization": f"Bearer {EMBED_API_KEY}"},
                json={"model": EMBED_MODEL, "input": texts},
            )
            resp.raise_for_status()
            data = resp.json()
            return [item["embedding"] for item in data["data"]]

    # ========== 同步接口 ==========
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量嵌入：先查缓存，未命中的批量调 API"""
        results = [None] * len(texts)
        to_embed = []  # (index, text)

        for i, text in enumerate(texts):
            key = self._cache_key(text)
            if key in self._cache:
                results[i] = self._cache[key]
            else:
                to_embed.append((i, text))

        if to_embed:
            # 批量调用未命中的文本
            batch_texts = [t for _, t in to_embed]
            embeddings = self._embed_api(batch_texts)
            for (idx, text), emb in zip(to_embed, embeddings):
                results[idx] = emb
                self._cache[self._cache_key(text)] = emb
            self._save_cache()

        return results

    def embed_query(self, text: str) -> List[float]:
        """单条查询嵌入"""
        return self.embed_documents([text])[0]

    # ========== 异步接口 ==========
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """异步批量嵌入"""
        results = [None] * len(texts)
        to_embed = []

        for i, text in enumerate(texts):
            key = self._cache_key(text)
            if key in self._cache:
                results[i] = self._cache[key]
            else:
                to_embed.append((i, text))

        if to_embed:
            batch_texts = [t for _, t in to_embed]
            embeddings = await self._aembed_api(batch_texts)
            for (idx, text), emb in zip(to_embed, embeddings):
                results[idx] = emb
                self._cache[self._cache_key(text)] = emb
            self._save_cache()

        return results

    async def aembed_query(self, text: str) -> List[float]:
        """异步单条查询嵌入"""
        return (await self.aembed_documents([text]))[0]
