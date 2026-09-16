"""生产级健康检查：检测所有外部依赖的连通性

检查项：
- LLM API（DeepSeek）：发最小请求验证 Key 和网络
- Embedding API（硅基流动）：发最小向量化请求
- 向量库（Chroma）：执行最小检索
- 会话存储（SQLite/Redis）：读写测试
"""
import time
import asyncio
from common.logger import get_logger

logger = get_logger("health")

# 健康检查超时（秒），避免某个依赖卡住整个检查
CHECK_TIMEOUT = 5


async def check_llm() -> dict:
    """检查 LLM API 连通性"""
    start = time.time()
    try:
        from services.llm import get_llm
        from langchain_core.messages import HumanMessage
        llm = get_llm(temperature=0)
        # 发一个最小请求，timeout 由 llm 客户端控制
        response = await asyncio.wait_for(
            llm.ainvoke([HumanMessage(content="hi")]),
            timeout=CHECK_TIMEOUT,
        )
        elapsed = round((time.time() - start) * 1000)
        return {"name": "llm", "status": "ok", "latency_ms": elapsed}
    except asyncio.TimeoutError:
        return {"name": "llm", "status": "timeout", "latency_ms": CHECK_TIMEOUT * 1000}
    except Exception as e:
        elapsed = round((time.time() - start) * 1000)
        return {"name": "llm", "status": "error", "latency_ms": elapsed, "error": str(e)[:100]}


async def check_embedding() -> dict:
    """检查 Embedding API 连通性"""
    start = time.time()
    try:
        from rag.embeddings import SiliconFlowEmbeddings
        embeddings = SiliconFlowEmbeddings()
        result = await asyncio.wait_for(
            asyncio.to_thread(embeddings.embed_query, "test"),
            timeout=CHECK_TIMEOUT,
        )
        elapsed = round((time.time() - start) * 1000)
        dim = len(result) if result else 0
        return {"name": "embedding", "status": "ok", "latency_ms": elapsed, "dimension": dim}
    except asyncio.TimeoutError:
        return {"name": "embedding", "status": "timeout", "latency_ms": CHECK_TIMEOUT * 1000}
    except Exception as e:
        elapsed = round((time.time() - start) * 1000)
        return {"name": "embedding", "status": "error", "latency_ms": elapsed, "error": str(e)[:100]}


async def check_vectorstore() -> dict:
    """检查向量库连通性"""
    start = time.time()
    try:
        from rag.retriever import get_vectorstore
        vs = get_vectorstore()
        result = await asyncio.wait_for(
            asyncio.to_thread(vs.similarity_search, "health check", 1),
            timeout=CHECK_TIMEOUT,
        )
        elapsed = round((time.time() - start) * 1000)
        count = len(result)
        return {"name": "vectorstore", "status": "ok", "latency_ms": elapsed, "results": count}
    except asyncio.TimeoutError:
        return {"name": "vectorstore", "status": "timeout", "latency_ms": CHECK_TIMEOUT * 1000}
    except Exception as e:
        elapsed = round((time.time() - start) * 1000)
        return {"name": "vectorstore", "status": "error", "latency_ms": elapsed, "error": str(e)[:100]}


async def check_checkpointer(checkpointer) -> dict:
    """检查会话存储连通性（读写测试）"""
    start = time.time()
    try:
        if checkpointer is None:
            return {"name": "checkpointer", "status": "error", "error": "not initialized"}

        # 写测试：存一个测试线程的状态
        test_config = {"configurable": {"thread_id": "__health_check__"}}
        test_state = {"messages": []}
        await asyncio.wait_for(
            checkpointer.aput(test_config, test_state, {}),
            timeout=CHECK_TIMEOUT,
        )

        # 读测试：读回来
        saved = await asyncio.wait_for(
            checkpointer.aget(test_config),
            timeout=CHECK_TIMEOUT,
        )

        elapsed = round((time.time() - start) * 1000)
        ok = saved is not None
        return {"name": "checkpointer", "status": "ok" if ok else "error", "latency_ms": elapsed}
    except asyncio.TimeoutError:
        return {"name": "checkpointer", "status": "timeout", "latency_ms": CHECK_TIMEOUT * 1000}
    except Exception as e:
        elapsed = round((time.time() - start) * 1000)
        return {"name": "checkpointer", "status": "error", "latency_ms": elapsed, "error": str(e)[:100]}


async def run_health_checks(checkpointer=None) -> dict:
    """运行所有健康检查，返回汇总结果"""
    start = time.time()

    checks = await asyncio.gather(
        check_llm(),
        check_embedding(),
        check_vectorstore(),
        check_checkpointer(checkpointer),
    )

    total_ms = round((time.time() - start) * 1000)
    all_ok = all(c["status"] == "ok" for c in checks)
    degraded = any(c["status"] == "timeout" for c in checks)

    return {
        "status": "ok" if all_ok else ("degraded" if degraded else "error"),
        "total_ms": total_ms,
        "checks": {c["name"]: c for c in checks},
        "timestamp": time.time(),
    }
