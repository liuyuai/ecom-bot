"""Trace 留痕模块：请求级 trace + 工具调用明细 + 危险操作审计

三张表：
- traces：每次用户请求一条记录（问题、回答、工具、耗时、状态）
- tool_calls：每次工具调用明细（工具名、参数、结果、耗时）
- audit_log：危险操作审计（删单/退款，谁、什么时候、操作了什么）

用 SQLite 异步存储，和 checkpoints.db 分开，避免互相影响。
"""
import json
import time
import os
import aiosqlite
from datetime import datetime

from config import PROJECT_ROOT
from common.logger import get_logger

logger = get_logger("trace")

# Trace 数据库路径（和 checkpoints.db 分开）
TRACE_DB_PATH = os.path.join(PROJECT_ROOT, "traces.db")

# 建表 SQL
_INIT_SQL = """
CREATE TABLE IF NOT EXISTS traces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    question TEXT,
    answer TEXT,
    tools_called TEXT,          -- JSON 数组：["search_knowledge_base", "query_order"]
    latency_ms REAL,
    token_usage INTEGER,        -- 估算的 token 消耗
    status TEXT,                -- success / failed / blocked / rate_limited
    intent TEXT,                -- 意图分类结果
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tool_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trace_id INTEGER,
    request_id TEXT,
    session_id TEXT,
    tool_name TEXT NOT NULL,
    params TEXT,                -- JSON：调用参数
    result TEXT,                -- 结果摘要（截断到 500 字）
    latency_ms REAL,
    success INTEGER,            -- 1=成功 0=失败
    error TEXT,                 -- 错误信息（失败时）
    created_at TEXT NOT NULL,
    FOREIGN KEY (trace_id) REFERENCES traces(id)
);

CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    action TEXT NOT NULL,       -- delete_order / refund_order
    order_id TEXT NOT NULL,
    result TEXT,                -- 执行结果
    success INTEGER,            -- 1=成功 0=失败
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_traces_session ON traces(session_id);
CREATE INDEX IF NOT EXISTS idx_traces_created ON traces(created_at);
CREATE INDEX IF NOT EXISTS idx_tool_calls_trace ON tool_calls(trace_id);
CREATE INDEX IF NOT EXISTS idx_tool_calls_name ON tool_calls(tool_name);
CREATE INDEX IF NOT EXISTS idx_audit_session ON audit_log(session_id);
CREATE INDEX IF NOT EXISTS idx_audit_order ON audit_log(order_id);
"""

# 全局 trace_id 上下文（一次请求内共享）
_current_trace_id = None


async def init_db():
    """初始化数据库（建表）"""
    async with aiosqlite.connect(TRACE_DB_PATH) as db:
        await db.executescript(_INIT_SQL)
        await db.commit()
    logger.info(f"Trace 数据库已初始化 | {TRACE_DB_PATH}")


async def start_trace(request_id: str, session_id: str, question: str) -> int:
    """开始一次请求的 trace，返回 trace_id"""
    global _current_trace_id
    now = datetime.now().isoformat()
    async with aiosqlite.connect(TRACE_DB_PATH) as db:
        cursor = await db.execute(
            "INSERT INTO traces (request_id, session_id, question, created_at) VALUES (?, ?, ?, ?)",
            (request_id, session_id, question, now),
        )
        await db.commit()
        _current_trace_id = cursor.lastrowid
    return _current_trace_id


async def finish_trace(
    trace_id: int,
    answer: str = "",
    tools_called: list = None,
    latency_ms: float = 0,
    token_usage: int = 0,
    status: str = "success",
    intent: str = "",
):
    """结束一次请求的 trace，补充结果"""
    tools_json = json.dumps(tools_called or [], ensure_ascii=False)
    answer_truncated = answer[:2000] if answer else ""
    async with aiosqlite.connect(TRACE_DB_PATH) as db:
        await db.execute(
            """UPDATE traces 
               SET answer=?, tools_called=?, latency_ms=?, token_usage=?, status=?, intent=?
               WHERE id=?""",
            (answer_truncated, tools_json, latency_ms, token_usage, status, intent, trace_id),
        )
        await db.commit()
    logger.debug(f"Trace 完成 | id={trace_id} | status={status} | tools={tools_called}")


async def record_tool_call(
    trace_id: int,
    request_id: str,
    session_id: str,
    tool_name: str,
    params: dict,
    result: str,
    latency_ms: float,
    success: bool,
    error: str = "",
):
    """记录一次工具调用"""
    now = datetime.now().isoformat()
    params_json = json.dumps(params or {}, ensure_ascii=False)[:500]
    result_truncated = (result or "")[:500]
    async with aiosqlite.connect(TRACE_DB_PATH) as db:
        await db.execute(
            """INSERT INTO tool_calls 
               (trace_id, request_id, session_id, tool_name, params, result, latency_ms, success, error, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (trace_id, request_id, session_id, tool_name, params_json, result_truncated,
             latency_ms, 1 if success else 0, error[:200], now),
        )
        await db.commit()


async def record_audit(
    session_id: str,
    action: str,
    order_id: str,
    result: str,
    success: bool,
):
    """记录危险操作审计（删单/退款）"""
    now = datetime.now().isoformat()
    async with aiosqlite.connect(TRACE_DB_PATH) as db:
        await db.execute(
            """INSERT INTO audit_log (session_id, action, order_id, result, success, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (session_id, action, order_id, result[:500], 1 if success else 0, now),
        )
        await db.commit()
    logger.info(f"审计记录 | action={action} | order={order_id} | success={success}")


async def get_trace_stats(days: int = 7) -> dict:
    """获取 trace 统计（最近 N 天）"""
    since = datetime.now().timestamp() - days * 86400
    since_str = datetime.fromtimestamp(since).isoformat()
    async with aiosqlite.connect(TRACE_DB_PATH) as db:
        db.row_factory = aiosqlite.Row

        # 请求统计
        cursor = await db.execute(
            """SELECT 
               COUNT(*) as total,
               SUM(CASE WHEN status='success' THEN 1 ELSE 0 END) as success,
               SUM(CASE WHEN status='failed' THEN 1 ELSE 0 END) as failed,
               SUM(CASE WHEN status='blocked' THEN 1 ELSE 0 END) as blocked,
               AVG(latency_ms) as avg_latency,
               SUM(token_usage) as total_tokens
               FROM traces WHERE created_at >= ?""",
            (since_str,),
        )
        row = await cursor.fetchone()

        # 工具调用统计
        cursor2 = await db.execute(
            """SELECT tool_name, COUNT(*) as cnt, AVG(latency_ms) as avg_latency,
               SUM(CASE WHEN success=0 THEN 1 ELSE 0 END) as failures
               FROM tool_calls WHERE created_at >= ?
               GROUP BY tool_name ORDER BY cnt DESC""",
            (since_str,),
        )
        tool_rows = await cursor2.fetchall()

        # 审计统计
        cursor3 = await db.execute(
            """SELECT action, COUNT(*) as cnt,
               SUM(CASE WHEN success=1 THEN 1 ELSE 0 END) as success
               FROM audit_log WHERE created_at >= ?
               GROUP BY action""",
            (since_str,),
        )
        audit_rows = await cursor3.fetchall()

    return {
        "period_days": days,
        "total_requests": row["total"] or 0,
        "successful": row["success"] or 0,
        "failed": row["failed"] or 0,
        "blocked": row["blocked"] or 0,
        "avg_latency_ms": round(row["avg_latency"] or 0, 1),
        "total_tokens": row["total_tokens"] or 0,
        "tool_calls": [
            {"tool": r["tool_name"], "count": r["cnt"],
             "avg_latency_ms": round(r["avg_latency"] or 0, 1),
             "failures": r["failures"] or 0}
            for r in tool_rows
        ],
        "audit_actions": [
            {"action": r["action"], "count": r["cnt"], "success": r["success"] or 0}
            for r in audit_rows
        ],
    }
