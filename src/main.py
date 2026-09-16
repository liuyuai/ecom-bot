"""FastAPI 后端：电商客服机器人 API（企业级版）
入口层：组装 API 层、Agent 层、服务层、数据层
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import json
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# 限流
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# 各层导入
from agent.workflow import build_workflow, stream_agent
from agent.tools import execute_delete_order, execute_refund_order
from services.query_rewrite import rewrite_query
from services.intent import classify_intent, get_direct_reply, build_intent_context
from services.feedback import save_good_feedback, save_bad_feedback, get_good_feedback_count, get_bad_feedback_count
from services.security import security_check, filter_output
from services.health import run_health_checks
from common.metrics import metrics
from common.logger import get_logger, new_request_id, set_request_id
from config import (
    SQLITE_DB_PATH, HOST, PORT, LOG_LEVEL,
    REDIS_URL, MAX_MESSAGE_LENGTH, RATE_LIMIT_PER_MINUTE,
    CORS_ORIGINS,
)

logger = get_logger("api")

# 限流器：按 IP 限制
limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动初始化，关闭时优雅停机"""
    logger.info("应用启动中...")

    # 会话存储：优先 Redis（生产），回退 SQLite（开发）
    checkpointer = None
    if REDIS_URL:
        try:
            from langgraph.checkpoint.redis.aio import AsyncRedisSaver
            checkpointer = AsyncRedisSaver.from_conn_string(REDIS_URL)
            logger.info("使用 Redis 会话存储")
        except Exception as e:
            logger.warning(f"Redis 连接失败，回退 SQLite: {e}")

    if checkpointer is None:
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        checkpointer = AsyncSqliteSaver.from_conn_string(SQLITE_DB_PATH)
        logger.info("使用 SQLite 会话存储")

    async with checkpointer as cp:
        workflow = build_workflow()
        app.state.agent = workflow.compile(checkpointer=cp)
        app.state.checkpointer = cp
        logger.info("应用启动完成，Agent 已编译")
        yield
        logger.info("正在关闭，等待请求完成...")

    logger.info("应用已关闭")


app = FastAPI(title="电商客服机器人", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS：开发默认 "*"，生产环境在 .env 配置白名单域名
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ========== 请求模型（带校验）==========

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=MAX_MESSAGE_LENGTH)
    session_id: str = Field(..., min_length=1, max_length=100)

    @field_validator("message")
    @classmethod
    def strip_and_check(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("消息不能为空")
        return v

    @field_validator("session_id")
    @classmethod
    def validate_session(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("session_id 不能为空")
        return v


class FeedbackRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=MAX_MESSAGE_LENGTH)
    answer: str = Field(..., min_length=1, max_length=5000)
    rating: str = Field(..., description="up 或 down")
    session_id: str = Field(..., min_length=1, max_length=100)

    @field_validator("rating")
    @classmethod
    def validate_rating(cls, v: str) -> str:
        if v not in ("up", "down"):
            raise ValueError("rating 只能是 up 或 down")
        return v


class HitlConfirmRequest(BaseModel):
    """HITL 人工确认请求：用户确认后直接执行危险操作，不经过大模型"""
    action: str = Field(..., description="操作类型：delete 或 refund")
    order_id: str = Field(..., min_length=1, max_length=50)

    @field_validator("action")
    @classmethod
    def validate_action(cls, v: str) -> str:
        if v not in ("delete", "refund"):
            raise ValueError("action 只能是 delete 或 refund")
        return v


# ========== 接口 ==========

@app.get("/")
async def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/health/live")
async def health_live():
    """存活探针：进程是否在运行（负载均衡/容器编排用）"""
    return {"status": "ok", "timestamp": time.time()}


@app.get("/health/ready")
async def health_ready():
    """就绪探针：所有依赖是否可用（流量接入前检查）"""
    result = await run_health_checks(getattr(app.state, "checkpointer", None))
    status_code = 200 if result["status"] == "ok" else 503
    return JSONResponse(content=result, status_code=status_code)


@app.get("/health")
async def health():
    """健康检查：兼容旧接口，返回完整检查结果"""
    result = await run_health_checks(getattr(app.state, "checkpointer", None))
    status_code = 200 if result["status"] == "ok" else 503
    return JSONResponse(content=result, status_code=status_code)


@app.get("/metrics")
async def get_metrics():
    """监控指标：请求统计、延迟、错误率、工具调用"""
    return metrics.get_stats()


@app.post("/api/chat")
@limiter.limit(f"{RATE_LIMIT_PER_MINUTE}/minute")
async def chat(request: Request, chat_request: ChatRequest):
    """流式聊天接口，SSE 格式"""
    rid = new_request_id()
    start_time = time.time()
    logger.info(f"收到请求 | session={chat_request.session_id} | 问题={chat_request.message[:50]}")
    metrics.record_request(chat_request.session_id)

    # 安全检查：Prompt 注入 + 内容审核
    security = security_check(chat_request.message)
    if not security["passed"]:
        logger.warning(f"安全拦截 | 原因={security['reason']} | session={chat_request.session_id}")
        metrics.record_security_block()
        metrics.record_failure(chat_request.session_id, time.time() - start_time)
        return JSONResponse(
            status_code=403,
            content={"error": "请求包含不安全内容，已被拦截", "reason": security["reason"]},
        )
    user_message = security["cleaned_input"]

    async def event_generator():
        set_request_id(rid)
        config = {"configurable": {"thread_id": chat_request.session_id}}

        # 1. 意图分类（生产级：3秒超时，低置信度降级）
        intent_result = await classify_intent(user_message)
        yield f"data: {json.dumps({'type': 'intent', 'intent': intent_result.intent.value, 'confidence': round(intent_result.confidence, 2)}, ensure_ascii=False)}\n\n"

        # 2. 直接回复意图（投诉/问候）→ 不走 Agent，省 LLM 调用和延迟
        direct_reply = get_direct_reply(intent_result.intent, user_message)
        if direct_reply:
            logger.info(f"意图直接回复 | {intent_result.intent.value} | 跳过Agent")
            yield f"data: {json.dumps({'type': 'token', 'content': direct_reply}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'messages': [{'type': 'ai', 'content': direct_reply}]}, ensure_ascii=False)}\n\n"
            elapsed = time.time() - start_time
            metrics.record_success(chat_request.session_id, elapsed, [])
            return

        # 3. 读取会话历史
        try:
            state = await app.state.agent.aget_state(config)
            history = [
                {"type": m.type, "content": m.content}
                for m in state.values.get("messages", [])
            ]
            logger.debug(f"历史消息 {len(history)} 条")
        except Exception as e:
            logger.warning(f"读取历史失败: {e}")
            history = []

        # 2. Query Rewrite
        try:
            rewritten = await rewrite_query(user_message, history)
            if rewritten != user_message:
                logger.info(f"Query Rewrite | 原={user_message[:30]} → 改={rewritten[:30]}")
                yield f"data: {json.dumps({'type': 'rewrite', 'original': user_message, 'rewritten': rewritten}, ensure_ascii=False)}\n\n"
        except Exception as e:
            logger.error(f"Query Rewrite 失败: {e}，使用原问题")
            rewritten = user_message

        # 4. 注入意图上下文，帮助 Agent 更快选对工具（减少LLM判断负担）
        intent_context = build_intent_context(intent_result.intent)
        if intent_context:
            rewritten = f"{intent_context}\n\n用户问题：{rewritten}"

        # 5. 调用 Agent（带输出过滤）
        tool_calls = []
        response_text = ""       # 累积所有 token 文本，用于泄露检测
        output_leaked = False    # 是否检测到输出泄露
        try:
            async for event in stream_agent(app.state.agent, rewritten, chat_request.session_id):
                if event["type"] == "tool":
                    tool_calls.append(event["name"])
                    logger.info(f"工具调用 | {event['name']}({json.dumps(event['args'], ensure_ascii=False)})")
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

                elif event["type"] == "tool_result":
                    logger.debug(f"工具结果 | {event['content'][:80]}")
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

                elif event["type"] == "token":
                    # 累积文本，检测系统提示词泄露
                    response_text += event["content"]
                    if not output_leaked:
                        check = filter_output(response_text)
                        if check["leaked"]:
                            output_leaked = True
                            logger.warning(f"输出泄露已拦截 | 已发送{len(response_text)}字 | session={chat_request.session_id}")
                            metrics.record_security_block()
                            # 替换为安全回复
                            yield f"data: {json.dumps({'type': 'token', 'content': check['filtered']}, ensure_ascii=False)}\n\n"
                        else:
                            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                    # 已泄露则丢弃后续 token，不再转发

                elif event["type"] == "done":
                    # 泄露时替换最终消息内容
                    if output_leaked:
                        for msg in event.get("messages", []):
                            if msg["type"] == "ai":
                                msg["content"] = "抱歉，我无法提供相关信息。有什么可以帮您的吗？"
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

                else:
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as e:
            logger.error(f"Agent 执行失败: {e}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'message': '服务暂时不可用，请稍后重试'}, ensure_ascii=False)}\n\n"

        elapsed = time.time() - start_time
        logger.info(f"请求完成 | 耗时={elapsed:.2f}s | 工具调用={tool_calls or '无'}")
        metrics.record_success(chat_request.session_id, elapsed, tool_calls)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/feedback")
@limiter.limit(f"{RATE_LIMIT_PER_MINUTE}/minute")
async def feedback(request: Request, fb_request: FeedbackRequest):
    """用户反馈接口：👍 好评自动加入知识库，👎 差评记录待分析"""
    rid = new_request_id()
    logger.info(f"收到反馈 | rating={fb_request.rating} | Q={fb_request.question[:40]}")

    if fb_request.rating == "up":
        save_good_feedback(fb_request.question, fb_request.answer)
        return {"status": "ok", "message": "已收录到知识库", "good_count": get_good_feedback_count()}
    else:
        save_bad_feedback(fb_request.question, fb_request.answer)
        return {"status": "ok", "message": "已记录，后续优化", "bad_count": get_bad_feedback_count()}


@app.get("/api/feedback/stats")
async def feedback_stats():
    """反馈统计"""
    return {
        "good_count": get_good_feedback_count(),
        "bad_count": get_bad_feedback_count(),
    }


@app.post("/api/hitl/confirm")
async def hitl_confirm(request: HitlConfirmRequest):
    """
    HITL 人工确认接口：用户确认后直接执行危险操作。
    不经过大模型，代码强制执行，防止模型绕过确认直接调用。
    """
    rid = new_request_id()
    logger.info(f"HITL确认 | action={request.action} | order={request.order_id}")

    if request.action == "delete":
        result = execute_delete_order.invoke({"order_id": request.order_id})
    else:  # refund
        result = execute_refund_order.invoke({"order_id": request.order_id})

    logger.info(f"HITL执行结果 | {result[:80]}")
    return {"result": result}


if __name__ == "__main__":
    import uvicorn
    logger.info(f"启动服务 {HOST}:{PORT}")
    uvicorn.run(app, host=HOST, port=PORT, log_level=LOG_LEVEL)
