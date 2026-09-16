"""意图路由服务：生产级意图分类 + 路由决策

分类器用 LLM（temperature=0）做确定性分类，返回意图 + 置信度。
路由层根据意图决定走 Agent 还是直接回复。
"""
import json
import time
from enum import Enum
from dataclasses import dataclass

from services.llm import get_llm
from common.logger import get_logger

logger = get_logger("intent")


class Intent(str, Enum):
    """意图枚举"""
    PRODUCT_INQUIRY = "product_inquiry"      # 商品咨询
    ORDER_QUERY = "order_query"              # 订单查询
    AFTER_SALES_POLICY = "after_sales_policy"  # 售后政策
    REFUND_REQUEST = "refund_request"        # 退款申请
    DELETE_REQUEST = "delete_request"        # 删除/取消订单
    COMPLAINT = "complaint"                  # 投诉抱怨
    GREETING = "greeting"                    # 问候闲聊
    UNKNOWN = "unknown"                      # 其他/不确定


# 意图描述，用于分类 Prompt
INTENT_DESCRIPTIONS = {
    Intent.PRODUCT_INQUIRY: "用户询问商品信息，包括有没有某商品、价格、库存、推荐商品、商品规格参数等",
    Intent.ORDER_QUERY: "用户询问订单状态、物流进度、订单详情，需要订单号",
    Intent.AFTER_SALES_POLICY: "用户咨询售后政策，包括退货规则、退款流程、换货、运费、发票、保修、优惠券使用等规则类问题",
    Intent.REFUND_REQUEST: "用户明确要求退款、退钱、申请退款，是操作请求而非政策咨询",
    Intent.DELETE_REQUEST: "用户要求删除订单、取消订单、撤销订单，是操作请求",
    Intent.COMPLAINT: "用户表达不满、抱怨、投诉、威胁差评、要求赔偿、情绪激动",
    Intent.GREETING: "用户打招呼、问候、感谢、道别、闲聊，不涉及具体业务",
    Intent.UNKNOWN: "无法明确归类，或同时涉及多个意图",
}

# 需要直接回复、不走 Agent 的意图
DIRECT_REPLY_INTENTS = {Intent.COMPLAINT, Intent.GREETING}

# 分类超时（秒）
CLASSIFY_TIMEOUT = 3


INTENT_CLASSIFY_PROMPT = """你是一个电商客服的意图分类器。请判断用户消息属于哪个意图类别。

意图类别及定义：
{intent_list}

用户消息："{user_message}"

请以JSON格式返回，不要返回其他文字：
{{"intent": "意图类别", "confidence": 0.0到1.0的置信度, "reason": "简短判断依据"}}"""


@dataclass
class IntentResult:
    """意图分类结果"""
    intent: Intent
    confidence: float
    reason: str
    latency_ms: int


async def classify_intent(message: str) -> IntentResult:
    """对用户消息做意图分类

    生产级特性：
    - temperature=0 保证确定性
    - 3秒超时，超时降级为 UNKNOWN
    - 置信度 < 0.6 降级为 UNKNOWN（走Agent兜底）
    """
    start = time.time()
    intent_list = "\n".join(
        f"- {k.value}: {v}" for k, v in INTENT_DESCRIPTIONS.items()
    )
    prompt = INTENT_CLASSIFY_PROMPT.format(
        intent_list=intent_list,
        user_message=message[:500],
    )

    try:
        import asyncio
        from langchain_core.messages import HumanMessage
        llm = get_llm(temperature=0)
        response = await asyncio.wait_for(
            llm.ainvoke([HumanMessage(content=prompt)]),
            timeout=CLASSIFY_TIMEOUT,
        )
        result = _parse_classify_response(response.content)
        latency = round((time.time() - start) * 1000)

        # 低置信度降级
        if result.confidence < 0.6:
            logger.info(f"意图低置信度降级 | {result.intent.value}({result.confidence:.2f}) → unknown")
            result.intent = Intent.UNKNOWN

        logger.info(f"意图分类 | {result.intent.value} | 置信度={result.confidence:.2f} | {latency}ms | {result.reason}")
        result.latency_ms = latency
        return result

    except Exception as e:
        latency = round((time.time() - start) * 1000)
        logger.warning(f"意图分类失败，降级unknown: {e} | {latency}ms")
        return IntentResult(
            intent=Intent.UNKNOWN,
            confidence=0.0,
            reason=f"分类失败: {type(e).__name__}",
            latency_ms=latency,
        )


def _parse_classify_response(text: str) -> IntentResult:
    """解析 LLM 返回的分类结果"""
    # 尝试直接解析 JSON
    try:
        data = json.loads(text)
    except Exception:
        # 提取 JSON 块
        import re
        match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
            except Exception:
                data = {}
        else:
            data = {}

    intent_str = data.get("intent", "unknown")
    confidence = float(data.get("confidence", 0.5))
    reason = data.get("reason", "")

    # 验证意图合法性
    try:
        intent = Intent(intent_str)
    except ValueError:
        intent = Intent.UNKNOWN

    return IntentResult(
        intent=intent,
        confidence=confidence,
        reason=reason,
        latency_ms=0,
    )


def get_direct_reply(intent: Intent, message: str) -> str | None:
    """对于不需要走 Agent 的意图，直接生成回复

    返回 None 表示需要走 Agent。
    """
    if intent == Intent.COMPLAINT:
        return (
            "非常抱歉给您带来不好的体验！您的反馈我们非常重视，"
            "我马上为您转接人工客服，由专属客服经理为您处理，请稍等片刻～"
        )
    if intent == Intent.GREETING:
        return (
            "亲，您好呀～欢迎光临！我是您的专属客服小助手，"
            "有什么可以帮您的吗？无论是商品咨询、订单物流还是退换货问题，都可以随时告诉我哦～"
        )
    return None


def build_intent_context(intent: Intent) -> str:
    """根据意图构建注入 Agent 的上下文提示，帮助 LLM 更快选对工具"""
    context_map = {
        Intent.PRODUCT_INQUIRY: "【意图：商品咨询】应优先调用 product_search 工具查询商品信息。",
        Intent.ORDER_QUERY: "【意图：订单查询】应优先调用 query_order 工具，需要订单号时主动询问。",
        Intent.AFTER_SALES_POLICY: "【意图：售后政策】应优先调用 search_knowledge_base 工具查询相关政策，严格按知识库回答。",
        Intent.REFUND_REQUEST: "【意图：退款申请】应调用 refund_order 工具发起退款确认流程。",
        Intent.DELETE_REQUEST: "【意图：删除订单】应调用 delete_order 工具发起删除确认流程。",
    }
    return context_map.get(intent, "")
