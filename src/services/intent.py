"""意图路由服务：生产级意图分类 + 路由决策

分类策略：规则前置过滤（零成本）→ LLM 分类（兜底）
- 问候、投诉等简单意图用正则匹配，不调 LLM
- 匹配不上的才调 LLM 做精细分类
"""
import json
import re
import time
from enum import Enum
from dataclasses import dataclass

from services.llm import get_llm
from config import SMALL_LLM_CONFIDENCE_THRESHOLD
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

# ============================================================
# 规则前置过滤：零成本匹配简单意图，匹配不上才调 LLM
# ============================================================

# 问候模式：打招呼、感谢、道别
GREETING_PATTERNS = [
    r"^(你好|您好|hi|hello|hey|在吗|在不在|有人吗|在么|在不)",
    r"^(谢谢|感谢|多谢|谢谢啦|谢谢哦|感谢感谢|thx|thanks)",
    r"^(再见|拜拜|88|bye|goodbye|先走了|下次聊|回见)",
    r"^(嗯|哦|好的|好|ok|okay|行|可以|没问题)$",
]

# 投诉模式：表达不满、威胁、要求赔偿
COMPLAINT_PATTERNS = [
    r"(投诉|举报|12315|消协|消费者协会)",
    r"(垃圾|坑爹|骗人|骗子|什么玩意|什么鬼|扯淡|扯犊子)",
    r"(差评|给差评|一星|打一星)",
    r"(气死我了|太过分了|什么态度|什么服务|垃圾服务)",
    r"(赔偿|赔钱|补偿|退一赔三|假一赔十)",
    r"(我要告|起诉|法院|律师)",
]

# 预编译正则
_greeting_re = [re.compile(p, re.IGNORECASE) for p in GREETING_PATTERNS]
_complaint_re = [re.compile(p, re.IGNORECASE) for p in COMPLAINT_PATTERNS]


def rule_based_classify(message: str) -> Intent | None:
    """规则前置过滤：用正则匹配简单意图

    返回 None 表示规则没匹配上，需要调 LLM 分类。
    匹配顺序：先投诉（高优先级），再问候。
    """
    text = message.strip()

    # 投诉优先（投诉中可能包含问候词，如"你好，我要投诉"）
    for pattern in _complaint_re:
        if pattern.search(text):
            return Intent.COMPLAINT

    # 问候：必须是纯问候，不能包含业务关键词
    for pattern in _greeting_re:
        if pattern.match(text):
            # 二次确认：如果消息里有业务关键词，不算纯问候
            business_keywords = ["订单", "退款", "退货", "商品", "物流", "发票", "价格", "发货", "质量"]
            if not any(kw in text for kw in business_keywords):
                return Intent.GREETING

    return None


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

    三级级联策略：
    1. 规则前置过滤（零成本）→ 命中直接返回
    2. 小模型分类（便宜快速）→ 置信度≥阈值直接返回
    3. 主 LLM 分类（兜底）→ 小模型低置信度或失败时调用

    生产级特性：
    - 规则匹配成功不调任何模型，零成本零延迟
    - 小模型用硅基流动 7B 模型，成本约为主模型的 1/10
    - 小模型置信度不足时才调主 LLM，大部分请求止步于前两级
    - 任何一级失败都自动降级到下一级，不阻塞请求
    """
    start = time.time()

    # ===== 第一级：规则前置过滤（零成本）=====
    rule_intent = rule_based_classify(message)
    if rule_intent is not None:
        latency = round((time.time() - start) * 1000)
        logger.info(f"意图分类(规则) | {rule_intent.value} | 置信度=1.00 | {latency}ms | 正则匹配")
        return IntentResult(
            intent=rule_intent,
            confidence=1.0,
            reason="规则匹配",
            latency_ms=latency,
        )

    # ===== 第二级：小模型分类（便宜快速）=====
    try:
        small_result = await _classify_with_small_model(message)
        if small_result.confidence >= SMALL_LLM_CONFIDENCE_THRESHOLD:
            latency = round((time.time() - start) * 1000)
            logger.info(f"意图分类(小模型) | {small_result.intent.value} | 置信度={small_result.confidence:.2f} | {latency}ms | {small_result.reason}")
            small_result.latency_ms = latency
            return small_result
        else:
            logger.info(f"小模型低置信度({small_result.confidence:.2f})，升级主LLM | {small_result.intent.value}")
    except Exception as e:
        logger.warning(f"小模型分类失败，升级主LLM: {type(e).__name__}")

    # ===== 第三级：主 LLM 分类（兜底）=====
    try:
        result = await _classify_with_main_llm(message)
        latency = round((time.time() - start) * 1000)

        # 低置信度降级
        if result.confidence < 0.6:
            logger.info(f"意图低置信度降级 | {result.intent.value}({result.confidence:.2f}) → unknown")
            result.intent = Intent.UNKNOWN

        logger.info(f"意图分类(主LLM) | {result.intent.value} | 置信度={result.confidence:.2f} | {latency}ms | {result.reason}")
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


def _build_classify_prompt(message: str) -> str:
    """构建分类 Prompt（小模型和主 LLM 共用）"""
    intent_list = "\n".join(
        f"- {k.value}: {v}" for k, v in INTENT_DESCRIPTIONS.items()
    )
    return INTENT_CLASSIFY_PROMPT.format(
        intent_list=intent_list,
        user_message=message[:500],
    )


async def _classify_with_small_model(message: str) -> IntentResult:
    """用小模型（硅基流动 7B）做意图分类"""
    import asyncio
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage
    from config import SMALL_LLM_API_KEY, SMALL_LLM_BASE_URL, SMALL_LLM_MODEL

    small_llm = ChatOpenAI(
        model=SMALL_LLM_MODEL,
        api_key=SMALL_LLM_API_KEY,
        base_url=SMALL_LLM_BASE_URL,
        temperature=0,
        timeout=CLASSIFY_TIMEOUT,
        max_retries=1,
    )
    prompt = _build_classify_prompt(message)
    response = await asyncio.wait_for(
        small_llm.ainvoke([HumanMessage(content=prompt)]),
        timeout=CLASSIFY_TIMEOUT,
    )
    return _parse_classify_response(response.content)


async def _classify_with_main_llm(message: str) -> IntentResult:
    """用主 LLM（DeepSeek）做意图分类"""
    import asyncio
    from langchain_core.messages import HumanMessage

    llm = get_llm(temperature=0)
    prompt = _build_classify_prompt(message)
    response = await asyncio.wait_for(
        llm.ainvoke([HumanMessage(content=prompt)]),
        timeout=CLASSIFY_TIMEOUT,
    )
    return _parse_classify_response(response.content)


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
