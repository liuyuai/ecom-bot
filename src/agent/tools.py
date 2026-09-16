"""Agent 层：工具定义"""
from langchain_core.tools import tool
import httpx

from rag.retriever import format_docs, search_knowledge
from data.repository import get_order, search_products, delete_order as data_delete_order, refund_order as data_refund_order, order_exists
from common.logger import get_logger

logger = get_logger("agent.tools")

# 可重试的瞬时错误类型（工具内部 try/except 手动处理，不使用装饰器）
RETRYABLE_EXCEPTIONS = (
    TimeoutError,
    ConnectionError,
    httpx.ConnectError,
    httpx.ReadTimeout,
    httpx.WriteTimeout,
    httpx.PoolTimeout,
    httpx.RemoteProtocolError,
)


@tool
def search_knowledge_base(query: str) -> str:
    """搜索电商知识库，获取售后政策、物流说明、常见问题等信息。
    当用户询问退换货、发货时间、发票、账户等问题时使用。
    """
    try:
        docs = search_knowledge(query, k=3)
        if not docs:
            logger.warning(f"知识库未命中 | query={query}")
            return "知识库中未找到相关信息，请尝试换个说法或联系人工客服。"
        logger.info(f"知识库检索命中 {len(docs)} 条 | query={query}")
        return format_docs(docs)
    except RETRYABLE_EXCEPTIONS as e:
        # 瞬时错误：Reranker/Embedding API 超时或网络问题
        logger.error(f"知识库检索瞬时错误 | query={query} | error={e}")
        return f"[错误:超时] 知识库检索暂时不可用（{type(e).__name__}），可以稍后重试。"
    except Exception as e:
        logger.error(f"知识库检索失败 | query={query} | error={e}")
        return f"[错误:系统异常] 知识库检索失败（{type(e).__name__}），请联系人工客服。"


@tool
def query_order(order_id: str) -> str:
    """查询订单状态和物流信息。需要用户提供订单号。
    当用户问"我的订单到哪了""什么时候发货""物流状态"时使用。
    """
    try:
        order = get_order(order_id)
        if not order:
            # 永久错误：订单不存在，重试也没用
            return f"[错误:不存在] 未找到订单 {order_id}，请确认订单号是否正确。"
        return (
            f"订单 {order['order_id']}\n"
            f"商品：{order['product']} x{order['quantity']}\n"
            f"金额：{order['amount']}元\n"
            f"状态：{order['status']}\n"
            f"下单时间：{order['create_time']}\n"
            f"物流：{order['logistics']}"
        )
    except RETRYABLE_EXCEPTIONS as e:
        # 瞬时错误：订单 API 超时，可重试
        logger.error(f"订单查询瞬时错误 | order={order_id} | error={e}")
        return f"[错误:超时] 订单查询暂时不可用（{type(e).__name__}），可以稍后重试。"
    except Exception as e:
        return f"[错误:系统异常] 订单查询失败（{type(e).__name__}），请稍后重试或联系人工客服。"


@tool
def product_search(keyword: str) -> str:
    """搜索商品信息、价格、库存。
    当用户询问任何商品相关问题时必须调用此工具，包括但不限于：
    "有没有XXX"、"XXX多少钱"、"推荐XXX"、"XXX有货吗"、"XXX怎么样"。
    不要凭常识直接回答商品问题，必须先调用此工具查询。
    """
    try:
        return search_products(keyword)
    except RETRYABLE_EXCEPTIONS as e:
        logger.error(f"商品搜索瞬时错误 | keyword={keyword} | error={e}")
        return f"[错误:超时] 商品搜索暂时不可用（{type(e).__name__}），可以稍后重试。"
    except Exception as e:
        return f"[错误:系统异常] 商品搜索失败（{type(e).__name__}），请稍后重试。"


@tool
def delete_order(order_id: str) -> str:
    """申请删除/取消订单。这是危险操作，调用后不会立即删除，
    而是返回确认请求，等待用户明确确认后才能执行。
    当用户说"删除订单""取消订单""删掉订单XXX"时调用此工具。
    """
    if not order_exists(order_id):
        return f"[错误:不存在] 订单 {order_id} 不存在，无法删除。"

    return (
        f"[HITL_CONFIRM] 即将删除订单 {order_id}。\n"
        f"⚠️ 此操作不可逆，删除后订单数据将无法恢复。\n"
        f"请确认是否删除？"
    )


@tool
def execute_delete_order(order_id: str) -> str:
    """执行删除订单。仅在用户明确确认"确认删除"后调用此工具。
    不要在未确认的情况下调用此工具。
    """
    try:
        success = data_delete_order(order_id)
        if success:
            return f"✅ 订单 {order_id} 已成功删除。"
        else:
            return f"[错误:不存在] 删除失败，订单 {order_id} 不存在。"
    except Exception as e:
        return f"[错误:系统异常] 删除失败（{type(e).__name__}），请稍后重试。"


@tool
def refund_order(order_id: str, reason: str = "") -> str:
    """申请订单退款。这是危险操作，调用后不会立即退款，
    而是返回确认请求，等待用户明确确认后才能执行。
    当用户说"退款""退钱""申请退款"时调用此工具。
    """
    if not order_exists(order_id):
        return f"[错误:不存在] 订单 {order_id} 不存在，无法退款。"

    reason_text = f"\n退款原因：{reason}" if reason else ""
    return (
        f"[HITL_CONFIRM] 即将为订单 {order_id} 申请退款。{reason_text}\n"
        f"⚠️ 退款将原路返回，1-3个工作日到账。\n"
        f"请确认是否退款？"
    )


@tool
def execute_refund_order(order_id: str) -> str:
    """执行订单退款。仅在用户明确确认"确认退款"后调用此工具。
    不要在未确认的情况下调用此工具。
    """
    try:
        success = data_refund_order(order_id)
        if success:
            return f"✅ 订单 {order_id} 退款申请已提交，预计1-3个工作日原路退回。"
        else:
            return f"[错误:不存在] 退款失败，订单 {order_id} 不存在。"
    except Exception as e:
        return f"[错误:系统异常] 退款失败（{type(e).__name__}），请稍后重试。"


# 工具列表，传给大模型
# 注意：execute_delete_order / execute_refund_order 不在列表中——
# 这两个是危险操作，由后端 /api/hitl/confirm 直接执行，不暴露给大模型
TOOLS = [
    search_knowledge_base,
    query_order,
    product_search,
    delete_order,
    refund_order,
]
