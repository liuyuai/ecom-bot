"""数据访问层：模拟订单、商品、用户数据。
以后接真实数据库/API时，只改这个文件，工具层不用动。
"""
from typing import Optional
from common.logger import get_logger

logger = get_logger("data")


# ========== 模拟数据 ==========

# 订单数据
_ORDERS = {
    "ORD001": {
        "order_id": "ORD001",
        "status": "已发货",
        "product": "iPhone 15 Pro",
        "quantity": 1,
        "amount": 7999.00,
        "create_time": "2026-09-10 14:30",
        "logistics": "顺丰速运 SF1234567890，当前在【上海转运中心】",
        "address": "上海市浦东新区xxx路xxx号",
    },
    "ORD002": {
        "order_id": "ORD002",
        "status": "待收货",
        "product": "小米14",
        "quantity": 1,
        "amount": 3999.00,
        "create_time": "2026-09-12 09:15",
        "logistics": "中通快递 ZT9876543210，已到达【北京朝阳分拣中心】",
        "address": "北京市朝阳区xxx路xxx号",
    },
    "ORD003": {
        "order_id": "ORD003",
        "status": "已签收",
        "product": "AirPods Pro",
        "quantity": 2,
        "amount": 3798.00,
        "create_time": "2026-09-08 16:45",
        "logistics": "已签收，签收人：本人",
        "address": "广州市天河区xxx路xxx号",
    },
}

# 商品数据
_PRODUCTS = {
    "手机": "iPhone 15 Pro：7999元，现货；小米14：3999元，现货；华为Mate 60：6999元，缺货",
    "耳机": "AirPods Pro：1899元，现货；索尼WH-1000XM5：2499元，现货；小米Buds 4：599元，现货",
    "电脑": "MacBook Air M3：8999元，现货；联想小新Pro：5499元，现货；戴尔XPS 13：9999元，缺货",
    "平板": "iPad Pro 11寸：6799元，现货；华为MatePad：3299元，现货；小米平板6：1999元，现货",
    "手表": "Apple Watch Series 9：2999元，现货；华为Watch GT4：1488元，现货",
}


# ========== API 函数（工具层调用这些）==========

def get_order(order_id: str) -> Optional[dict]:
    """查询订单详情。不存在返回 None。"""
    order = _ORDERS.get(order_id)
    if order:
        logger.debug(f"查询订单 | {order_id} → 状态={order['status']}")
    else:
        logger.debug(f"查询订单 | {order_id} → 不存在")
    return order


def search_products(keyword: str) -> str:
    """搜索商品。返回商品信息字符串，未找到返回提示。"""
    for category, info in _PRODUCTS.items():
        if category in keyword or keyword in category:
            logger.debug(f"商品搜索 | {keyword} → 命中 {category}")
            return info
    logger.debug(f"商品搜索 | {keyword} → 未命中")
    return f"未找到与'{keyword}'相关的商品，建议您在商城搜索栏直接搜索"


def delete_order(order_id: str) -> bool:
    """删除订单。返回是否成功。"""
    if order_id in _ORDERS:
        del _ORDERS[order_id]
        logger.info(f"订单已删除 | {order_id}")
        return True
    logger.warning(f"删除失败 | 订单不存在 | {order_id}")
    return False


def refund_order(order_id: str, reason: str = "") -> bool:
    """申请退款。返回是否成功。"""
    if order_id in _ORDERS:
        logger.info(f"退款申请 | {order_id} | 原因={reason or '未填写'}")
        return True
    logger.warning(f"退款失败 | 订单不存在 | {order_id}")
    return False


def order_exists(order_id: str) -> bool:
    """检查订单是否存在。"""
    return order_id in _ORDERS


def get_all_order_ids() -> list:
    """获取所有订单ID（用于测试/调试）。"""
    return list(_ORDERS.keys())
