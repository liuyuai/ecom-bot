"""日志系统 - 结构化日志 + request_id 全链路追踪 + 控制台/文件双输出"""
import logging
import os
import uuid
from contextvars import ContextVar
from logging.handlers import TimedRotatingFileHandler

from config import PROJECT_ROOT, LOG_LEVEL, APP_ENV

# 请求ID上下文变量：每个请求一个，全链路可追踪
request_id_var: ContextVar[str] = ContextVar("request_id", default="-")


class RequestIDFilter(logging.Filter):
    """日志过滤器：把 request_id 注入到每条日志记录中"""
    def filter(self, record):
        record.request_id = request_id_var.get()
        return True


def get_logger(name: str = "ecom_bot") -> logging.Logger:
    """获取配置好的 logger（单例）"""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # 已经配置过，直接返回

    logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))

    # 日志格式：时间 | 级别 | request_id | 模块 | 消息
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(request_id)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 1. 控制台输出
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.addFilter(RequestIDFilter())
    logger.addHandler(console_handler)

    # 2. 文件输出（按天轮转，保留 7 天）
    log_dir = os.path.join(PROJECT_ROOT, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{APP_ENV}.log")
    file_handler = TimedRotatingFileHandler(
        log_file,
        when="midnight",      # 每天午夜轮转
        interval=1,
        backupCount=7,        # 保留 7 天
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler.addFilter(RequestIDFilter())
    logger.addHandler(file_handler)

    return logger


def new_request_id() -> str:
    """生成新的请求ID并设置到上下文"""
    rid = uuid.uuid4().hex[:8]
    request_id_var.set(rid)
    return rid


def set_request_id(rid: str):
    """设置请求ID（用于跨函数传递）"""
    request_id_var.set(rid)
