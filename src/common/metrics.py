"""监控指标模块：请求统计、延迟、错误率、工具调用统计"""
import time
import threading
from collections import defaultdict
from common.logger import get_logger

logger = get_logger("metrics")


class MetricsCollector:
    """
    内存指标收集器。
    单实例部署够用；多实例时可迁到 Redis/Prometheus。
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._reset()

    def _reset(self):
        self.start_time = time.time()
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.request_latencies = []  # 保留最近1000条延迟
        self.tool_call_counts = defaultdict(int)
        self.active_sessions = set()
        self.security_blocks = 0
        self.rate_limited = 0
        self.token_usage = 0  # 估算的 token 消耗

    def record_request(self, session_id: str):
        """记录一次请求开始"""
        with self._lock:
            self.total_requests += 1
            self.active_sessions.add(session_id)

    def record_success(self, session_id: str, latency: float, tools: list = None):
        """记录请求成功"""
        with self._lock:
            self.successful_requests += 1
            self.request_latencies.append(latency)
            if len(self.request_latencies) > 1000:
                self.request_latencies.pop(0)
            if tools:
                for t in tools:
                    self.tool_call_counts[t] += 1

    def record_failure(self, session_id: str, latency: float):
        """记录请求失败"""
        with self._lock:
            self.failed_requests += 1
            self.request_latencies.append(latency)
            if len(self.request_latencies) > 1000:
                self.request_latencies.pop(0)

    def record_security_block(self):
        """记录安全拦截"""
        with self._lock:
            self.security_blocks += 1

    def record_rate_limit(self):
        """记录限流触发"""
        with self._lock:
            self.rate_limited += 1

    def record_tokens(self, tokens: int):
        """记录 token 消耗（估算）"""
        with self._lock:
            self.token_usage += tokens

    def get_stats(self) -> dict:
        """获取统计快照"""
        with self._lock:
            uptime = time.time() - self.start_time
            latencies = sorted(self.request_latencies)
            avg_latency = sum(latencies) / len(latencies) if latencies else 0
            p95_latency = latencies[int(len(latencies) * 0.95)] if latencies else 0
            p99_latency = latencies[int(len(latencies) * 0.99)] if latencies else 0
            error_rate = (self.failed_requests / self.total_requests * 100) if self.total_requests > 0 else 0
            qps = self.total_requests / uptime if uptime > 0 else 0

            return {
                "uptime_seconds": round(uptime, 1),
                "uptime_human": f"{int(uptime // 3600)}h {int((uptime % 3600) // 60)}m {int(uptime % 60)}s",
                "total_requests": self.total_requests,
                "successful_requests": self.successful_requests,
                "failed_requests": self.failed_requests,
                "error_rate_percent": round(error_rate, 2),
                "qps": round(qps, 2),
                "avg_latency_ms": round(avg_latency * 1000, 1),
                "p95_latency_ms": round(p95_latency * 1000, 1),
                "p99_latency_ms": round(p99_latency * 1000, 1),
                "active_sessions": len(self.active_sessions),
                "security_blocks": self.security_blocks,
                "rate_limited": self.rate_limited,
                "estimated_token_usage": self.token_usage,
                "tool_call_counts": dict(self.tool_call_counts),
            }


# 全局单例
metrics = MetricsCollector()
