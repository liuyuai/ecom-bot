"""安全模块：Prompt 注入防护 + 内容审核 + 输出过滤"""
import re
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.logger import get_logger

logger = get_logger("security")

# ========== Prompt 注入检测 ==========

# 常见注入模式（正则匹配）
INJECTION_PATTERNS = [
    # 要求忽略/覆盖系统指令
    r"(忽略|无视|忘记|跳过|不要管|别管).{0,10}(之前|先前|上面|上述|系统|所有).{0,10}(指令|提示|规则|要求|设定)",
    r"(ignore|forget|disregard|skip).{0,10}(previous|above|system|all).{0,10}(instructions|prompt|rules|directives)",
    # 要求输出系统提示词
    r"(输出|显示|打印|告诉我|发我|重复|给我|发给我).{0,10}(系统提示|系统指令|prompt|system prompt|初始提示|你的提示)",
    r"(系统提示|系统指令|你的提示|初始指令|prompt).{0,10}(发给我|给我|告诉我|输出|显示|打印|重复)",
    r"(reveal|print|show|repeat|output).{0,10}(system prompt|your instructions|your prompt|initial prompt)",
    # 角色扮演覆盖
    r"(现在|从现在开始|接下来).{0,10}(你是|扮演|变成|作为).{0,10}(黑客|攻击工具|恶意软件|病毒|不受限制|没有限制)",
    r"(now|from now on).{0,10}(you are|act as|become|pretend to be).{0,10}(hacker|unrestricted|no limits|DAN|jailbreak)",
    # DAN / jailbreak 经典手法
    r"(DAN|do anything now|jailbreak|developer mode|god mode)",
    r"(解除限制|打破规则|绕过限制|越狱模式|开发者模式|上帝模式)",
    # 要求不遵循安全策略
    r"(不要|别|禁止).{0,10}(遵守|遵循|执行).{0,10}(安全|策略|规则|政策)",
    r"(do not|don't|never).{0,10}(follow|comply with|obey).{0,10}(safety|policy|rules|guidelines)",
]

# 编译正则
_COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]


def detect_prompt_injection(text: str) -> dict:
    """
    检测用户输入是否包含 Prompt 注入。
    返回 {"detected": bool, "patterns": [匹配到的模式], "risk": "high"/"medium"/"low"}
    """
    if not text or len(text) < 3:
        return {"detected": False, "patterns": [], "risk": "low"}

    matched = []
    for pattern in _COMPILED_PATTERNS:
        if pattern.search(text):
            matched.append(pattern.pattern[:50])

    # 涉及系统提示词泄露的直接算高风险
    prompt_leak = any("系统提示" in p or "系统指令" in p or "prompt" in p.lower() for p in matched)

    if len(matched) >= 2 or prompt_leak:
        risk = "high"
    elif len(matched) == 1:
        risk = "medium"
    else:
        risk = "low"

    return {"detected": len(matched) > 0, "patterns": matched, "risk": risk}


def sanitize_input(text: str) -> str:
    """
    输入清洗：去除可能的注入手法。
    不直接删除内容（可能误杀正常对话），而是做标记和隔离。
    """
    # 去除零宽字符和不可见字符（注入常用隐藏手段）
    text = re.sub(r'[\u200b-\u200f\u202a-\u202e\ufeff]', '', text)
    # 去除控制字符（保留换行和制表）
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    return text.strip()


# ========== 内容审核 ==========

# 敏感词（简化版，生产环境应接入专业内容审核 API）
SENSITIVE_PATTERNS = [
    r"(暴力|恐怖|袭击|爆炸|杀人|自残|自杀)",
    r"(色情|淫秽|裸体|性服务|嫖娼)",
    r"(毒品|吸毒|贩毒|冰毒|海洛因)",
    r"(赌博|赌球|六合彩|时时彩)",
    r"(诈骗|洗钱|非法集资|传销)",
]
_COMPILED_SENSITIVE = [re.compile(p, re.IGNORECASE) for p in SENSITIVE_PATTERNS]


def content_moderation(text: str) -> dict:
    """
    内容审核：检测输入是否包含违规内容。
    返回 {"safe": bool, "categories": [命中类别]}
    """
    if not text:
        return {"safe": True, "categories": []}

    categories = []
    for pattern in _COMPILED_SENSITIVE:
        if pattern.search(text):
            categories.append(pattern.pattern[:20])

    return {"safe": len(categories) == 0, "categories": categories}


# ========== 输出过滤 ==========

# 系统提示词泄露检测
LEAK_PATTERNS = [
    r"(你是一个|你的职责|你的任务|系统提示|system prompt|你的设定|你的角色).{0,50}(电商客服|assistant|AI)",
    r"(ECOM_SYSTEM_PROMPT|系统提示词|初始指令|你被要求|你的规则)",
]
_COMPILED_LEAK = [re.compile(p, re.IGNORECASE) for p in LEAK_PATTERNS]


def filter_output(text: str) -> dict:
    """
    输出过滤：检测模型输出是否泄露了系统提示词或敏感信息。
    返回 {"safe": bool, "filtered": str, "leaked": bool}
    """
    if not text:
        return {"safe": True, "filtered": text, "leaked": False}

    leaked = any(p.search(text) for p in _COMPILED_LEAK)

    if leaked:
        # 不直接返回原文，替换为安全回复
        logger.warning(f"检测到输出可能泄露系统信息，已拦截")
        return {
            "safe": False,
            "filtered": "抱歉，我无法提供相关信息。有什么可以帮您的吗？",
            "leaked": True,
        }

    return {"safe": True, "filtered": text, "leaked": False}


# ========== 综合安全检查 ==========

def security_check(user_input: str) -> dict:
    """
    综合安全检查：注入检测 + 内容审核 + 输入清洗。
    在请求进入 Agent 之前调用。
    """
    # 1. 清洗输入
    cleaned = sanitize_input(user_input)

    # 2. 注入检测
    injection = detect_prompt_injection(cleaned)

    # 3. 内容审核
    moderation = content_moderation(cleaned)

    # 4. 综合判断
    blocked = injection["risk"] == "high" or not moderation["safe"]
    reason = []
    if injection["detected"]:
        reason.append(f"Prompt注入风险({injection['risk']})")
    if not moderation["safe"]:
        reason.append(f"内容违规({','.join(moderation['categories'])})")

    result = {
        "passed": not blocked,
        "cleaned_input": cleaned,
        "blocked": blocked,
        "reason": "; ".join(reason) if reason else "",
        "injection_risk": injection["risk"],
        "content_safe": moderation["safe"],
    }

    if blocked:
        logger.warning(f"安全拦截 | 原因={result['reason']} | 输入={cleaned[:50]}")
    elif injection["detected"]:
        logger.info(f"注入风险(中低) | 风险={injection['risk']} | 输入={cleaned[:50]}")

    return result
