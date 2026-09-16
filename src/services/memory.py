"""会话记忆管理：滑动窗口 + 摘要压缩

策略：保留最近 N 轮完整对话，更早的用 LLM 总结成一段摘要。
客服对话一般不超过 20 轮，此方案性价比最高。
"""
from langchain_core.messages import SystemMessage, HumanMessage, AnyMessage

from services.llm import get_llm
from common.logger import get_logger

logger = get_logger("memory")

# 保留最近多少轮对话（1轮 = 1条用户消息 + 对应的回复）
MAX_RECENT_TURNS = 10


async def compress_messages(messages: list[AnyMessage]) -> list[AnyMessage]:
    """滑动窗口 + 摘要：超过 MAX_RECENT_TURNS 轮时，把旧对话压缩成摘要。

    返回压缩后的消息列表，不修改原 state（完整历史仍存在 checkpointer 里）。
    """
    # 统计用户消息数 = 轮数
    user_msgs = [m for m in messages if m.type == "human"]
    if len(user_msgs) <= MAX_RECENT_TURNS:
        return messages

    # 找到截断点：保留最后 MAX_RECENT_TURNS 轮用户消息及之后的所有消息
    cutoff_user = user_msgs[len(user_msgs) - MAX_RECENT_TURNS]
    cutoff_idx = messages.index(cutoff_user)

    old_messages = messages[:cutoff_idx]
    recent_messages = messages[cutoff_idx:]

    if not old_messages:
        return messages

    # 分离已有摘要和待总结的旧消息
    existing_summary = ""
    old_to_summarize = []
    for m in old_messages:
        if isinstance(m, SystemMessage) and isinstance(m.content, str) and m.content.startswith("[对话摘要]"):
            existing_summary = m.content.replace("[对话摘要] ", "")
        else:
            old_to_summarize.append(m)

    # 没有新内容需要总结，直接复用已有摘要
    if not old_to_summarize:
        if existing_summary:
            return [SystemMessage(content=f"[对话摘要] {existing_summary}")] + recent_messages
        return recent_messages

    # 构建待总结的对话文本
    lines = []
    for m in old_to_summarize:
        if m.type == "human":
            role = "用户"
        elif m.type == "ai":
            role = "客服"
        elif m.type == "tool":
            role = "工具结果"
        else:
            role = m.type
        content = m.content if isinstance(m.content, str) else str(m.content)
        lines.append(f"{role}: {content[:300]}")
    conversation_text = "\n".join(lines)

    # 调用 LLM 更新摘要（temperature=0 保证确定性）
    llm = get_llm(temperature=0)
    prompt = (
        "请更新对话摘要。保留关键信息：订单号、用户诉求、已解决的问题、待处理事项。"
        "不超过200字，不要包含工具调用细节。\n\n"
        f"已有摘要：{existing_summary or '（无）'}\n\n"
        f"新增对话：\n{conversation_text}\n\n"
        "更新后的摘要："
    )
    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        new_summary = f"[对话摘要] {response.content}"
        logger.info(f"对话摘要已更新 | 旧消息{len(old_to_summarize)}条 → 摘要")
    except Exception as e:
        logger.warning(f"摘要生成失败，降级为仅保留最近对话: {e}")
        return recent_messages

    return [SystemMessage(content=new_summary)] + recent_messages
