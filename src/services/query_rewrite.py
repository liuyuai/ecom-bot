"""Query Rewrite：把多轮对话中的省略/代词问题改写成独立完整句"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import HumanMessage, SystemMessage
from services.llm import get_llm


REWRITE_PROMPT = """你是一个查询改写助手。

任务：根据对话历史，把用户当前的问题改写成一个独立、完整、可用于检索的查询。

规则：
1. 补全省略的主语、宾语（如"它""那个""这个"→具体指什么）
2. 去掉语气词和口语化表达（如"呢""啊""吧"）
3. 保留核心意图和关键词
4. 如果问题已经完整清晰，直接返回原问题
5. 只返回改写后的查询，不要解释，不要加引号

对话历史：
{history}

当前问题：{question}

改写后的查询："""


async def rewrite_query(question: str, history: list) -> str:
    """
    根据对话历史改写用户问题。
    history: 之前的消息列表 [{"type": "human"/"ai", "content": "..."}]
    返回：改写后的独立完整问题
    """
    # 没有历史，不需要改写
    if not history:
        return question

    # 只取最近 5 轮，避免上下文过长
    recent_history = history[-10:]
    history_text = "\n".join(
        f"{'用户' if m['type'] == 'human' else '客服'}: {m['content']}"
        for m in recent_history
    )

    prompt = REWRITE_PROMPT.format(history=history_text, question=question)

    llm = get_llm(temperature=0)  # 改写用低温度，保证确定性
    response = await llm.ainvoke([SystemMessage(content="你是一个查询改写助手，只输出改写后的查询。"), HumanMessage(content=prompt)])

    rewritten = response.content.strip()
    # 安全兜底：如果改写结果为空或异常长，用原问题
    if not rewritten or len(rewritten) > 200:
        return question

    return rewritten
