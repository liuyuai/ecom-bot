"""服务层：大模型封装"""
from langchain_openai import ChatOpenAI
from config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, LLM_TEMPERATURE, LLM_TIMEOUT


def get_llm(temperature=LLM_TEMPERATURE):
    """获取大模型实例。DeepSeek 兼容 OpenAI API 格式，直接用 ChatOpenAI。"""
    return ChatOpenAI(
        model=LLM_MODEL,
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        temperature=temperature,
        timeout=LLM_TIMEOUT,
        max_retries=2,
    )
