"""Agent 层：LangGraph 工作流编排 + 流式调用"""
from typing import TypedDict, Annotated
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from agent.prompt import ECOM_SYSTEM_PROMPT
from agent.tools import TOOLS
from services.llm import get_llm
from services.memory import compress_messages
from common.logger import get_logger

logger = get_logger("agent.workflow")


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


async def agent_node(state: AgentState):
    """Agent 节点：异步调用大模型，决定调工具还是直接回答"""
    # 滑动窗口+摘要：超过10轮时压缩旧对话，减少 token
    compressed = await compress_messages(state["messages"])
    messages = [SystemMessage(content=ECOM_SYSTEM_PROMPT)] + compressed
    llm = get_llm()
    llm_with_tools = llm.bind_tools(TOOLS)
    response = await llm_with_tools.ainvoke(messages)
    return {"messages": [response]}


tool_node = ToolNode(TOOLS)


def should_continue(state: AgentState) -> str:
    """路由函数：判断大模型是要调工具还是已经回答完了"""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    else:
        return END


def build_workflow():
    """构建 Agent 工作流（不含 checkpointer，由调用方注入）"""
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", END: END},
    )
    workflow.add_edge("tools", "agent")
    return workflow


async def stream_agent(agent_app, question: str, session_id: str):
    """
    真·流式调用 Agent：用 astream_events 穿透节点，拿到 LLM 逐 token 事件。
    agent_app: 编译好的 LangGraph 应用（由 lifespan 初始化）
    session_id: 会话ID，用于会话隔离
    """
    config = {"configurable": {"thread_id": session_id}}

    async for event in agent_app.astream_events(
        {"messages": [HumanMessage(content=question)]},
        config=config,
        version="v2",
    ):
        event_type = event["event"]

        # LLM 逐 token 输出文本
        if event_type == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            # 纯文本 token（不是工具调用的 chunk）
            if chunk.content:
                yield {"type": "token", "content": chunk.content}

        # 工具开始执行（此时工具名和参数已完整）
        elif event_type == "on_tool_start":
            tool_name = event.get("name", "")
            tool_input = event.get("data", {}).get("input", {})
            # 过滤掉 LangGraph 内部节点，只上报我们的业务工具
            if tool_name and not tool_name.startswith("LangGraph"):
                yield {"type": "tool", "name": tool_name, "args": tool_input}

        # 工具执行完毕
        elif event_type == "on_tool_end":
            tool_output = event.get("data", {}).get("output", "")
            if tool_output:
                yield {"type": "tool_result", "content": tool_output}

    # 流结束后拿最终 state
    final_state = await agent_app.aget_state(config)
    serializable_messages = [
        {"type": msg.type, "content": msg.content}
        for msg in final_state.values["messages"]
    ]
    yield {"type": "done", "messages": serializable_messages}
