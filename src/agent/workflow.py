"""Agent 层：LangGraph 工作流编排 + 流式调用"""
from typing import TypedDict, Annotated
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from agent.prompt import ECOM_SYSTEM_PROMPT
from agent.tools import TOOLS
from services.llm import get_llm
from common.logger import get_logger

logger = get_logger("agent.workflow")


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


async def agent_node(state: AgentState):
    """Agent 节点：异步调用大模型，决定调工具还是直接回答"""
    messages = [SystemMessage(content=ECOM_SYSTEM_PROMPT)] + state["messages"]
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
    异步流式调用 Agent，按节点输出执行过程。
    agent_app: 编译好的 LangGraph 应用（由 lifespan 初始化）
    session_id: 会话ID，用于会话隔离
    """
    config = {"configurable": {"thread_id": session_id}}

    async for event in agent_app.astream(
        {"messages": [HumanMessage(content=question)]},
        config=config,
    ):
        for node_name, state_update in event.items():
            if "messages" not in state_update:
                continue
            for msg in state_update["messages"]:
                if msg.type == "ai":
                    if msg.tool_calls:
                        for tc in msg.tool_calls:
                            yield {"type": "tool", "name": tc["name"], "args": tc["args"]}
                    if msg.content:
                        yield {"type": "token", "content": msg.content}
                elif msg.type == "tool":
                    yield {"type": "tool_result", "content": msg.content}

    final_state = await agent_app.aget_state(config)
    serializable_messages = [
        {"type": msg.type, "content": msg.content}
        for msg in final_state.values["messages"]
    ]
    yield {"type": "done", "messages": serializable_messages}
