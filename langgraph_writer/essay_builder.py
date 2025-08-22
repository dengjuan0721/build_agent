# 文件: essay_writer/essay_builder.py

import os
from typing import TypedDict, List
from tavily import TavilyClient
from pydantic import BaseModel

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_deepseek import ChatDeepSeek

# 从我们的 prompts 模块导入所有 prompt
from .prompts import (
    PLAN_PROMPT, WRITER_PROMPT, REFLECTION_PROMPT,
    RESEARCH_PLAN_PROMPT, RESEARCH_CRITIQUE_PROMPT
)


# --- 定义状态和模型 ---

class AgentState(TypedDict):
    task: str
    plan: str
    draft: str
    critique: str
    content: List[str]
    revision_number: int
    max_revisions: int


class Queries(BaseModel):
    queries: List[str]


class EssayWriterGraph:
    """
    一个封装了完整的 LangGraph 论文写作流程的类。
    """

    def __init__(self, conn_string=":memory:"):
        self.llm = ChatDeepSeek(
            model="deepseek-reasoner",
            openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
            openai_api_base=os.getenv("DEEPSEEK_API_BASE"),
            temperature=0.0
        )
        self.tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
        self.conn_string = conn_string
        self.builder = self._build_graph()
        self.graph = None
        self.checkpointer = SqliteSaver.from_conn_string(self.conn_string)

    def __enter__(self):
        """进入 with 语句时被调用。"""
        self.active_checkpointer = self.checkpointer.__enter__()
        self.graph = self.builder.compile(checkpointer=self.active_checkpointer)
        return self  # ! 返回实例本身，以便在 with 块中使用

    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出 with 语句时被调用，用于清理。"""
        self.checkpointer.__exit__(exc_type, exc_val, exc_tb)

    def _build_graph(self):
        """构建并编译 LangGraph。"""
        builder = StateGraph(AgentState)

        builder.add_node("planner", self.plan_node)
        builder.add_node("generate", self.generation_node)
        builder.add_node("reflect", self.reflection_node)
        builder.add_node("research_plan", self.research_plan_node)
        builder.add_node("research_critique", self.research_critique_node)

        builder.set_entry_point("planner")

        builder.add_conditional_edges(
            "generate",
            self.should_continue,
            {END: END, "reflect": "reflect"}
        )
        builder.add_edge("planner", "research_plan")
        builder.add_edge("research_plan", "generate")
        builder.add_edge("reflect", "research_critique")
        builder.add_edge("research_critique", "generate")

        return builder

    # --- 节点函数 ---

    def plan_node(self, state: AgentState):
        messages = [
            SystemMessage(content=PLAN_PROMPT),
            HumanMessage(content=state['task'])
        ]
        response = self.llm.invoke(messages)
        return {"plan": response.content}

    def research_plan_node(self, state: AgentState):
        queries = self.llm.with_structured_output(Queries).invoke([
            SystemMessage(content=RESEARCH_PLAN_PROMPT),
            HumanMessage(content=state['task'])
        ])
        content = state.get('content', [])
        for q in queries.queries:
            response = self.tavily_client.search(query=q, max_results=2)
            for r in response['results']:
                content.append(r['content'])
        return {"content": content}

    def generation_node(self, state: AgentState):
        content = "\n\n".join(state['content'] or [])
        user_message = HumanMessage(
            content=f"{state['task']}\n\nHere is my plan:\n\n{state['plan']}")
        messages = [
            SystemMessage(content=WRITER_PROMPT.format(content=content)),
            user_message
        ]
        response = self.llm.invoke(messages)
        return {
            "draft": response.content,
            "revision_number": state.get("revision_number", 0) + 1
        }

    def reflection_node(self, state: AgentState):
        messages = [
            SystemMessage(content=REFLECTION_PROMPT),
            HumanMessage(content=state['draft'])
        ]
        response = self.llm.invoke(messages)
        return {"critique": response.content}

    def research_critique_node(self, state: AgentState):
        queries = self.llm.with_structured_output(Queries).invoke([
            SystemMessage(content=RESEARCH_CRITIQUE_PROMPT),
            HumanMessage(content=state['critique'])
        ])
        content = state.get('content', [])  # Start with fresh content or append? Let's append.
        for q in queries.queries:
            response = self.tavily_client.search(query=q, max_results=2)
            for r in response['results']:
                content.append(r['content'])
        return {"content": content}

    def should_continue(self, state: AgentState):
        if state["revision_number"] > state["max_revisions"]:
            return END
        return "reflect"

    def run(self, task: str, max_revisions: int = 2):
        """
        运行论文写作流程。
        """
        if self.graph is None:
            raise RuntimeError("Graph is not compiled. Use this object within a 'with' statement.")
        thread = {"configurable": {"thread_id": "1"}}  # You can make this dynamic

        # Stream the output
        for s in self.graph.stream({
            'task': task,
            "max_revisions": max_revisions,
            "revision_number": 0,  # Start at 0
        }, thread):
            yield s