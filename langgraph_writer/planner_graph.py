# 文件: planner_module/essay_builder.py
import os
from typing import TypedDict, List
from tavily import TavilyClient
from pydantic import BaseModel

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_deepseek import ChatDeepSeek

# ... (imports)
from .prompts import (
    INITIAL_PLAN_PROMPT, REFLECTION_AND_REFINEMENT_PROMPT, CRITIQUE_PLAN_PROMPT,
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


# --- 1. 修改 AgentState ---
class AgentState(TypedDict):
    task: str
    plan: str  # 将存储初始和最终的计划
    critique: str
    content: List[str]  # 用于存储研究结果
    revision_number: int
    max_revisions: int


class PlannerGraph:
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
        builder = StateGraph(AgentState)

        # --- 2. 节点保持大部分，但功能改变 ---
        builder.add_node("planner", self.initial_plan_node)
        builder.add_node("research_plan", self.research_plan_node)
        builder.add_node("reflect_on_plan", self.reflection_node)  # 重命名以示清晰
        builder.add_node("research_critique", self.research_critique_node)
        builder.add_node("refine_plan", self.refinement_node)  # 新的“最终计划”生成节点

        # --- 3. 重构图的流程 ---
        builder.set_entry_point("planner")

        builder.add_edge("planner", "research_plan")
        builder.add_edge("research_plan", "reflect_on_plan")  # 研究后直接进行反思
        builder.add_edge("reflect_on_plan", "research_critique")
        builder.add_edge("research_critique", "refine_plan")  # 研究完批判后，去优化计划

        builder.add_conditional_edges(
            "refine_plan",
            self.should_continue,
            {END: END, "reflect": "reflect_on_plan"}  # 如果需要，可以再次循环反思
        )

        return builder

    # --- 4. 修改节点函数 ---

    def initial_plan_node(self, state: AgentState):
        messages = [
            SystemMessage(content=INITIAL_PLAN_PROMPT.format(task=state['task'])),
            HumanMessage(content=state['task'])
        ]
        response = self.llm.invoke(messages)
        return {"plan": response.content, "revision_number": 0}

    # research_plan_node 保持不变
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

    def reflection_node(self, state: AgentState):  # 现在反思的是 Plan
        messages = [
            SystemMessage(content=CRITIQUE_PLAN_PROMPT.format(plan=state['plan'])),
            HumanMessage(content="Please critique this plan.")
        ]
        response = self.llm.invoke(messages)
        return {"critique": response.content}

    # research_critique_node 保持不变
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

    def refinement_node(self, state: AgentState):  # 替代了原来的 generation_node
        messages = [
            SystemMessage(content=REFLECTION_AND_REFINEMENT_PROMPT.format(
                plan=state['plan'],
                critique=state['critique']
            )),
            HumanMessage(content="Generate the final, improved plan based on the critique and research.")
        ]
        response = self.llm.invoke(messages)
        # 最终的计划会覆盖掉旧的 plan
        return {"plan": response.content, "revision_number": state.get("revision_number", 0) + 1}

    def should_continue(self, state: AgentState):
        # 我们可以让它只循环一次或两次来优化计划
        if state["revision_number"] >= state["max_revisions"]:
            return END
        return "reflect"  # 返回到反思节点

    def run(self, task: str, max_revisions: int = 1):  # 默认优化1次
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