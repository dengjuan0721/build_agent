
import os
from typing import TypedDict, List
from tavily import TavilyClient
from pydantic import BaseModel

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_deepseek import ChatDeepSeek
from tenacity import retry, stop_after_attempt, wait_exponential, RetryCallState

from .prompts_chinese import SECTION_WRITER_PROMPT, SECTION_REFLECTION_PROMPT, SECTION_RESEARCH_PROMPT, SECTION_PLAN_PROMPT, SECTION_RESEARCH_PLAN_PROMPT
import json


def log_before_retry(retry_state: RetryCallState):
    """
    一个回调函数，用于在 tenacity 触发重试前打印日志。
    """
    # retry_state.fn 是正在被重试的函数
    fn_name = retry_state.fn.__name__ if retry_state.fn else "unknown function"

    # retry_state.attempt_number 是当前的尝试次数
    current_attempt = retry_state.attempt_number

    # retry_state.seconds_since_start 是从第一次尝试开始经过的时间
    time_since_start = retry_state.seconds_since_start

    # retry_state.outcome.exception() 是导致这次失败的异常
    exception = retry_state.outcome.exception()

    print(
        f"\n⚠️ Retrying function '{fn_name}'! "
        f"Attempt #{current_attempt} failed after {time_since_start:.2f}s. "
        f"Reason: {exception}. "
        f"Waiting for {retry_state.next_action.sleep:.2f}s before next attempt."
    )



class SectionWriterState(TypedDict):
    # Inputs for the writer
    writing_context: dict

    # The dynamic state
    micro_plan: str  # plan_node 生成的微观计划
    draft: str
    critique: str
    content: List[str]
    revision_number: int
    max_revisions: int

class Queries(BaseModel):
    queries: List[str]

class SectionWriterGraph:
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

        builder = StateGraph(SectionWriterState)

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

    # --- 节点函数，使用新的 Prompts 和 State ---

    def plan_node(self, state: SectionWriterState):
        """生成微观计划"""
        context = state['writing_context']
        prompt = SECTION_PLAN_PROMPT.format(
            objective=context['objective'],
            tone=context['tone'],
            table_of_contents=context['table_of_contents'],
            section_to_write_json=json.dumps(context['section_to_write'], indent=2)
        )
        messages = [SystemMessage(content=prompt)]
        response = self.llm.invoke(messages)
        return {"micro_plan": response.content, "revision_number": 0}

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), before_sleep=log_before_retry)
    def research_plan_node(self, state: SectionWriterState):
        """为微观计划进行研究"""
        context = state['writing_context']
        prompt = SECTION_RESEARCH_PLAN_PROMPT.format(
            objective=context['objective'],
            micro_plan=state['micro_plan']
        )
        messages = [SystemMessage(content=prompt)]
        queries = self.llm.with_structured_output(Queries).invoke(messages)
        content = []  # 每次都开始新的、针对性的研究
        print(f"-> Researching queries: {[q for q in queries.queries]}")
        for q in queries.queries:
            response = self.tavily_client.search(query=q, max_results=2)
            for r in response['results']:
                content.append(r['content'])
        return {"content": content}

    def generation_node(self, state: SectionWriterState):
        """根据微观计划和研究结果撰写草稿"""
        context = state['writing_context']
        content_str = "\n\n".join(state['content'] or [])
        prompt = SECTION_WRITER_PROMPT.format(
            objective=context['objective'],
            tone=context['tone'],
            table_of_contents=context['table_of_contents'],
            micro_plan=state['micro_plan'],
            content=content_str
        )
        messages = [SystemMessage(content=prompt)
                    ]
        response = self.llm.invoke(messages)
        return {
            "draft": response.content,
            "revision_number": state.get("revision_number", 0) + 1
        }

    def reflection_node(self, state: SectionWriterState):
        """反思草稿"""
        context = state['writing_context']
        prompt = SECTION_REFLECTION_PROMPT.format(
            objective=context['objective'],
            tone=context['tone'],
            micro_plan=state['micro_plan'],
            draft=state['draft']
        )
        messages = [SystemMessage(content=prompt)]
        response = self.llm.invoke(messages)
        return {"critique": response.content}

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), before_sleep=log_before_retry)
    def research_critique_node(self, state: SectionWriterState):
        """为批判意见进行研究"""
        prompt = SECTION_RESEARCH_PROMPT.format(critique=state['critique'])
        messages = [SystemMessage(content=prompt)]
        queries = self.llm.with_structured_output(Queries).invoke(messages)
        content = state.get('content', [])  # Start with fresh content or append? Let's append.
        for q in queries.queries:
            response = self.tavily_client.search(query=q, max_results=2)
            for r in response['results']:
                content.append(r['content'])
        return {"content": content}

    def should_continue(self, state: SectionWriterState):
        if state["revision_number"] > state["max_revisions"]:
            return END
        return "reflect"

    def run(self, inputs: dict):
        """
        运行论文写作流程。
        """
        if self.graph is None:
            raise RuntimeError("Graph is not compiled. Use this object within a 'with' statement.")
        thread = {"configurable": {"thread_id": "1"}}  # You can make this dynamic

        # Stream the output
        for s in self.graph.stream({
            "writing_context": inputs.get("writing_context"),
            "max_revisions": inputs.get("max_revisions", 1),
            "revision_number": 0
        }, thread):
            yield s