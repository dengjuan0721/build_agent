
import os
from typing import TypedDict, List
from tavily import TavilyClient
from pydantic import BaseModel

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_deepseek import ChatDeepSeek

from .prompts_chinese import INITIAL_PARSE_PROMPT, REFLECTION_PARSE_PROMPT, REVISION_PARSE_PROMPT
from utils.schema import StructuredPlan  # ! 从 schemas.py 导入


class ParserState(TypedDict):
    # Inputs
    unstructured_plan: str

    # Dynamic State
    parsed_json: dict  # 将存储 Pydantic 模型的 dict 表示
    critique: str
    revision_number: int
    max_revisions: int


class AiParserGraph:
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
        builder = StateGraph(ParserState)

        # 定义三个核心节点
        builder.add_node("generate_initial", self.initial_parse_node)
        builder.add_node("reflect_on_parse", self.reflection_node)
        builder.add_node("revise_parse", self.revision_node)

        builder.set_entry_point("generate_initial")

        # 定义循环
        builder.add_edge("generate_initial", "reflect_on_parse")
        builder.add_edge("revise_parse", "reflect_on_parse")

        builder.add_conditional_edges(
            "reflect_on_parse",
            self.should_continue_parsing,
            {
                "revise": "revise_parse",  # 如果需要修正，就去 revise 节点
                END: END  # 如果完美，就结束
            }
        )
        return builder

    # --- 节点函数 ---

    def initial_parse_node(self, state: ParserState):
        parser_llm = self.llm.with_structured_output(StructuredPlan)
        messages = [SystemMessage(content=INITIAL_PARSE_PROMPT.format(unstructured_plan=state['unstructured_plan']))]
        response = parser_llm.invoke(messages)
        # print(f"Type of response: {type(response)}")
        # print(response)
        return {"parsed_json": response.model_dump(), "revision_number": 0}

    def reflection_node(self, state: ParserState):
        import json
        messages = [SystemMessage(content=REFLECTION_PARSE_PROMPT.format(
            unstructured_plan=state['unstructured_plan'],
            parsed_json=json.dumps(state['parsed_json'], indent=2)
        ))]
        response = self.llm.invoke(messages)
        return {"critique": response.content}

    def revision_node(self, state: ParserState):
        import json
        parser_llm = self.llm.with_structured_output(StructuredPlan)
        messages = [SystemMessage(content=REVISION_PARSE_PROMPT.format(
            unstructured_plan=state['unstructured_plan'],
            parsed_json=json.dumps(state['parsed_json'], indent=2),
            critique=state['critique']
        ))]
        response = parser_llm.invoke(messages)
        return {"parsed_json": response.model_dump(), "revision_number": state['revision_number'] + 1}

    def should_continue_parsing(self, state: ParserState):
        if state["revision_number"] >= state["max_revisions"]:
            return END
        if "PERFECTLY_PARSED" in state['critique']:
            return END
        return "revise"

    def run(self, unstructured_plan: str, max_revisions: int = 1):
        """
        运行论文写作流程。
        """
        if self.graph is None:
            raise RuntimeError("Graph is not compiled. Use this object within a 'with' statement.")
        thread = {"configurable": {"thread_id": "1"}}  # You can make this dynamic

        # Stream the output
        for s in self.graph.stream({
            'unstructured_plan': unstructured_plan,
            "max_revisions": max_revisions,
            "revision_number": 0,  # Start at 0
        }, thread):
            yield s