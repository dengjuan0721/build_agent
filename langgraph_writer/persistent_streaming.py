# libraries
from dotenv import load_dotenv
import os
from tavily import TavilyClient

# load environment variables from .env file
_ = load_dotenv(dotenv_path="/Users/dengjuan1/build_agent/.env")

from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI

# agentic search
from langchain_tavily import TavilySearch

tool = TavilySearch(max_results=4)

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]


# 在计算机的内存（RAM）中创建一个完整功能的、临时的 SQLite 数据库
from langgraph.checkpoint.sqlite import SqliteSaver

memory = SqliteSaver.from_conn_string(":memory:")

class Agent:
    def __init__(self, model, tools, checkpointer, system=""):
        print("--- init ---")
        self.system = system
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_openai)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges("llm", self.exists_action, {True: "action", False: END})
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        # 有状态执行
        self.graph = graph.compile(checkpointer=checkpointer)
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)

    def call_openai(self, state: AgentState):
        messages = state['messages']
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        message = self.model.invoke(messages)
        print("--- message update ---")
        print(message)
        return {'messages': [message]}

    def exists_action(self, state: AgentState):
        result = state['messages'][-1]
        return len(result.tool_calls) > 0

    def take_action(self, state: AgentState):
        tool_calls = state['messages'][-1].tool_calls
        results = []
        for t in tool_calls:
            print(f"Calling: {t}")
            result = self.tools[t['name']].invoke(t['args'])
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
        print("Back to the model!")
        return {'messages': results}

prompt = """You are a smart research assistant. Use the search engine to look up information. \
You are allowed to make multiple calls (either together or in sequence). \
Only look up information when you are sure of what you want. \
If you need to look up some information before asking a follow up question, you are allowed to do that!
"""

llm = ChatOpenAI(
    model="deepseek-chat",
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
    openai_api_base=os.getenv("DEEPSEEK_API_BASE"),
    temperature=0.0
)


with SqliteSaver.from_conn_string(":memory:") as checkpointer:
    abot = Agent(llm, [tool], system=prompt, checkpointer=checkpointer)
    query = "What is the weather in Zhenjiang ?"
    messages = [HumanMessage(content=query)]
    thread = {"configurable": {"thread_id": "1"}}
    for event in abot.graph.stream({"messages": messages}, thread):
        for v in event.values():
            print(v['messages'])

    print("\n\n\n================= STARTING SECOND QUESTION =================")
    for event in abot.graph.stream({"messages": messages}, thread):
        for v in event.values():
            print(v)


