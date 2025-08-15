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

tool = TavilySearch(max_results=2)

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]


# 在计算机的内存（RAM）中创建一个完整功能的、临时的 SQLite 数据库
from langgraph.checkpoint.sqlite import SqliteSaver

memory = SqliteSaver.from_conn_string(":memory:")

from uuid import uuid4
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage

"""
In previous examples we've annotated the `messages` state key
with the default `operator.add` or `+` reducer, which always
appends new messages to the end of the existing messages array.

Now, to support replacing existing messages, we annotate the
`messages` key with a customer reducer function, which replaces
messages with the same `id`, and appends them otherwise.
"""
def reduce_messages(left: list[AnyMessage], right: list[AnyMessage]) -> list[AnyMessage]:
    # assign ids to messages that don't have them
    for message in right:
        if not message.id:
            message.id = str(uuid4())
    # merge the new messages with the existing messages
    merged = left.copy()
    for message in right:
        for i, existing in enumerate(merged):
            # replace any existing messages with the same id
            if existing.id == message.id:
                merged[i] = message
                break
        else:
            # append any new messages to the end
            merged.append(message)
    return merged

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], reduce_messages]

class Agent:
    def __init__(self, model, tools, system="", checkpointer=None):
        self.system = system
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_openai)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges("llm", self.exists_action, {True: "action", False: END})
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        self.graph = graph.compile(
            checkpointer=checkpointer,
            # “请在即将进入（before） 名为 action 的节点之前，暂停（interrupt） 整个图的执行。”
            interrupt_before=["action"]
        )
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)

    def call_openai(self, state: AgentState):
        messages = state['messages']
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        message = self.model.invoke(messages)
        return {'messages': [message]}

    def exists_action(self, state: AgentState):
        print(state)
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
            print(v)

    print("------")

    # 获取并打印状态
    current_state = abot.graph.get_state(thread)
    print("Full state:", current_state)  # 打印完整状态 print("Full state:", graph.get_state(thread))
    # 或者只打印消息
    # print("Messages:", current_state.values['messages'])

    print("------")

    # 获取并打印下一步的节点
    # .next 属性直接可用，不需要再次调用 get_state
    next_node = current_state.next
    print(f"Next node to execute is: {next_node}")

    for event in abot.graph.stream(None, thread):
        for v in event.values():
            print(v)
    abot.graph.get_state(thread)
    abot.graph.get_state(thread).next

    messages = [HumanMessage("Whats the weather in LA?")]
    thread = {"configurable": {"thread_id": "2"}}
    for event in abot.graph.stream({"messages": messages}, thread):
        for v in event.values():
            print(v)
    while abot.graph.get_state(thread).next:
        print("\n", abot.graph.get_state(thread), "\n")
        _input = input("proceed?")
        if _input != "y":
            print("aborting")
            break
        for event in abot.graph.stream(None, thread):
            for v in event.values():
                print(v)

