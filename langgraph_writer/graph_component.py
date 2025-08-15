from dotenv import load_dotenv
import os
_ = load_dotenv(dotenv_path="/Users/dengjuan1/build_agent/.env")

# 核心包的导入
from langgraph.graph import StateGraph, END

from typing import TypedDict, Annotated
import operator
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI

# 搜索工具
# from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_tavily import TavilySearch

tool = TavilySearch(max_results=4) #increased number of results
print(type(tool))
print(tool.name)

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]

class Agent:

    def __init__(self, model, tools, system=""):
        self.system = system
        # 通过AgentState来创建一个流程图
        graph = StateGraph(AgentState)
        # 在图上加点（两个函数）
        graph.add_node("llm", self.call_openai)
        graph.add_node("action", self.take_action)
        # 加上边连接llm和action
        graph.add_conditional_edges(
            "llm",
            self.exists_action,
            {True: "action", False: END}
        )
        graph.add_edge("action", "llm")
        # 添加起始位置
        graph.set_entry_point("llm")
        self.graph = graph.compile()
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)

    def exists_action(self, state: AgentState):
        result = state['messages'][-1]
        return len(result.tool_calls) > 0

    def call_openai(self, state: AgentState):
        messages = state['messages']
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        message = self.model.invoke(messages)
        return {'messages': [message]}

    def take_action(self, state: AgentState):
        tool_calls = state['messages'][-1].tool_calls
        results = []
        for t in tool_calls:
            print(f"Calling: {t}")
            if not t['name'] in self.tools:      # check for bad tool name from LLM
                print("\n ....bad tool name....")
                result = "bad tool name, retry"  # instruct LLM to retry if bad
            else:
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

abot = Agent(llm, [tool], system=prompt)

with open("agent_graph.png", "wb") as f:
    f.write(abot.graph.get_graph().draw_png()) #graph.get_graph().draw_png()

messages = [HumanMessage(content="What is the weather in sf?")]
#会接收提供的初始消息，并将其作为“状态”在预先定义好的图中（llm -> action -> llm ...）流动，
# 直到图的流程到达终点 (END)，最后返回整个过程的最终状态。
result = abot.graph.invoke({"messages": messages})
print(result)