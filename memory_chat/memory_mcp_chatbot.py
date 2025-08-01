# mcp_project/mcp_chatbot.py (Memory-Enabled Version)
import asyncio
from typing import Dict

# --- 1. 新增 Imports ---
from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import ConversationChain
from langchain_openai import ChatOpenAI  # 我们将用 LangChain 的 LLM 封装来集成记忆
from dotenv import load_dotenv
import os
# ... 其他 mcp, os, json, asyncio 等 imports
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory # 这是一个简单的内存中历史记录存储
from openai import AsyncOpenAI

load_dotenv(dotenv_path="/Users/dengjuan1/build_agent/.env")


class MCP_ChatBot:
    def __init__(self):

        # --- 1. 初始化 LangChain LLM 和会话记忆管理器 ---

        self.llm = ChatOpenAI(
            model="deepseek-chat",
            openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
            openai_api_base=os.getenv("DEEPSEEK_API_BASE"),
            temperature=0.7
        )
        # --- 2. 记忆管理器现在存储 ChatMessageHistory 对象 ---
        self.session_histories: Dict[str, ChatMessageHistory] = {}

        # --- 3. 在初始化时就定义好带记忆的链 (chain) ---
        # 这条链现在是一个可复用的模板
        prompt = ChatPromptTemplate.from_messages([
            ("system", "The following is a friendly conversation between a human and an AI."),
            MessagesPlaceholder(variable_name="history"),  # 告诉链在哪里插入历史消息
            ("user", "{input}"),
        ])

        # 基础的对话链
        base_chain = prompt | self.llm

        # 使用 RunnableWithMessageHistory 包装基础链，赋予它记忆能力
        self.chain_with_history = RunnableWithMessageHistory(
            base_chain,
            self.get_session_history,  # 这是一个函数，告诉链如何根据 session_id 获取历史记录
            input_messages_key="input",  # 告诉链用户的输入在哪
            history_messages_key="history",  # 告诉链历史记录应该插入到 prompt 的哪个位置
        )

    def get_session_history(self, session_id: str) -> ChatMessageHistory:
        """根据 session_id 获取或创建聊天历史记录对象。"""
        if session_id not in self.session_histories:
            self.session_histories[session_id] = ChatMessageHistory()
        return self.session_histories[session_id]

    # ... connect_to_server 和 connect_to_servers 方法保持不变 ...

    # --- 3. 修改 process_query 以使用记忆 ---
    async def process_query(self, query: str, session_id: str = "default_session"):
        """
        处理单个查询，现在会使用与 session_id 关联的记忆。
        """
        # 配置信息，包含了 session_id
        config = {"configurable": {"session_id": session_id}}

        # 调用链
        # .ainvoke 会自动处理：
        # 1. 使用 self.get_session_history(session_id) 加载历史
        # 2. 将历史和当前输入填充到 prompt 中
        # 3. 调用 llm
        # 4. 获取 AIMessage 响应
        # 5. 将用户的输入 (HumanMessage) 和 AI 的响应 (AIMessage) 保存回历史记录中
        response = await self.chain_with_history.ainvoke(
            {"input": query},
            config=config,
        )

        print(f"\nAI Response: {response.content}")

    # --- 4. 修改 chat_loop 以管理会话 ---
    async def chat_loop(self):
        """
        运行一个交互式聊天循环，现在支持会话。
        """
        print("\nMCP Chatbot (with Memory) Started!")
        session_id = input(
            "Enter a session ID (e.g., 'user123') or press Enter for default: ").strip() or "default_session"
        print(f"Starting chat session: {session_id}. Type 'quit' to exit.")

        while True:
            try:
                query = input(f"\n[{session_id}] Query: ").strip()
                if query.lower() == 'quit':
                    print("Ending session.")
                    break

                # 将 session_id 传递给 process_query
                await self.process_query(query, session_id)

            except Exception as e:
                print(f"\nError: {str(e)}")



async def main():
    chatbot = MCP_ChatBot()
    await chatbot.chat_loop()


# ... main 函数保持不变 ...
if __name__ == "__main__":
    asyncio.run(main())