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
# --- 1. 导入新的类名 ---
from mem0 import MemoryClient

load_dotenv(dotenv_path="/Users/dengjuan1/build_agent/.env")


class MCP_ChatBot:
    def __init__(self):

        self.llm = ChatOpenAI(
            model="deepseek-chat",
            openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
            openai_api_base=os.getenv("DEEPSEEK_API_BASE"),
            temperature=0.7
        )
        # --- 2.1 记忆管理器 mem0 对象 ---
        self.memory_client = MemoryClient(api_key=os.getenv("MEM0_API_KEY"))
        # --- 2.2 短期记忆对象 ---
        self.short_term_memory: Dict[str, ChatMessageHistory] = {}

    def get_short_term_history(self, session_id: str) -> ChatMessageHistory:
        """获取或创建当前会话的短期聊天历史。"""
        if session_id not in self.short_term_memory:
            self.short_term_memory[session_id] = ChatMessageHistory()
        return self.short_term_memory[session_id]

    # --- 3. 修改 process_query 以使用记忆 ---
    async def process_query(self, query: str, session_id: str = "default_session"):
        """
        处理查询，使用最新的 mem0ai API 进行记忆管理。
        """
        # --- 步骤 A: 获取短期记忆 ---
        short_term_history = self.get_short_term_history(session_id)
        # 为了 RAG，我们可以将短期历史格式化为字符串
        recent_conversation = "\n".join(
            [f"{msg.type}: {msg.content}" for msg in short_term_history.messages[-4:]]  # 只取最近4条消息
        )

        # --- 步骤 B: 从记忆中检索 ---
        print(f"Retrieving relevant context from mem0 for session '{session_id}'...")
        # 假设 search 是同步的，在异步函数中安全地调用它
        loop = asyncio.get_running_loop()
        relevant_memories = await loop.run_in_executor(
            None,
            lambda: self.memory_client.search(query=query, user_id=session_id)
        )
        print(relevant_memories)
        # 格式化检索到的记忆
        # mem0 的 search 返回一个字典列表，每个字典包含 'text', 'score', 'metadata' 等
        context = "\n".join(
            [mem['memory'] for mem in relevant_memories]) if relevant_memories else "No relevant memories found."
        print(f"Retrieved context:\n---\n{context}\n---")

        # --- 步骤 B: RAG - 构造最终的 Prompt ---
        final_prompt = f"""
        You are a helpful AI assistant. Use the following pieces of long-term memory and recent conversation to help answer the user's question. If the memory context or recent conversation is not relevant, ignore it.
    
        <Memory Context>
        {context}
        </Memory Context>
        
        <Recent Conversation>
        {recent_conversation}
        </Recent Conversation>
    
        User's current question: {query}
        """

        # 调用 LLM
        response = await self.llm.ainvoke(final_prompt)
        ai_response_content = response.content

        print(f"\nAI Response: {ai_response_content}")

        # --- 步骤 C: 将新信息存入记忆 ---
        print("Adding current conversation to mem0...")

        # 【核心修改】构造符合官方 API 格式的消息列表
        messages_to_add = [
            {"role": "user", "content": query},
            {"role": "assistant", "content": ai_response_content}
        ]

        # 调用 add 方法
        await loop.run_in_executor(
            None,
            lambda: self.memory_client.add(messages_to_add, user_id=session_id)
        )
        # --- 步骤 D: 更新短期记忆 ---
        short_term_history.add_user_message(query)
        short_term_history.add_ai_message(ai_response_content)

        print("Memory updated.")

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