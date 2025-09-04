import os
import asyncio
import sqlite3
# import sqlite_vss

from dotenv import load_dotenv

from langmem import create_manage_memory_tool, create_search_memory_tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langchain_community.embeddings import DashScopeEmbeddings

from .get_json import json_get_and_usage

from langchain_community.vectorstores import SQLiteVSS # 导入 SQLiteVSS
from langgraph.prebuilt import create_react_agent

from .stores import VectorStoreWrapperForLangMem

# 加载环境变量
load_dotenv(dotenv_path="/Users/dengjuan1/build_agent/.env")

# 定义 Agent 的 System Prompt
MEMORY_AGENT_SYSTEM_PROMPT = """你是一位专业的记忆管理助手。
你的核心职责是理解并存储用户提供的关键信息到长期记忆系统中，并在需要时从中检索信息。
<操作指南>
- 当接收到需要记忆的信息时，请使用 `manage_memory` 工具进行存储。
- 当用户提出问题时，请首先使用 `search_memory` 工具查询是否存在相关记忆。
- 根据搜索结果，提供一个全面、准确的回答。
- 在所有记忆操作中，请始终保持严谨和精确。
"""
#%%
# a. 初始化 Embedding 模型 (用于向量化)
embedding_model = DashScopeEmbeddings(model="text-embedding-v2")

# b. 初始化 sqlite
db_file = "semantic_memory.db"
table_name = "memory_collection"
# table_name = "debug_collection"

# Use from_texts to create a thread-safe vector store
underlying_vector_store = SQLiteVSS.from_texts(
    texts=[],
    embedding=embedding_model,
    db_file=db_file,
    table_name=table_name
)
print("✅ Underlying VectorStore (SQLiteVSS) is configured.")

# underlying_vector_store = SQLiteVSS.from_texts(
#     texts=[],
#     embedding=embedding_model, # 你的 embedding_model
#     db_file=db_file,
#     table_name=table_name
# )
# print(f"✅ Underlying VectorStore (SQLiteVSS) is configured.")

persistent_store = VectorStoreWrapperForLangMem(vector_store=underlying_vector_store)
print(f"✅ VectorStore wrapped in a BaseStore-compatible adapter.")


# d. 创建持久化的 Checkpointer (对话历史记忆)
# memory_checkpointer = AsyncSqliteSaver.from_conn_string(":memory:")

# e. 创建记忆工具
manage_memory_tool = create_manage_memory_tool(
    namespace=("edit_mem_assistant", "{langgraph_user_id}", "{collection_name}")
)
search_memory_tool = create_search_memory_tool(
    namespace=("edit_mem_assistant", "{langgraph_user_id}", "{collection_name}")
)
memory_tools = [manage_memory_tool, search_memory_tool]


# f. 定义 Prompt 构建函数
def build_prompt(state: dict):
    return [
        {"role": "system", "content": MEMORY_AGENT_SYSTEM_PROMPT}
    ] + state['messages']



#%%
# ==============================================================================
# 4. 主执行入口
# ==============================================================================

async def main():
    checkpointer_manager = AsyncSqliteSaver.from_conn_string(":memory:")

    # b. ✨ 使用 async with 来正确地进入上下文，获取可用的 checkpointer 对象
    async with checkpointer_manager as memory_checkpointer:

        MemoryAgent = create_react_agent(
            model="deepseek-chat",
            tools=memory_tools,
            prompt=build_prompt,
            checkpointer=memory_checkpointer,  # 👈 现在传入的是一个完全合法的 checkpointer
            store=persistent_store
        )
        print("✅ Memory Agent is ready.")

        # d. 【所有对 Agent 的调用也必须在 with 语句内部】
        # 因为 checkpointer 只在这个上下文中有效
        notebook_file_to_ingest = "/Users/dengjuan1/build_agent/Notebook4Revise/Tokenizer_chinese.json"
        user = "dengjuan"

        await json_get_and_usage(
            MemoryAgent=MemoryAgent,
            user_id=user,
            notebook_path=notebook_file_to_ingest
        )


if __name__ == "__main__":

    asyncio.run(main())

