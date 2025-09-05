import os
import json
from dotenv import load_dotenv

# --- LangChain & LangGraph & LangMem Imports ---
# (假设你的记忆 Agent 定义在 memory_agent.py 中)
# from memory_agent import response_agent, config # 如果你需要调用 agent

# 为了直接与 store 交互，我们需要这些
# from langgraph.store.lancedb import LanceDBStore
# import lancedb
from langmem import async_manage_memory  # langmem 提供了直接的函数调用

# 导入阿里 Embedding 模型
from langchain_community.embeddings import DashScopeEmbeddings

# 加载环境变量 (确保 DEEPSEEK_API_KEY 和 DASHSCOPE_API_KEY 已设置)
load_dotenv()


async def ingest_notebook_to_long_term_memory(notebook_path: str, user_id: str):
    """
    读取 Notebook.json 文件，提取所有 edit_history 中的 ai_reasoning,
    并根据 type 字段将它们写入到不同的长期记忆集合中。

    Args:
        notebook_path (str): Notebook JSON 文件的路径。
        user_id (str): 用户的唯一ID，用于构建 namespace。
    """
    print(f"--- 🚀 Starting ingestion process for notebook: {notebook_path} ---")

    # --- 1. 初始化持久化存储和 Embedding ---
    # 这部分代码需要与你的记忆 Agent 使用的 store 配置完全一致
    db_path = "./langmem_db"  # 确保路径与你的 Agent 脚本一致

    # a. 初始化阿里 Embedding 模型
    # (确保你的 DASHSCOPE_API_KEY 在环境变量中)
    embedding_model = DashScopeEmbeddings(model="text-embedding-v2")

    # b. 连接到 LanceDB 数据库 (已注释，不再使用)
    # db_connection = lancedb.connect(db_path)

    # c. 初始化 Store，并传入自定义的 Embedding Function (已注释)
    # store = LanceDBStore(
        db_connection,
        # namespace_template 在这里只是为了结构完整，实际操作时我们会手动构建
        namespace_template=("placeholder", "{langgraph_user_id}", "placeholder"),
        embedder=embedding_model  # ✨ 关键：指定使用阿里的 Embedding
    )

    # --- 2. 读取并解析 Notebook 文件 ---
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook_data = json.load(f)
    except FileNotFoundError:
        print(f"❌ ERROR: Notebook file not found at '{notebook_path}'")
        return
    except json.JSONDecodeError:
        print(f"❌ ERROR: Failed to decode JSON from '{notebook_path}'")
        return

    # --- 3. 遍历 Notebook，提取并写入记忆 ---
    total_memories_ingested = 0
    for entry in notebook_data:
        entry_type = entry.get('type')
        edit_history = entry.get('edit_history', [])

        if not entry_type or not edit_history:
            continue

        # a. 根据你的要求，构建动态的 namespace
        namespace = (
            "edit_mem_assistant",
            user_id,
            entry_type  # 动态部分：intro, leaf, 或 conclusion
        )

        print(f"\nProcessing section: {entry.get('location_description', 'N/A')}")
        print(f"Target namespace: {namespace}")

        # b. 遍历该条目的所有编辑历史
        for i, edit_event in enumerate(edit_history):
            ai_reasoning = edit_event.get('ai_reasoning')

            if not ai_reasoning:
                print(f"  - Skipping edit event #{i + 1}: No 'ai_reasoning' found.")
                continue

            # c. 调用 langmem 的异步函数写入记忆
            # 我们将 ai_reasoning 作为要记忆的内容
            try:
                await async_manage_memory(
                    ai_reasoning,
                    store=store,
                    namespace=namespace,
                    # 可以添加元数据，增强未来检索能力
                    metadata={
                        "source_notebook": os.path.basename(notebook_path),
                        "location": entry.get('location_description', 'N/A'),
                        "timestamp": edit_event.get('timestamp', 'N/A')
                    }
                )
                print(f"  - ✅ Successfully ingested memory #{i + 1} into collection '{entry_type}'.")
                total_memories_ingested += 1
            except Exception as e:
                print(f"  - ❌ FAILED to ingest memory #{i + 1}. Error: {e}")

    print(f"\n--- ✨ Ingestion process complete. Total memories ingested: {total_memories_ingested} ---")


