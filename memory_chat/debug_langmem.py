import os
import asyncio
from dotenv import load_dotenv
from typing import TypedDict, Annotated

# --- Imports ---
import redis
from langgraph_redis.checkpoint import RedisSaver  # 使用正确的导入
from langmem import create_manage_memory_tool
from langchain_community.embeddings import DashScopeEmbeddings
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages


# --- State Definition for our simple graph ---
class DebugState(TypedDict):
    messages: Annotated[list, add_messages]
    tool_output: str


# ==============================================================================
# Main Debug Logic
# ==============================================================================
async def main():
    print("--- 🔬 Starting In-Graph Tool Invocation Test ---")
    load_dotenv()

    # --- 1. 基础设施设置 (与你的 Agent 完全一致) ---
    embedding_model = DashScopeEmbeddings(model="text-embedding-v2")
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    redis_client = redis.Redis.from_url(redis_url)

    # a. 创建 RedisSaver 实例
    try:
        memory_db = RedisSaver.from_conn(
            conn=redis_client,
            index_schema={"embed": embedding_model}
        )
        print("✅ RedisSaver initialized.")
    except Exception as e:
        print(f"❌ FAILED to initialize RedisSaver: {e}")
        return

    # b. 创建 langmem 工具
    manage_memory_tool = create_manage_memory_tool(
        namespace=("edit_mem_assistant", "{langgraph_user_id}", "{collection_name}")
    )
    tools = [manage_memory_tool]

    # --- 2. 创建一个极简的 LangGraph 图 ---

    # a. 定义图的唯一节点
    def tool_caller_node(state: DebugState, config: dict):
        """This node's only job is to call the tool and capture the result."""
        print("\n--- 📞 Attempting to call manage_memory_tool inside the graph ---")
        try:
            # 模拟 Agent 决定调用工具
            tool_input = {
                "content": "This is a direct test from within a LangGraph environment.",
                "action": "create"
            }

            # 直接调用工具的 invoke 方法
            # 我们需要把 store 传入 config，因为 langmem 会从中寻找
            # 这是一个关键的假设，我们需要验证
            config["configurable"]["store"] = memory_db

            result = manage_memory_tool.invoke(tool_input, config=config)

            print(f"  - ✅ Tool call seems to have succeeded. Result: {result}")
            return {"tool_output": str(result)}

        except Exception as e:
            import traceback
            error_message = f"Tool call FAILED with an exception: {traceback.format_exc()}"
            print(f"  - ❌ {error_message}")
            return {"tool_output": error_message}

    # b. 构建图
    builder = StateGraph(DebugState)
    builder.add_node("tool_caller", tool_caller_node)
    builder.set_entry_point("tool_caller")

    # c. 编译图，并【关键】在这里传入 checkpointer 和 store
    # 我们使用同一个 memory_db 对象
    graph = builder.compile(checkpointer=memory_db, store=memory_db)
    print("✅ Minimal graph compiled.")

    # --- 3. 调用图并观察结果 ---
    thread = {"configurable": {
        "thread_id": "debug_thread_1",
        "langgraph_user_id": "debug_user",
        "collection_name": "debug_collection"
    }}

    initial_state = {"messages": []}

    print("\n--- ▶️ Invoking the graph... ---")
    final_state = await graph.ainvoke(initial_state, config=thread)

    print("\n--- 🏁 Graph execution finished. ---")
    print("Final State Tool Output:")
    print(final_state.get("tool_output", "No output captured."))
    print("\nPlease check your Redis database (e.g., using RedisInsight) or the console for errors.")


if __name__ == "__main__":
    asyncio.run(main())