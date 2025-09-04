import json
import os
async def json_get_and_usage(MemoryAgent, user_id, notebook_path):
    """
    一个更真实的示例：
    1. 读取一个 Notebook.json 文件。
    2. 遍历其中所有的 edit_history。
    3. 提取 ai_reasoning 字段。
    4. 调用 MemoryAgent 将这些 "编辑技巧" 存入长期记忆。
    """
    # print("\n--- Running Realistic Ingestion Example from JSON file ---")

    # # --- a. 定义常量 ---
    # user_id = "dengjuan"
    # notebook_path = "/Users/dengjuan1/build_agent/Notebook4Revise/Tokenizer_chinese.json"

    # --- b. 读取并解析 JSON 文件 ---
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook_data = json.load(f)
        print(f"Successfully loaded notebook: {os.path.basename(notebook_path)}")
    except Exception as e:
        print(f"❌ ERROR: Could not load or parse the notebook file. {e}")
        return

    # --- c. 遍历并注入记忆 ---
    # 为这次注入任务创建一个唯一的会话 ID
    ingestion_thread_id = f"ingest_{os.path.basename(notebook_path).replace('.json', '')}_{user_id}"

    total_memories_found = 0
    for entry in notebook_data:
        collection = entry.get('type')
        edit_history = entry.get('edit_history', [])

        if not collection or not edit_history:
            continue

        # 对于每个条目内的编辑历史，我们都注入记忆
        for edit_event in edit_history:
            ai_reasoning = edit_event.get('ai_reasoning')

            if not ai_reasoning:
                continue

            total_memories_found += 1
            print(f"\nFound memory #{total_memories_found} in collection '{collection}': '{ai_reasoning[:60]}...'")

            # i. 构建本次调用的 config
            config = {
                "configurable": {
                    "langgraph_user_id": user_id,
                    "thread_id": ingestion_thread_id,
                    "collection_name": collection
                }
            }

            # ii. 准备要写入的记忆内容，包装成一个清晰的指令
            memory_to_add = f"请记忆此次修改的说理性文字: {ai_reasoning}"

            # iii. 异步调用 Agent 执行写入操作
            try:
                # Agent 会思考，然后决定调用 manage_memory_tool
                print("  - Streaming agent steps...")
                # ✨ 使用 astream 来观察每一步
                async for chunk in MemoryAgent.astream(
                        {"messages": [{"role": "user", "content": memory_to_add}]},
                        config=config
                ):
                    # 打印出每个节点的名字和它的输出
                    for key, value in chunk.items():
                        print(f"    Node: '{key}'")
                        # 我们特别关心 agent 的输出，因为它包含了工具调用
                        if key == "agent":
                            print(f"      - Agent Action: {value}")

                print("  - ✅ Agent stream finished.")
                # print("  - ✅ Successfully sent to Memory Agent for ingestion.")
            except Exception as e:
                print(f"  - ❌ FAILED to send memory to agent. Error: {e}")

    if total_memories_found == 0:
        print("\nNo 'ai_reasoning' entries found in the provided notebook file.")
    else:
        print(f"\n--- ✨ Ingestion task complete. Sent {total_memories_found} memories to the agent. ---")