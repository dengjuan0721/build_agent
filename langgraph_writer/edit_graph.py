import os
import json
from datetime import datetime
from typing import TypedDict, Optional, List, Dict
from langgraph.checkpoint.sqlite import SqliteSaver

from langgraph.graph import StateGraph, END

from langchain_deepseek import ChatDeepSeek

from utils.schema import ProposedEdit
from .prompts_chinese import PROPOSE_EDIT_PROMPT, SUMMARIZE_EDIT_HISTORY_PROMPT, SUMMARIZE_USER_INTENT_PROMPT

# --- 1. Define the State ---
class EditState(TypedDict):
    notebook_path: str
    file_to_edit_path: str
    notebook_mark: str
    original_content: str
    initial_user_prompt: str #用于记录最初始的命令
    proposed_edit: Optional[ProposedEdit] #修改建议
    user_feedback: str #
    revision_history: List[Dict[str, str]] #修改历史
    # revision_count: int


class EditGraph:
    def __init__(self, conn_string=":memory:"):
        self.llm = ChatDeepSeek(
            model="deepseek-chat",
            openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
            openai_api_base=os.getenv("DEEPSEEK_API_BASE"),
            temperature=0.0
        )
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
        builder = StateGraph(EditState)

        builder.add_node("propose_edit", self.propose_edit_node)
        builder.add_node("present_and_get_feedback", self.present_and_get_feedback_node)
        builder.add_node("update_history", self.update_history_node)
        builder.add_node("update_file", self.update_file_node)
        builder.add_node("update_memory", self.update_memory_node)

        builder.set_entry_point("propose_edit")
        builder.add_edge("propose_edit", "present_and_get_feedback")

        # ✨ 2. 流程改变：获取反馈后，先去更新历史，再做判断
        builder.add_edge("present_and_get_feedback", "update_history")

        builder.add_conditional_edges(
            "update_history",  # ✨ 3. 从新节点出发进行判断
            self.should_continue_editing,
            {
                "continue_editing": "propose_edit",
                "end_editing": "update_file",
                "quit": END  # ✨ 明确定义 quit 路径
            }
        )
        builder.add_edge("update_file", "update_memory")
        builder.add_edge("update_memory", END)

        return builder

    # --- 2. Define the Nodes ---
    def propose_edit_node(self, state: EditState) -> EditState:
        # 1. 智能格式化修正历史
        history = state.get('revision_history', [])
        revision_history_str = ""
        if history:
            formatted_history = []
            for i, turn in enumerate(history):
                formatted_history.append(
                    f"<第 {i + 1} 轮修订建议>\n{turn['proposal']}\n</第 {i + 1} 轮修订建议>\n"
                    f"<用户反馈>\n{turn['feedback']}\n</用户反馈>"
                )
            # 添加一个标题，让 LLM 更清楚这是历史记录
            revision_history_str = (
                    "--- 修订历史与反馈 ---\n"
                    + "\n\n".join(formatted_history)
                    + "\n--- 请根据以上历史和最新反馈，生成新一轮修订 ---"
            )

        # 2. 构建 Prompt
        prompt = PROPOSE_EDIT_PROMPT.format(
            original_content=state['original_content'],
            initial_user_prompt=state['initial_user_prompt'],
            revision_history_str=revision_history_str
        )

        structured_llm = self.llm.with_structured_output(ProposedEdit)
        proposal = structured_llm.invoke(prompt)

        return {
            "proposed_edit": proposal
        }

    def present_and_get_feedback_node(self, state: EditState) -> EditState:
        print("\n--- 📝 Proposed Edit ---")
        proposal = state['proposed_edit']
        print(f"AI Reasoning: {proposal.ai_reasoning}")
        print("\n--- ORIGINAL ---")
        print(state['original_content'])
        print("\n--- REVISED ---")
        print(proposal.revised_content)
        print("--------------------")

        feedback = input("Type 'accept' to apply the changes,'quit' to keep it the same or provide new instructions to revise again: ").strip()
        return {"user_feedback": feedback}

    def update_file_node(self, state: EditState) -> dict:
        """
        这个节点只负责将最终提案写入 .md 文件。
        它不返回任何状态更新。
        """
        print("\n--- 💾 Updating physical file... ---")
        proposal = state.get('proposed_edit')
        if not proposal:
            print("⚠️ WARNING: No proposed edit found. Skipping file update.")
            return {}

        try:
            with open(state['file_to_edit_path'], 'r', encoding='utf-8') as f:
                full_file_content = f.read()

            if state['original_content'] not in full_file_content:
                print(f"⚠️ WARNING: Original content not found in '{state['file_to_edit_path']}'. File not modified.")
            else:
                updated_content = full_file_content.replace(state['original_content'], proposal.revised_content)
                with open(state['file_to_edit_path'], 'w', encoding='utf-8') as f:
                    f.write(updated_content)
                print(f"Successfully updated file: {state['file_to_edit_path']}")

        except FileNotFoundError:
            print(f"❌ ERROR: File not found at '{state['file_to_edit_path']}'. Cannot apply changes.")
        except Exception as e:
            print(f"❌ ERROR: An unexpected error occurred while updating the file: {e}")

        return {}  # 这个节点不修改 state

    def update_memory_node(self, state: EditState) -> dict:
        """
        这个节点只负责总结历史并更新 Notebook.json (中期记忆)。
        它不返回任何状态更新。
        """
        print("\n--- 🧠 Updating medium-term memory (notebook)... ---")

        proposal = state.get('proposed_edit')
        if not proposal:
            print("⚠️ WARNING: No proposed edit found. Skipping memory update.")
            return {}
        # a. 意图总结
        print("--- 🧐 Summarizing initial user intent... ---")
        summarized_intent = ""
        try:
            intent_prompt = SUMMARIZE_USER_INTENT_PROMPT.format(raw_user_prompt=state['initial_user_prompt'])
            # 💡 可以使用辅助 LLM
            intent_response = self.llm.invoke(intent_prompt)
            summarized_intent = intent_response.content.strip()
        except Exception as e:
            print(f"⚠️ WARNING: Failed to summarize user intent, using raw prompt. Error: {e}")
            summarized_intent = state['initial_user_prompt']  # 回退到使用原始输入

        # b. 调用 LLM 生成高质量的 revision_summary
        history_summary = ""
        history = state.get('revision_history', [])

        if history:
            print("--- ✍️ Summarizing revision history... ---")
            revision_history_str = "\n".join(
                [f"AI提案:\n{turn.get('proposal', '')}\n用户反馈:\n{turn.get('feedback', '')}" for turn in history]
            )
            summary_prompt = SUMMARIZE_EDIT_HISTORY_PROMPT.format(
                initial_user_prompt=state['initial_user_prompt'],
                final_proposal_content=proposal.revised_content,
                revision_history_str=revision_history_str
            )
            try:
                summary_response = self.llm.invoke(summary_prompt)
                history_summary = summary_response.content
            except Exception as e:
                print(f"⚠️ WARNING: Failed to summarize history with LLM: {e}")
                history_summary = "Failed to generate summary. Raw history logged."
        else:
            history_summary = "User accepted the initial proposal without any revisions."

        # b. 构建 edit_event
        edit_event = {
            "timestamp": datetime.now().isoformat(),
            # ✨ 存入的是总结后的意图
            "initial_user_prompt": summarized_intent,
            # "final_content_preview": f"{proposal.revised_content[:40]}...",
            "ai_reasoning": proposal.ai_reasoning,
            "revision_summary": history_summary
        }

        # c. 写入 Notebook JSON
        try:
            with open(state['notebook_path'], 'r+', encoding='utf-8') as f:
                notebook_data = json.load(f)
                entry_found = False
                for entry in notebook_data:
                    current_entry_mark = f"{entry.get('type', '')}{entry.get('location_description', '')}"
                    if current_entry_mark == state['notebook_mark']:
                        if 'edit_history' not in entry or not isinstance(entry['edit_history'], list):
                            entry['edit_history'] = []
                        entry['edit_history'].append(edit_event)
                        entry_found = True
                        break
                if not entry_found:
                    print(
                        f"⚠️ WARNING: Could not find notebook entry with mark '{state['notebook_mark']}'. Memory not updated.")
                f.seek(0)
                f.truncate()
                json.dump(notebook_data, f, indent=4, ensure_ascii=False)
            print(f"Successfully updated notebook: {state['notebook_path']}")
        except Exception as e:
            print(f"❌ ERROR: An unexpected error occurred while updating the notebook: {e}")

        return {}  # 这个节点也不修改 state

    def update_history_node(self, state: EditState) -> dict:
        """
        这个节点唯一的职责就是更新 revision_history。
        它返回一个字典，LangGraph 会安全地将其合并回主状态。
        """
        history = state.get('revision_history', [])
        # 防御性编程，确保 history 始终是列表
        if not isinstance(history, list):
            history = []

        current_proposal = state.get('proposed_edit')
        if current_proposal:
            history.append({
                "proposal": current_proposal.revised_content,
                "feedback": state['user_feedback']
            })

        # ✨ 只返回需要更新的状态部分
        return {"revision_history": history}

    # --- 3. Define the Conditional Edge ---
    def should_continue_editing(self, state: EditState) -> str:
        """
        这个函数现在只负责判断，不修改任何状态。
        """
        feedback = state['user_feedback'].lower()

        if feedback == "accept":
            return "end_editing"

        if feedback == "quit":
            print("--- 🛑 User requested to quit. Ending session without saving. ---")
            return "quit"

        # 所有其他情况都视为继续
        return "continue_editing"

    def run(self, initial_state: dict):
        """
        可以循环的
        """
        if self.graph is None:
            raise RuntimeError("Graph is not compiled. Use this object within a 'with' statement.")

        thread = {"configurable": {"thread_id": "1"}}  # You can make this dynamic

        # Stream the output
        for s in self.graph.stream(initial_state, thread):
            yield s