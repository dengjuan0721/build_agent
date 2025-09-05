import os
from typing import TypedDict, List
from tavily import TavilyClient
from pydantic import BaseModel

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_deepseek import ChatDeepSeek
from .prompts_chinese import CONNECTOR_WRITER_PROMPT, CONNECTOR_REFLECTION_PROMPT, CONNECTOR_REVISION_PROMPT

# (还需要一个 REVISION_PROMPT，可以复用或定制)

class ConnectorWriterState(TypedDict):
    # Inputs
    writing_context: dict  # 包含 objective, tone, toc, section_data, sub_content
    max_revisions: int
    mode: str

    # Internal State
    draft: str
    critique: str
    revision_number: int


class ConnectorWriterGraph:
    def __init__(self, conn_string=":memory:"):
        self.llm = ChatDeepSeek(
            model="deepseek-reasoner",
            openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
            openai_api_base=os.getenv("DEEPSEEK_API_BASE"),
            temperature=0.0
        )
        self.tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
        self.conn_string = conn_string
        
        # 初始化记忆管理器
        try:
            from .memory_manager import WritingMemoryManager
            self.memory_manager = WritingMemoryManager()
            print("✅ ConnectorWriter: 长期记忆管理器已初始化")
        except Exception as e:
            print(f"⚠️ ConnectorWriter: 记忆管理器初始化失败: {e}")
            self.memory_manager = None
            
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
        builder = StateGraph(ConnectorWriterState)

        # 简化的三节点图
        builder.add_node("generate", self.generation_node)
        builder.add_node("reflect", self.reflection_node)
        builder.set_entry_point("generate")

        builder.add_edge("reflect", "generate")
        # should_continue 条件边现在从 reflect 出发
        builder.add_conditional_edges("generate", self.should_continue, {END: END, "reflect": "reflect"})

        return builder

    # --- 节点函数 ---

    def generation_node(self, state: ConnectorWriterState):
        """生成引言段落"""
        context = state['writing_context']
        critique = state.get('critique')

        # 1. 准备 System Prompt：定义 Agent 的角色
        #    这个角色在首次生成和修正时都是一样的
        system_prompt_content = "You are a Chinese master of prose and an expert Chinese editor, specializing in creating smooth narrative transitions."

        # 2. 根据是否存在 critique，选择不同的 Human Prompt 和格式化变量
        if critique and "PERFECTLY_WRITTEN" not in critique:
            # --- 修正阶段 ---
            print("-> Mode: Revising draft based on critique.")

            human_prompt_content = CONNECTOR_REVISION_PROMPT.format(
                objective=context.get('objective'),
                tone=context.get('tone'),
                section_title=context.get('section_data', {}).get('title'),
                sub_sections_content=context.get('sub_sections_content'),
                draft=state['draft'],
                critique=critique,
                mode=state['mode']
            )
        else:
            human_prompt_content = CONNECTOR_WRITER_PROMPT.format(
                objective=context.get('objective'),
                tone=context.get('tone'),
                table_of_contents=context.get('table_of_contents'),
                section_title=context.get('section_data', {}).get('title'),
                section_goal=context.get('section_data', {}).get('goal'),
                sub_sections_content=context.get('sub_sections_content'),
                mode=state['mode']
            )

        # 🧠 集成长期记忆：根据模式添加写作指导
        if self.memory_manager:
            try:
                # 根据模式确定section_type：intro/conclude
                section_type = "intro" if state['mode'] == 'intro' else "intro"  # conclude暂用intro的经验
                print(f"\n📝 [ConnectorWriter] 正在为{state['mode']}模式集成长期记忆...")
                
                # 获取该类型的写作指导
                memory_guidance = self.memory_manager.get_writing_guidance(section_type)
                
                if memory_guidance:
                    human_prompt_content = human_prompt_content + memory_guidance + "\n\n请参考以上写作原则进行内容创作。"
                    print(f"   ✅ 已将{section_type}类型写作指导集成到{state['mode']}模式prompt中")
                else:
                    print(f"   ⚠️ 未获取到写作指导，使用原始prompt")
                    
            except Exception as e:
                print(f"   ❌ 记忆集成失败: {e}")

        # 3. 构建消息列表并调用 LLM
        messages = [
            SystemMessage(content=system_prompt_content),
            HumanMessage(content=human_prompt_content)
        ]

        response = self.llm.invoke(messages)

        # 4. 返回更新后的状态
        return {
            "draft": response.content,
            "revision_number": state.get("revision_number", 0) + 1
        }

    def reflection_node(self, state: ConnectorWriterState):
        """反思引言段落"""
        # 1. 从 state 中解包出需要的上下文信息
        context = state['writing_context']

        # 2. 准备 System Prompt：定义 Agent 的角色和评审标准
        #    我们将 Prompt 的前半部分作为 SystemMessage
        system_prompt_content = CONNECTOR_REFLECTION_PROMPT.format(
            section_title=context.get('section_data', {}).get('title'),
            objective=context.get('objective'),
            sub_sections_content=context.get('sub_sections_content'),
            mode=state['mode']
        )
        # 移除 Draft to Review 和 Your Mission 部分，因为它们是具体任务
        system_prompt_content = system_prompt_content.split("**Draft to Review:**")[0].strip()

        # 3. 准备 HumanMessage：提供需要被评审的具体草稿
        human_prompt_content = f"Here is the draft you need to review and critique:\n\n---\n{state['draft']}\n---"

        # 🧠 集成长期记忆：添加详细的反思检查清单
        if self.memory_manager:
            try:
                # 根据模式确定section_type：intro/conclude
                section_type = "intro" if state['mode'] == 'intro' else "intro"  # conclude暂用intro的经验
                print(f"\n🔍 [ConnectorWriter] 正在为{state['mode']}模式集成反思检查清单...")
                
                # 获取该类型的反思检查清单
                reflection_checklist = self.memory_manager.get_reflection_checklist(section_type)
                
                if reflection_checklist:
                    human_prompt_content = human_prompt_content + reflection_checklist + "\n\n请仔细对照以上检查清单，逐项检验当前草稿是否符合历史修改建议。"
                    print(f"   ✅ 已将{section_type}类型检查清单集成到{state['mode']}模式reflection prompt中")
                else:
                    print(f"   ⚠️ 未获取到检查清单，使用原始prompt")
                    
            except Exception as e:
                print(f"   ❌ 检查清单集成失败: {e}")

        # 4. 构建消息列表并调用 LLM
        messages = [
            SystemMessage(content=system_prompt_content),
            HumanMessage(content=human_prompt_content)
        ]

        response = self.llm.invoke(messages)

        # 5. 返回批判意见
        return {"critique": response.content}

    def should_continue(self, state: ConnectorWriterState):
        if state["revision_number"] >= state["max_revisions"]:
            return END
        if "PERFECTLY_WRITTEN" in state.get('critique', ''):
            return END
        return "reflect"

    def run(self, inputs: dict, thread_config: dict):
        """
        运行论文写作流程。
        """
        if self.graph is None:
            raise RuntimeError("Graph is not compiled. Use this object within a 'with' statement.")
        for s in self.graph.stream({
            "writing_context": inputs.get("writing_context"),
            "max_revisions": inputs.get("max_revisions", 1), # 提供一个默认值

            # 初始化内部动态状态
            "draft": "",
            "critique": "",
            "revision_number": 0,
        }, thread_config):
            yield s
