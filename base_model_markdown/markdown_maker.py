import os
import logging
import json
import requests
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from langchain_openai import ChatOpenAI
from logging.handlers import RotatingFileHandler  # 推荐使用这个，可以自动分割日志文件
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
# LangChain 相关导入
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- 初始化服务 ---
mcp = FastMCP("markdown_maker")
llm = None

# --- 1. Prompt 模板 (用于生成单个子文件) ---
# in markdown_maker.py

SUB_DOCUMENT_PROMPT_TEMPLATE = """
你是一位技艺精湛、风格多变的技术作家。你正在为一个大型文档项目撰写一个独立的章节。

首先，请牢记整个项目的核心主题思想，它将指导你所有的写作：
**项目主题 (Main Thesis): {main_thesis}**

---
### **当前章节写作任务**

现在，请专注于以下针对本章节的具体要求。

#### **1. 章节标题**
{sub_topic_title}

#### **2. 行文风格要求 (必须严格遵守)**
> {sub_topic_style}

#### **3. 核心内容要点 (请围绕这些要点展开，并确保内容与项目核心思想保持一致)**
{sub_topic_points_str}

#### **4. 章节结论 (对本章内容进行总结，并可选择性地与项目核心思想或下一章节建立联系)**
> {sub_topic_conclusion}
---

请完全沉浸在为 **{sub_topic_title}** 这一章节设定的角色和风格中，并时刻围绕 **{main_thesis}** 进行创作。
请直接输出完整的 Markdown 内容，不要包含任何额外的对话、解释或引言。
请使用中文进行撰写，并且尽量多的给出代码示例。
"""



# --- 2. 核心工具函数 ---
@mcp.tool()
def execute_writing_plan(
    main_thesis: str,
    writing_plan_json: str,
    output_directory: str,
    ) -> str:
    """
    根据一份详细的写作计划和整个项目的主题思想，分批次生成所有 Markdown 子文件。

    :param main_thesis: 整个文档项目的核心思想或总标题，用于为所有章节提供写作上下文。
    :param writing_plan_json: 由 subtitle_planner 生成的、包含所有子主题详细规划的 JSON 字符串。
    :param output_directory: 所有生成的 Markdown 文件将被保存到的目标目录。
    :return: 一个 JSON 字符串，报告操作状态和所有已创建文件的路径列表。
    """
    global llm
    if not llm:
        return json.dumps({"error": "LLM 服务未初始化。"})

    logging.info(f"开始执行写作计划，将在 '{output_directory}' 目录下生成文件...")

    try:
        # --- 步骤 1: 解析所有输入指南 ---
        writing_plan = json.loads(writing_plan_json)
        # --- 【这就是解决当前问题的核心代码】 ---
        plan = None
        if "plan" in writing_plan and isinstance(writing_plan.get("plan"), dict):
            # 情况 A: 接收到了正确的、嵌套的格式 (LLM 忠实传递)
            logging.info("接收到标准的、包含 'plan' 键的写作计划。")
            plan = writing_plan["plan"]
        elif isinstance(writing_plan, dict):
            # 情况 B: 接收到了扁平的格式 (LLM "智能"剥离了外壳)
            # 我们假设如果顶层没有 'plan'，那么整个对象就是 plan 本身
            logging.warning("接收到扁平格式的写作计划，已自动适配。这可能是由 LLM 在参数传递中进行了提取。")
            plan = writing_plan

        if not plan:
            # 情况 C: 接收到了无法识别的格式
            error_message = "执行失败：写作计划的 JSON 结构无法识别。"
            logging.error(f"{error_message} 接收到的数据: {writing_plan_json}")
            return json.dumps({"status": "error", "message": error_message})
        # --- 【容错逻辑结束】 ---

        # 准备通用的 Prompt 数据，现在包含了 main_thesis
        common_prompt_data = {
            "main_thesis": main_thesis,
        }



        # 创建 LangChain 调用链
        prompt_template = ChatPromptTemplate.from_template(SUB_DOCUMENT_PROMPT_TEMPLATE)
        chain = prompt_template | llm | StrOutputParser()

        created_files = []
        # --- 步骤 3: 循环执行写作计划中的每个子任务 ---
        # 对写作计划按键（文件名）进行排序，确保生成顺序可预测
        sorted_sub_topics = sorted(plan.keys())

        for sub_topic_key in sorted_sub_topics:
            task_details = plan[sub_topic_key]

            # 从文件名（如 '01_introduction'）生成一个更易读的标题（如 'Introduction'）
            # 这假设序号只是为了排序，标题中不需要
            title_parts = sub_topic_key.split('_')[1:]
            sub_topic_title = ' '.join(title_parts).title()

            logging.info(f"--- 开始生成子文档: {sub_topic_title} ---")

            # 准备当前子任务的 Prompt 数据
            # 先复制通用数据，再更新特定于本章节的数据
            sub_topic_prompt_data = common_prompt_data.copy()
            sub_topic_prompt_data.update({
                "sub_topic_title": sub_topic_title,
                "sub_topic_style": task_details.get("style", "请使用清晰、专业的技术写作风格。"),  # 提供一个安全的默认值
                "sub_topic_points_str": "\n".join([f"- {p}" for p in task_details.get("points", [])]),
                "sub_topic_conclusion": task_details.get("conclusion", f"对 {sub_topic_title} 的简要总结。")
            })

            # 调用 LLM 生成内容
            logging.info(f"正在为 '{sub_topic_title}' 调用 LLM...")
            markdown_content = chain.invoke(sub_topic_prompt_data)

            # 保存文件
            filename = f"{sub_topic_key}.md"
            file_path = os.path.join(output_directory, filename)

            # 确保输出目录存在
            os.makedirs(output_directory, exist_ok=True)

            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)

            file_path_abs = os.path.abspath(file_path)
            created_files.append(file_path_abs)
            logging.info(f"成功创建文件: {file_path_abs}")

        logging.info("所有写作任务已完成！")
        return json.dumps({"status": "success", "created_files": created_files})

    except json.JSONDecodeError as e:
        logging.error(f"解析输入的 'writing_plan_json' 时出错: {e}")
        return json.dumps({"status": "error", "message": f"输入的写作计划 JSON 格式不正确: {e}"})
    except Exception as e:
        logging.error(f"执行写作计划时出错: {e}", exc_info=True)
        return json.dumps({"status": "error", "message": f"执行写作计划时发生未知异常: {e}"})


def log_setup():
    # 1. 定义日志文件路径和名称
    # 推荐将日志放在一个专门的 logs 目录下
    log_directory = "logs"
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    # 你可以动态生成文件名，比如加上服务名
    log_file_path = os.path.join(log_directory, "markdown_maker.log")

    # 2. 配置日志记录器
    # 清除所有现有的 handlers，确保我们的配置是唯一的
    logging.getLogger().handlers = []

    logging.basicConfig(
        level=logging.INFO,  # 设置日志级别
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # 定义日志格式
        handlers=[
            # Handler 1: 将日志写入文件
            # 使用 RotatingFileHandler 可以防止日志文件无限增大
            # maxBytes: 单个文件最大大小 (这里是 5MB)
            # backupCount: 保留的旧日志文件数量
            RotatingFileHandler(log_file_path, maxBytes=5 * 1024 * 1024, backupCount=5, encoding='utf-8'),

            # Handler 2: 同时也在控制台打印一份日志，方便开发时查看
            logging.StreamHandler()
        ]
    )

#--- 4. 初始化和主函数 (与之前类似) ---
def initialize_services():
    global llm
    logging.info("正在初始化 Markdown Maker 服务...")
    load_dotenv(dotenv_path="/Users/dengjuan1/build_agent/.env")
    try:
        llm = ChatOpenAI(
            model="deepseek-chat",
            openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
            openai_api_base=os.getenv("DEEPSEEK_API_BASE"),
            temperature=0.0
        )
        logging.info("deepseek LLM 客户端初始化成功。")
    except Exception as e:
        logging.error(f"初始化deepseek LLM 失败: {e}")
        llm = None


def main():
    log_setup()
    initialize_services()
    logging.info("Markdown Maker MCP 服务已准备就绪...")
    mcp.run(transport='stdio')


if __name__ == "__main__":
    main()