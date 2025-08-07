import os
import logging
import json
from dotenv import load_dotenv

# LangChain 相关导入
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from mcp.server.fastmcp import FastMCP
#init
from langchain_openai import ChatOpenAI
from logging.handlers import RotatingFileHandler  # 推荐使用这个，可以自动分割日志文件

# --- 初始化服务 ---
mcp = FastMCP("subtitle_planner")
llm = None

# --- 1. 定义 Pydantic 输出模型 ---
# 这个模型就是我们期望的“写作计划”的结构
class SubTopicPlan(BaseModel):
    points: list[str] = Field(description="该子主题下需要详细阐述的核心要点列表。")
    conclusion: str = Field(description="对该子主题内容的简短总结或承上启下的话。")
    style: str = Field(description="对该子主题语言风格的总结，体现与其他子主题的共性和区分。")


class WritingPlan(BaseModel):
    plan: dict[str, SubTopicPlan] = Field(
        description="一个包含完整写作计划的 JSON 对象。键是子主题名，应采用'序号_关键词'格式（例如 '01_introduction'）；值是包含 'points' 和 'conclusion' 的对象。"
    )


# --- 2. Prompt 模板 ---
# 这个 Prompt 的任务非常聚焦：做规划
SUBTITLE_PLANNING_PROMPT = ChatPromptTemplate.from_template(
    """你是一位顶级的课程设计师和内容架构师。
    你的任务是为一个复杂的大主题，规划出一系列逻辑清晰、循序渐进的子主题，并合理地将用户提供的原始要点分配到这些子主题中。
    
    ---
    ### 指南：宏观组织策略 (你必须模仿的范例)
    这是从一个优秀范例中学到的主题组织方法，请严格遵循此策略。
    
    - **主题分解策略**: {topic_decomposition_strategy}
    - **信息呈现顺序**: {information_flow_pattern}
    
    ---
    ### 本次规划任务
    - **总主题**: {main_topic_title}
    - **用户提供的参考要点 (需要检查是否包含了以下要点，没有包含需要进行补充)**:
    {user_points_str}
    
    ---
    请根据上述“宏观组织策略”的指导，对“本次规划任务”进行拆分。
    输出一个完整的写作计划 JSON 对象。
    
    {format_instructions}
    """
)


# --- 3. 核心工具函数 ---
@mcp.tool()
def plan_subtitles(
        main_topic_title: str,
        user_points: list[str],
        topic_framework_json: str,
) -> str:
    """
    为一个大主题规划子主题结构，并将用户要点分配到各子主题中。

    此工具是写作流程的“规划师”。它接收一个宏观主题、用户要点和一个从范例中学到的“主题组织框架”，
    然后输出一份详细的、结构化的写作计划（user_points_map），供下游的写作工具使用。

    :param main_topic_title: 需要规划的整个主题的标题，例如 "PyTorch 深度学习入门"。
    :param user_points: 一个包含用户希望在整个主题中涵盖的所有原始要点的字符串列表。
    :param topic_framework_json: 从 distill_final_topic_framework 学到的宏观主题组织框架的 JSON 字符串。
    :return: 一个 JSON 字符串，其中包含结构化的写作计划。
    """
    global llm
    if not llm:
        return json.dumps({"error": "LLM 服务未初始化。"})

    logging.info(f"开始为主题 '{main_topic_title}' 规划子主题...")

    try:
        # --- 准备 Prompt 数据 ---
        topic_data = json.loads(topic_framework_json)
        user_points_str = "\n".join([f"- {p}" for p in user_points])

        prompt_data = {
            "topic_decomposition_strategy": topic_data.get('topic_decomposition_strategy', '按逻辑顺序分解'),
            "information_flow_pattern": topic_data.get('information_flow_pattern', '从介绍开始，然后深入细节，最后总结'),
            "main_topic_title": main_topic_title,
            "user_points_str": user_points_str
        }

        # --- LangChain 调用流程 ---
        parser = JsonOutputParser(pydantic_object=WritingPlan)
        chain = SUBTITLE_PLANNING_PROMPT | llm | parser

        prompt_data["format_instructions"] = parser.get_format_instructions()

        logging.info("正在调用 LLM 进行子主题规划...")
        # LLM 的输出直接被 parser 解析为 Pydantic 对象
        writing_plan_dict = chain.invoke(prompt_data)

        # 【核心修复】直接将这个字典序列化为 JSON 字符串
        # 确保它包含我们期望的顶级 "plan" 键
        if "plan" not in writing_plan_dict:
            # 增加一个防御性检查，如果 LLM 没有按要求输出 'plan' 键
            error_msg = "LLM output did not contain the top-level 'plan' key."
            logging.error(f"{error_msg} Got: {writing_plan_dict}")
            raise ValueError(error_msg)

        writing_plan_json = json.dumps(
            writing_plan_dict,
            ensure_ascii=False,
            indent=2
        )

        logging.info("子主题规划成功！")
        logging.info(f"Generated Writing Plan JSON:\n{writing_plan_json}")

        return writing_plan_json

    except json.JSONDecodeError as e:
        logging.error(f"解析输入的框架 JSON 时出错: {e}")
        return json.dumps({"error": f"输入的框架 JSON 格式不正确: {e}"})
    except Exception as e:
        logging.error(f"规划子主题时出错: {e}", exc_info=True)
        return json.dumps({"error": f"规划时发生未知异常: {e}"})



# --- 4. 初始化和主函数 (与之前类似) ---
def initialize_services():
    global llm
    logging.info("正在初始化 Subtitle Planner 服务...")
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


def log_setup():
    # 1. 定义日志文件路径和名称
    # 推荐将日志放在一个专门的 logs 目录下
    log_directory = "logs"
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    # 你可以动态生成文件名，比如加上服务名
    log_file_path = os.path.join(log_directory, "subtitle_planner.log")

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


def main():
    log_setup()
    initialize_services()
    logging.info("Subtitle Planner MCP 服务已准备就绪...")
    mcp.run(transport='stdio')


if __name__ == "__main__":
    main()