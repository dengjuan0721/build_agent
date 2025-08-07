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

# --- 初始化服务 ---
mcp = FastMCP("content_logic_learner")
llm = None

# --- 1. 定义期望的 JSON 输出结构 (Pydantic 模型) ---
class ReasoningFramework(BaseModel):
    main_thesis: str = Field(description="用一句话总结这篇文章最核心、最想传达的观点或目的。")
    key_arguments: list[str] = Field(description="列出支撑核心观点的 3-5 个关键分论点或主要内容板块。")
    reasoning_flow: str = Field(description="描述整篇文章的逻辑流或论证顺序。例如：'从定义入手，然后分类讨论，最后通过案例总结' 或 '提出一个普遍问题，分析其成因，并给出具体的解决方案步骤'。")
    target_audience: str = Field(description="分析并描述这篇文章最可能的目标读者是谁，例如：'寻求快速入门的初学者'、'需要决策依据的技术管理者' 或 '进行学术研究的专家'。")

# --- 2. 创建新的 Prompt 模板 ---
REASONING_ANALYSIS_PROMPT = ChatPromptTemplate.from_template(
    """你是一位顶级的逻辑分析师和内容策略专家。你的任务是深入分析给定文本的内在逻辑和论证结构，而不是它的表面格式。
请忽略 Markdown 语法、代码块等格式细节，专注于理解作者的“思路”。

请根据你的分析，以 JSON 格式返回结果。

{format_instructions}

以下是待分析的文本内容：
---
{webpage_text}
---
"""
)

# --- 3. 工具函数 ---
@mcp.tool()
def learn_logic_from_url(url: str) -> str:
    """
    从给定的 URL 学习其行文规范和内容框架。

    :param url: 要分析的网页 URL。
    :return: 一个包含分析结果的 JSON 字符串。
    """
    global llm
    if not llm:
        return json.dumps({"error": "LLM 服务未初始化。"})

    logging.info(f"开始从 URL 学习内容逻辑: {url}")

    try:
        # 步骤 1: 使用 requests 获取网页 HTML
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=15)
        # response.raise_for_status()  # 如果请求失败则抛出异常
        #
        # # 步骤 2: 使用 BeautifulSoup 提取纯文本内容
        # soup = BeautifulSoup(response.text, 'html.parser')
        # # 尝试移除脚本和样式，只保留主要内容
        # for script_or_style in soup(['script', 'style', 'nav', 'footer', 'header']):
        #     script_or_style.decompose()
        # body_text = soup.body.get_text(separator='\n', strip=True)
        #
        # if len(body_text) > 15000:  # 限制送给 LLM 的文本长度，避免超出 token 限制
        #     body_text = body_text[:15000]
        # 直接获得结果不需要bs
        body_text = response.text

        # 1. 创建一个 JSON 解析器，并从 Pydantic 模型获取格式指令
        parser = JsonOutputParser(pydantic_object=ReasoningFramework)

        # 2. 将模板、LLM 和解析器链接成一个 Chain
        chain = REASONING_ANALYSIS_PROMPT | llm | parser

        # 3. 调用 (invoke) Chain，传入所需变量
        # LangChain 会自动将 format_instructions 注入到 prompt 中
        reasoning_data = chain.invoke({
            "webpage_text": body_text,
            "format_instructions": parser.get_format_instructions(),
        })

        # 4. 将解析后的 Python 字典转换回 JSON 字符串以便返回
        # parser 已经帮我们把结果解析成了字典
        framework_json = json.dumps(reasoning_data, ensure_ascii=False, indent=2)

        logging.info("内容逻辑学习成功！")
        return framework_json

    except Exception as e:
        # LangChain 可能会抛出 OutputParserException，如果 LLM 返回的不是有效 JSON
        logging.error(f"分析框架时出错: {e}", exc_info=True)  # exc_info=True 会记录详细的 traceback
        return json.dumps({"error": f"分析框架时发生异常: {e}"})


# --- 4. 初始化和主函数 (与之前类似) ---
def initialize_services():
    global llm
    logging.info("正在初始化 Logic Learner 服务...")
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
    log_file_path = os.path.join(log_directory, "content_logic_learner.log")

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
    logging.info("Content Logic Learner MCP 服务已准备就绪...")
    mcp.run(transport='stdio')


if __name__ == "__main__":
    main()