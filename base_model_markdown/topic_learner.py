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
from langchain_core.output_parsers import StrOutputParser
# --- 导入记忆agent ---
from mem0 import MemoryClient
import asyncio

# --- 初始化服务 ---
mcp = FastMCP("content_logic_learner")
llm = None

# --- 1. 为新功能定义 Pydantic 输出模型 ---
class TopicFramework(BaseModel):
    topic_decomposition_strategy: str = Field(
        description="分析并描述作者是如何将一个大主题拆分成多个子页面的。例如：'按概念难度递进，从基础到高级' 或 '按功能模块进行分类，每个页面讲解一个独立功能'。")
    information_flow_pattern: str = Field(
        description="描述这些子页面内容的组织顺序和逻辑。例如：'通常以一个概述(Overview)页面开始，接着介绍核心API和概念，然后提供详细的教程和示例，最后是最佳实践总结'。")
    cross_page_common_structure: list[str] = Field(
        description="列出在多个子页面中反复出现的通用内容结构或章节。例如：['每个页面都包含“基本用法”代码示例', '多数页面末尾都有“相关资源”链接', '关键概念会用引用块进行强调']。")
    # suggested_master_toc: list[str] = Field(
    #     description="基于对所有内容的理解，提供一个推荐的、综合性的目录结构(Table of Contents)，用于组织一篇关于该主题的“终极指南”。")


# --- 2. 为新功能设计 Prompt 模板 ---
TOPIC_ANALYSIS_PROMPT = ChatPromptTemplate.from_template(
    """你是一位顶级的知识架构师和课程设计师。你的任务是分析一个由多个独立页面组成的完整主题，并洞察其背后宏观的组织策略和知识结构。
你收到的文本内容由一个文档片段拼接而成，该片段以 "--- DOCUMENT SEPARATOR (URL: [url]) ---" 开头。

请你综合分析所有文档片段，忽略微观的句子和格式，专注于回答以下宏观问题，并以 JSON 格式返回。

{format_instructions}


"""
)

# --- Prompt for SINGLE part analysis ---
SINGLE_PART_ANALYSIS_PROMPT = ChatPromptTemplate.from_template(
    """你是一位知识架构师，正在分析一个大型主题的其中一个部分。
    请分析以下文档片段，并简要总结它在整个主题中可能扮演的角色、核心内容和结构特点。
    你的分析将作为后续综合分析的素材。
    
    文档来源 URL: {url}
    ---
    文档内容:
    {content}
    ---
    请用几句话简要总结此部分的核心贡献和结构模式：
    """
)

# --- Prompt for FINAL distillation ---
FINAL_DISTILLATION_PROMPT = ChatPromptTemplate.from_template(
    """你是一位顶级的知识架构师，你的任务是综合、提炼和升华一系列关于某个大主题的初步分析，形成一个统一的、宏观的组织策略。
    你收到的“过往分析摘要”是之前对该主题下各个独立页面的初步观察结果。
    
    请你仔细阅读所有的摘要，并从中“蒸馏”出整个主题的宏观组织框架。
    回答以下问题，并以 JSON 格式返回最终结论。
    
    {format_instructions}
    
    ---
    过往分析摘要 (来自所有相关页面):
    {context}
    ---
    """
)
# --- 3. 实现 learn_topic_framework_from_url 函数 ---

# 我们可以先创建一个辅助函数来抓取单个 URL 内容，避免代码重复
def _fetch_and_clean_url_text(url: str) -> str:
    """辅助函数：抓取并清理单个 URL 的文本内容。"""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 ...'}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        return response.text


    except Exception as e:
        logging.warning(f"抓取或清理 URL {url} 时失败: {e}")
        return ""  # 返回空字符串，而不是让整个过程失败


@mcp.tool()
async def learn_and_memorize_single_topic_part(url: str, session_id: str = "anthropic_prompt_engineering") -> str:
    """
    分析单个 URL 的内容，生成初步的结构分析，并将其存入记忆。
    这是宏观框架学习流程的第一步，为最终的综合分析提供素材。

    :param url: 要分析的单个子主题页面的 URL。
    :param session_id: 当前学习任务的唯一会话 ID。
    :return: 一个确认信息，表示该部分已学习并存入记忆。
    """
    global llm, memory_client  # 假设 memory_client 也已初始化
    if not llm:
        return json.dumps({"error": "LLM 服务未初始化。"})

    logging.info(f"[{session_id}] 正在学习部分: {url}")

    # 步骤 1: 抓取和清理内容
    content = _fetch_and_clean_url_text(url)  # 你的辅助函数
    if not content:
        return json.dumps({"status": "skipped", "reason": "无法获取内容"})

    # 步骤 2: 调用 LLM 生成初步分析
    chain = SINGLE_PART_ANALYSIS_PROMPT | llm | StrOutputParser()
    preliminary_analysis = await chain.ainvoke({"url": url, "content": content})

    # 步骤 3: 将这个初步分析存入记忆
    logging.info(f"[{session_id}] 正在记忆关于 {url} 的分析...")
    # memory_text = f"关于URL {url} 的分析摘要: {preliminary_analysis}"


    # 【核心修复】将字符串包装成符合 Mem0 API 规范的消息列表
    memory_to_add = [
        {
            "role": "assistant",  # 或者 "system"，取决于你如何组织记忆
            "content": f"关于URL {url} 的分析摘要: {preliminary_analysis}"
        }
    ]
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(
        None,  # 使用默认的线程池执行器
        lambda: memory_client.add(messages=memory_to_add, user_id=session_id)
    )

    logging.info(f"[{session_id}] 成功记忆部分: {url}")
    return json.dumps({"status": "success", "url": url, "analysis_memorized": preliminary_analysis})


@mcp.tool()
async def distill_final_topic_framework(query: str = "Anthropic风格的 Prompt Engineering 文档的组织结构", session_id: str = "anthropic_prompt_engineering") -> str:
    """
    综合记忆中关于一个主题的所有初步分析，“蒸馏”出最终的、统一的宏观组织框架。
    这是宏观框架学习流程的最后一步。

    :param session_id: 当前学习任务的唯一会话 ID。
    :param query: 一个描述整个主题的查询，用于从记忆中检索所有相关分析。例如："关于 Anthropic Prompt Engineering 文档的组织结构"。
    :return: 一个包含最终主题级框架分析结果的 JSON 字符串。
    """
    global llm, memory_client
    if not llm:
        return json.dumps({"error": "LLM 服务未初始化。"})

    logging.info(f"[{session_id}] 开始最终蒸馏过程...")

    # 步骤 1: 从记忆中检索所有相关的初步分析
    # 【核心修复】将同步的 search 方法在子线程中运行
    loop = asyncio.get_running_loop()
    relevant_memories = await loop.run_in_executor(
        None,
        lambda: memory_client.search(query=query, user_id=session_id, limit=10)  # 加上 limit 是个好习惯
    )

    if not relevant_memories:
        return json.dumps({"error": "记忆中没有找到相关信息以进行蒸馏。"})

    # 格式化所有记忆，形成上下文
    context = "\n\n".join([mem['memory'] for mem in relevant_memories])
    logging.info(f"[{session_id}] 检索到用于蒸馏的上下文:\n{context}")

    # 步骤 2: 调用 LLM 进行最终的综合分析
    parser = JsonOutputParser(pydantic_object=TopicFramework)
    chain = FINAL_DISTILLATION_PROMPT | llm | parser

    final_framework_data = await chain.ainvoke({
        "context": context,
        "format_instructions": parser.get_format_instructions()
    })

    final_framework_json = json.dumps(final_framework_data, ensure_ascii=False, indent=2)
    logging.info(f"[{session_id}] 最终框架蒸馏成功！")
    return final_framework_json

# @mcp.tool()
# async def learn_topic_framework_from_url(url: str, session_id: str, query: str) -> str:
#     """
#     从一个代表完整主题的 URL 列表中，学习其宏观的知识组织框架和内容呈现策略。
#
#     :param urls: 一个包含构成主题的所有子页面 URL 的列表。
#     :return: 一个包含主题级框架分析结果的 JSON 字符串。
#     """
#     global llm
#     if not llm:
#         return json.dumps({"error": "LLM 服务未初始化。"})
#
#     logging.info(f"开始从 {url} 学习主题级框架...")
#
#     # --- 步骤 B: 从记忆中检索 ---
#     print(f"Retrieving relevant context from mem0 for session '{session_id}'...")
#     # 假设 search 是同步的，在异步函数中安全地调用它
#     loop = asyncio.get_running_loop()
#     relevant_memories = await loop.run_in_executor(
#         None,
#         lambda: memory_client.search(query=query, user_id=session_id)
#     )
#     logging.info(relevant_memories)
#     # 格式化检索到的记忆
#     # mem0 的 search 返回一个字典列表，每个字典包含 'text', 'score', 'metadata' 等
#     context = "\n".join(
#         [mem['memory'] for mem in relevant_memories]) if relevant_memories else "No relevant memories found."
#     logging.info(f"Retrieved context:\n---\n{context}\n---")
#
#     # --- 步骤 1: 内容聚合 ---
#     aggregated_content_parts = ""
#     logging.info(f"正在处理 URL: {url}")
#     content = _fetch_and_clean_url_text(url)
#     if content:
#         # 加入分隔符和元信息
#         header = f"\n\n--- DOCUMENT SEPARATOR (URL: {url}) ---\n\n"
#         aggregated_content_parts = header+content
#
#     if not aggregated_content_parts:
#         return json.dumps({"error": "未能从任何提供的 URL 中成功获取内容。"})
#
#
#     # 限制总长度，避免超出 LLM 的 token 限制
#     # 对于非常大的主题，可能需要更复杂的摘要或分块处理策略
#     max_length = 60000  # 设定一个合理的总长度
#     if len(aggregated_content_parts) > max_length:
#         logging.warning(f"聚合内容过长，将被截断至 {max_length} 字符。")
#         aggregated_content_parts = aggregated_content_parts[:max_length]
#
#     # --- 步骤 2: LangChain 调用流程 ---
#     try:
#         parser = JsonOutputParser(pydantic_object=TopicFramework)
#         chain = TOPIC_ANALYSIS_PROMPT | llm | parser
#
#         topic_framework_data = chain.invoke({
#             "aggregated_content": aggregated_content_parts,
#             "format_instructions": parser.get_format_instructions(),
#         })
#
#         topic_framework_json = json.dumps(topic_framework_data, ensure_ascii=False, indent=2)
#         logging.info(f"URL: {url}主题级框架学习成功！")
#
#         # --- 步骤3: 将新信息存入记忆 ---
#         logging.info("Adding current conversation to mem0...")
#
#         # 【核心修改】构造符合官方 API 格式的消息列表
#         messages_to_add = [
#             {"role": "user", "content": query},
#             {"role": "assistant", "content": topic_framework_json}
#         ]
#
#         # 调用 add 方法
#         await loop.run_in_executor(
#             None,
#             lambda: memory_client.add(messages_to_add, user_id=session_id)
#         )
#
#         return topic_framework_json
#
#     except Exception as e:
#         logging.error(f"分析URL: {url}主题级框架时出错: {e}", exc_info=True)
#         return json.dumps({"error": f"分析URL: {url}主题级框架时发生异常: {e}"})





# --- 4. 初始化和主函数 (与之前类似) ---
def initialize_services():
    global llm, memory_client
    logging.info("正在初始化 Topic Learner 服务...")
    load_dotenv(dotenv_path="/Users/dengjuan1/build_agent/.env")
    try:
        llm = ChatOpenAI(
            model="deepseek-chat",
            openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
            openai_api_base=os.getenv("DEEPSEEK_API_BASE"),
            temperature=0.0
        )
        logging.info("deepseek LLM 客户端初始化成功。")
        memory_client = MemoryClient(api_key=os.getenv("MEM0_API_KEY"))
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
    log_file_path = os.path.join(log_directory, "topic_learner.log")

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
    logging.info("Topic Learner MCP 服务已准备就绪...")
    mcp.run(transport='stdio')


if __name__ == "__main__":
    main()