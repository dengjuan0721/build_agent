# mcp_project/data_architect_server.py

import os
import pandas as pd
from dotenv import load_dotenv
import json
import logging # 导入 logging
import httpx

from mcp.server.fastmcp import FastMCP
from langchain_openai import ChatOpenAI
# --- 初始化 日志 ---
log_file_path = f"logs/data_architect_server.log"
os.makedirs(os.path.dirname(log_file_path), exist_ok=True) # 确保 logs 目录存在

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path), # 写入文件
        # logging.StreamHandler() # 如果你还想在它独立运行时在控制台看到输出
    ]
)

# --- 初始化 FastMCP 和 LLM ---
mcp = FastMCP("data_architect_agent")
llm = None

def initialize_llm():
    global llm
    if llm is None:
        print("Initializing LLM for data analysis...")
        # custom_http_client = httpx.Client(
        #     proxies={},  # 明确设置为空字典，禁用任何系统代理
        #     timeout=120.0,  # 将超时时间延长到 120 秒
        # )
        llm = ChatOpenAI(
            model="deepseek-chat",
            openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
            openai_api_base=os.getenv("DEEPSEEK_API_BASE"),
            temperature=0.2, # 使用较低的温度以获得更稳定的结构化输出
            # 传递自定义的 HTTP 客户端
            # http_client=custom_http_client,
            # 明确禁用 LangChain 的默认重试机制
            max_retries=0
        )
        print("LLM initialized.")

# --- Prompt 模板 ---
KG_DESIGN_PROMPT_TEMPLATE = """
You are an expert Data Architect and Knowledge Graph designer. 
Your task is to analyze the schema and sample data from a Parquet file and propose several design blueprints for building a Knowledge Graph in Neo4j.

Here is the information extracted from the Parquet file:

**Column Schema and Data Types:**
{schema_info}

**Sample Data (first 5 rows):**
{sample_data_json}
Please provide a comprehensive analysis and design document in Markdown format. The document should include the following sections:
1. Data Summary and Field Analysis
Briefly summarize the dataset's purpose based on the fields.
For each field, provide a description of its likely meaning and purpose.
2. Knowledge Graph Design Blueprints
Propose at least two distinct blueprints for modeling this data as a knowledge graph. For each blueprint, provide:
A clear description of the design philosophy.
The proposed Nodes and their properties (e.g., (:employee {{name, age}})).
The proposed Relationships and their direction (e.g., (:employee)-[:WORKS_IN]->(:Department)).
A brief analysis of the Pros and Cons of this design.
Blueprint A: The Simple Property Graph Model
This model typically uses a single primary node type and stores most other information as properties.
Blueprint B: The Richly Connected Entity Model (Recommended)
This model identifies potential entities (like cities, departments) and promotes them to first-class nodes, creating a more connected and queryable graph.
Blueprint C (Optional): The Advanced Inferred Model
If applicable, suggest how advanced techniques like Named Entity Recognition (NER) on text fields (like 'bio') could create new nodes and relationships (e.g., (:employee)-[:HAS_SKILL]->(:Skill {{name: 'Python'}})).
Provide a clear, well-structured, and insightful response that a data engineer could use to start building the knowledge graph.
"""

@mcp.tool()
def analyze_parquet_for_kg_design(file_path: str) -> str:
    """
    Analyzes a Parquet file to understand its structure and proposes multiple Knowledge Graph design blueprints.
    Takes a file path as input and returns a Markdown-formatted analysis.
    """
    initialize_llm() # 确保 LLM 已初始化
    logging.info(f"Analyzing Parquet file: {file_path}")

    if not os.path.exists(file_path):
        return json.dumps({"error": f"File not found at path: {file_path}"})

    try:
        # 1. 读取文件样本以分析结构
        # 只读取前1000行作为样本，避免内存问题
        df_sample = pd.read_parquet(file_path, engine='pyarrow').head(1000)

        # 2. 提取 Schema 和样本数据
        schema_info = "\n".join([f"- `{col}`: `{dtype}`" for col, dtype in df_sample.dtypes.items()])
        sample_data_json = df_sample.head(5).to_json(orient="records", indent=2)

        # 3. 构造 Prompt
        prompt = KG_DESIGN_PROMPT_TEMPLATE.format(
            file_path=file_path,
            schema_info=schema_info,
            sample_data_json=sample_data_json
        )
        logging.info(f"{prompt}")
        # 4. 调用 LLM 生成分析报告
        logging.info("Invoking LLM to generate KG design blueprints...")
        response = llm.invoke(prompt)

        # response.content 就是 LLM 返回的 Markdown 字符串
        analysis_report = response.content
        logging.info("Successfully generated analysis report...")
        return analysis_report

    except Exception as e:
        error_message = f"An error occurred while analyzing the file: {e}"
        logging.info(f"[ERROR] {error_message}")
        return json.dumps({"error": error_message})

def main():
    load_dotenv(dotenv_path="/Users/dengjuan1/build_agent/.env")
    logging.info("Data Architect MCP Server starting...")
    mcp.run(transport='stdio')

if __name__ == "__main__":
    main()