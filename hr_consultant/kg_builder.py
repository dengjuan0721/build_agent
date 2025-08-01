# mcp_project/kg_builder_server.py

import os
import pandas as pd
from dotenv import load_dotenv
import json
import glob
import logging
from mcp.server.fastmcp import FastMCP
from langchain_openai import ChatOpenAI
from langchain_neo4j import Neo4jGraph

load_dotenv()

# --- 初始化服务 ---
mcp = FastMCP("kg_builder_agent")
llm = None
kg = None

def initialize_services():
    global llm, kg
    if llm is None:
        llm = ChatOpenAI(
            model="deepseek-chat",
            openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
            openai_api_base=os.getenv("DEEPSEEK_API_BASE"),
            temperature=0.0
        )
    if kg is None:
        kg = Neo4jGraph(
            url=os.getenv("NEO4J_URI"),
            username=os.getenv("NEO4J_USERNAME"),
            password=os.getenv("NEO4J_PASSWORD")
        )

# --- Prompt 模板 ---
CYPHER_GENERATION_PROMPT_TEMPLATE = """
You are an expert Neo4j Cypher query generator. Your task is to write a single, parameterized Cypher query that can be used to ingest a batch of data into Neo4j.

**Data Schema:**
The data comes from a Parquet file with the following columns:
{schema_info}

**Knowledge Graph Build Plan:**
The user wants to build a knowledge graph based on the following plan:
"{blueprint_description}"

**Your Task:**
Write a single Cypher query that takes a list of data objects (a batch) as a parameter named `$data`. The query should use `UNWIND` to iterate through the batch. For each item in the batch (named `record`), create the nodes and relationships as described in the build plan.

**Constraints:**
- Use `MERGE` to create nodes to avoid duplicates. Use a unique identifier like `employee_id` if available. For other entities like departments or cities, use the `name` property.
- Use `MERGE` to create relationships to avoid duplicates.
- Use `SET` or `SET e += record` to add properties to the nodes.
- The query MUST be parameterized and expect a single parameter named `$data`.
- Do NOT include any example data or explanations. Provide ONLY the pure Cypher query.

**Example for a simple plan:**
If the plan is "Create Employee nodes with all properties", you might generate:
```cypher
UNWIND $data AS record
MERGE (e:Employee {{employee_id: record.employee_id}})
SET e += record
```
Example for a connected plan:
If the plan is "Create Employee and Department nodes, and a WORKS_IN relationship", you might generate:
UNWIND $data AS record
MERGE (d:Department {{name: record.department}})
MERGE (e:Employee {{employee_id: record.employee_id}})
MERGE (e)-[:WORKS_IN]->(d)
SET e += record
Now, generate the Cypher query for the provided schema and build plan.
Generated Cypher Query:
"""
@mcp.tool()
def execute_kg_build_plan(directory_path: str, blueprint_description: str) -> str:
    """
    Executes a knowledge graph build plan. It reads all Parquet files from a directory,
    generates a Cypher query based on the blueprint, and ingests the data into Neo4j.
    """
    initialize_services()
    logging.info(f"Executing KG build plan for directory: {directory_path}")
    logging.info(f"Blueprint: {blueprint_description}")
    try:
        # --- 1. 验证路径并读取 Schema ---
        parquet_files = glob.glob(os.path.join(directory_path, '*.parquet'))
        if not parquet_files:
            return json.dumps({"error": "No Parquet files found in the specified directory."})

        # 读取第一个文件来获取 schema
        df_sample = pd.read_parquet(parquet_files[0], engine='pyarrow')
        schema_info = "\n".join([f"- `{col}` ({dtype})" for col, dtype in df_sample.dtypes.items()])

        # --- 2. 让 LLM 生成 Cypher 查询 ---
        prompt = CYPHER_GENERATION_PROMPT_TEMPLATE.format(
            schema_info=schema_info,
            blueprint_description=blueprint_description
        )
        print("Generating Cypher query from blueprint...")
        response = llm.invoke(prompt)
        logging.info(f"response:{response}")
        generated_cypher = response.content.strip().replace("```cypher", "").replace("```", "").strip()
        logging.info(f"Generated Cypher:\n{generated_cypher}")

        # --- 3. 流式处理文件并执行 Cypher ---
        total_processed_records = 0
        batch_size = 100
        for file_path in parquet_files:
            df = pd.read_parquet(file_path, engine='pyarrow')
            for i in range(0, len(df), batch_size):
                batch_df = df.iloc[i:i + batch_size]
                batch_records = batch_df.to_dict('records')

                # 执行 LLM 生成的 Cypher
                kg.query(generated_cypher, params={"data": batch_records})

            total_processed_records += len(df)
            logging.info(
                f"Processed file {os.path.basename(file_path)}. Total records ingested so far: {total_processed_records}")

        success_message = f"Successfully executed the build plan. Ingested {total_processed_records} records into the knowledge graph using the generated Cypher query."
        return json.dumps({"status": "success", "message": success_message})

    except Exception as e:
        error_message = f"An error occurred during execution: {e}"
        logging.info(f"[ERROR] {error_message}")
        return json.dumps({"error": error_message, "details": str(e)})

def main():
    load_dotenv(dotenv_path="/Users/dengjuan1/build_agent/.env")
    logging.info("Knowledge Graph Builder MCP Server starting...")
    mcp.run(transport='stdio')

if __name__ == "__main__":
    main()