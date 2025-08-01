# ingest_from_partitioned_data.py (Streaming/Iterative Version)

import os
import pandas as pd
from dotenv import load_dotenv
import glob  # 用于查找文件
from langchain_neo4j import Neo4jGraph
from langchain_community.embeddings import DashScopeEmbeddings

load_dotenv()


def ingest_data_to_neo4j_streamed():
    """
    流式处理分区目录下的每个 Parquet 文件，以节省内存。
    """

    # --- 1. 初始化服务 (保持不变) ---
    print("Initializing services (Neo4j, Embeddings)...")
    try:
        kg = Neo4jGraph(
        url=os.getenv("NEO4J_URI"),
        username=os.getenv("NEO4J_USERNAME"),
        password=os.getenv("NEO4J_PASSWORD"),
        database=os.getenv("NEO4J_DATABASE"),
        )
        embeddings = DashScopeEmbeddings(
        dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
        model="text-embedding-v2",
        )
    except Exception as e:
        print(f"[FATAL ERROR] Failed to initialize services: {e}")
        return

    # --- 2. 查找所有 Parquet 文件，而不是一次性读取 ---
    partition_directory = "~/build_agent/data_lake/employees/date=20250730"

    if not os.path.exists(partition_directory):
        print(f"[FATAL ERROR] Directory not found: {partition_directory}")
        return

    # 使用 glob 找到目录下所有的 .parquet 文件
    # partition_directory + '/*.parquet' 会匹配所有以 .parquet 结尾的文件
    parquet_files = glob.glob(os.path.join(partition_directory, '*.parquet'))

    if not parquet_files:
        print(f"No Parquet files found in directory: '{partition_directory}'")
        return

    print(f"Found {len(parquet_files)} Parquet files to process iteratively.")
    total_processed_records = 0

    # --- 3. 遍历并处理每一个 Parquet 文件 ---
    for file_num, file_path in enumerate(parquet_files, 1):
        print(f"\n--- Processing File {file_num}/{len(parquet_files)}: {os.path.basename(file_path)} ---")

        try:
            # 只将当前文件读入内存
            df = pd.read_parquet(file_path, engine='pyarrow')
            print(f"  Loaded {len(df)} records from this file.")
        except Exception as e:
            print(f"  [ERROR] Failed to read Parquet file {file_path}: {e}")
            continue  # 跳过这个文件，继续处理下一个

        # --- 4. 对当前文件的 DataFrame 进行批量处理 (这部分逻辑和之前一样) ---
        batch_size = 100
        for i in range(0, len(df), batch_size):
            batch_df = df.iloc[i:i + batch_size]
            print(f"    Processing batch {i // batch_size + 1}...")

            batch_records = batch_df.to_dict('records')
            texts_to_embed = [record["bio"] for record in batch_records]

            try:
                embedding_vectors = embeddings.embed_documents(texts_to_embed)
            except Exception as e:
                print(f"      [ERROR] Failed to generate embeddings for this batch: {e}")
                continue

            nodes_to_create = []
            for record, vector in zip(batch_records, embedding_vectors):
                node_data = record.copy()
                node_data['bioEmbedding'] = vector
                nodes_to_create.append(node_data)

            try:
                kg.query("""
                UNWIND $nodes AS node_data
                MERGE (e:employee {employee_id: node_data.employee_id})
                SET e += node_data
                """, params={"nodes": nodes_to_create})
            except Exception as e:
                print(f"      [ERROR] Failed to write this batch to Neo4j: {e}")
                continue

        total_processed_records += len(df)
        print(f"  Finished processing file. Total records ingested so far: {total_processed_records}")

    print(f"\nAll {len(parquet_files)} files have been processed.")
    print(f"Total records ingested into Neo4j: {total_processed_records}")

    # --- 5. 在所有文件处理完毕后，统一创建索引 ---
    print("\nCreating vector index if it doesn't exist...")
    try:
        kg.query("""
          CREATE VECTOR INDEX employee_embeddings IF NOT EXISTS
          FOR (e:employee) ON (e.bioEmbedding) 
          OPTIONS { indexConfig: {
            `vector.dimensions`: 1536,
            `vector.similarity_function`: 'cosine'
          }}
        """)
        print("Index setup is complete.")
    except Exception as e:
        print(f"[ERROR] Failed to create vector index: {e}")


if __name__ == "__main__":
    ingest_data_to_neo4j_streamed()