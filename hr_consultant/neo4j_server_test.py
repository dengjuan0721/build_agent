import os
import json
from dotenv import load_dotenv

# --- 1. 导入 FastMCP 和 LangChain 组件 ---
from mcp.server.fastmcp import FastMCP
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
# neo4j_lcel_server.py
from openai import OpenAI
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.embeddings.dashscope import DashScopeEmbeddings
# 或者，在某些最新版本中，它可能直接是:
# from langchain_dashscope import DashscopeEmbeddings
from langchain_neo4j import Neo4jGraph

load_dotenv(dotenv_path="/Users/dengjuan1/build_agent/.env")

# --- 2. 创建 FastMCP 实例 ---
mcp = FastMCP("knowledge_graph_agent")

# --- 3. 全局/应用上下文的状态管理 ---
# 因为函数是无状态的，我们需要一个地方来持有昂贵的、可复用的 RAG 链对象。
# 一个简单的字典或一个专门的类实例都可以。
app_context = {}


@mcp.tool()
def setup_knowledge_graph_index() -> str:
    """
    Checks if the knowledge graph vector index exists. If not, it generates embeddings for all employees and creates the index.
    This is a one-time setup or maintenance tool.
    """
    kg = app_context["kg"]
    embeddings = app_context["embeddings"]
    index_name = "employee_embeddings"

    # 步骤 1: 检查索引是否存在
    print(f"Checking for index '{index_name}'...")
    indexes_info = kg.query(f"SHOW VECTOR INDEXES WHERE name = '{index_name}'")

    if indexes_info:
        message = f"Index '{index_name}' already exists. No action needed."
        print(message)
        return json.dumps({"status": "success", "message": message, "details": indexes_info[0]})

    # kg.query(f"DROP INDEX {index_name}")

    # 步骤 2: 如果索引不存在，则生成 Embeddings
    print(f"Index not found. Starting embedding process for employees...")

    # 从 Neo4j 中提取需要 embedding 的数据
    query = "MATCH (e:employee) WHERE e.bio IS NOT NULL RETURN e.employee_name AS name, e.bio AS bio"
    results = kg.query(query)

    if not results:
        message = "No employees with 'bio' property found to embed."
        print(message)
        return json.dumps({"status": "skipped", "message": message})

    embedded_count = 0
    for record in results:
        name, bio = record["name"], record["bio"]
        if bio:
            print(f"📎 Embedding for: {name}")
            # 使用 TongyiEmbeddings 生成向量
            embedding_vector = embeddings.embed_query(bio)

            # 将向量写回 Neo4j
            update_query = """
            MATCH (e:employee {employee_name: $name})
            SET e.bioEmbedding = $embedding
            """
            kg.query(update_query, params={"name": name, "embedding": embedding_vector})
            print(f"✅ Stored embedding for: {name}")
            embedded_count += 1

    # 步骤 3: 创建向量索引
    print(f"Creating vector index '{index_name}' with dimension 1536 for Qwen...")
    kg.query(f"""
      CREATE VECTOR INDEX {index_name} IF NOT EXISTS
      FOR (e:employee) ON (e.bioEmbedding) 
      OPTIONS {{ indexConfig: {{
        `vector.dimensions`: 1536,
        `vector.similarity_function`: 'cosine'
      }}
    }}""")

    message = f"Successfully embedded {embedded_count} employees and created index '{index_name}'."
    print(message)
    return json.dumps({"status": "success", "message": message})

@mcp.tool()
def initialize_rag_chain():
    """
    初始化 LangChain RAG 链并将其存储在全局上下文中。
    这个函数只会在服务器启动时执行一次。
    """
    if app_context.get("rag_chain"):
        return

    print("Initializing Neo4j LCEL RAG Chain...")

    app_context["kg"] = Neo4jGraph(
        url=os.getenv("NEO4J_URI"),
        username=os.getenv("NEO4J_USERNAME"),
        password=os.getenv("NEO4J_PASSWORD"),
        database=os.getenv("NEO4J_DATABASE"),
    )
    # 初始化组件 (和之前完全一样)
    app_context["embeddings"] = DashScopeEmbeddings(
        dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
        model="text-embedding-v2",
    )

    employee_vector_store = Neo4jVector.from_existing_index(
        embedding=app_context["embeddings"],
        url=os.getenv("NEO4J_URI"),
        username=os.getenv("NEO4J_USERNAME"),
        password=os.getenv("NEO4J_PASSWORD"),
        database=os.getenv("NEO4J_DATABASE"),
        index_name="employee_embeddings",
    )

    retriever = employee_vector_store.as_retriever()

    llm = ChatOpenAI(
        model="deepseek-chat",
        openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
        openai_api_base=os.getenv("DEEPSEEK_API_BASE"),
        temperature=0
    )

    def format_docs(docs):
        formatted_docs = []
        for doc in docs:
            employee_name = doc.metadata.get('employee_name', 'Unknown Employee')
            formatted_doc = f"Employee Name: {employee_name}\nBio: {doc.page_content}"
            formatted_docs.append(formatted_doc)
        return "\n\n".join(formatted_docs)

    template = """You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. 
    When you answer, you MUST mention the name of the employee who is the expert.

    Context:
    {context}
    
    Question: {question}
    
    Answer:"""
    prompt = ChatPromptTemplate.from_template(template)

    # 构建并存储 RAG 链
    rag_chain = (
            {"context": retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

    app_context["rag_chain"] = rag_chain
    print("Neo4j LCEL RAG Chain initialized successfully.")


# --- 4. 使用 @mcp.tool() 装饰器暴露工具 ---

@mcp.tool()
def ask_knowledge_graph(question: str) -> str:
    """
    Answers questions about employees, their skills, and expertise by querying a knowledge graph.
    Use this for questions like 'Who is an expert in data science?' or 'Tell me about Michael's background'.
    """
    print(f"Invoking RAG chain with question: '{question}'")

    # 从上下文中获取已初始化的 RAG 链
    rag_chain = app_context.get("rag_chain")
    if not rag_chain:
        # 这是一个备用安全措施，正常情况下不会执行
        return json.dumps({"error": "RAG chain is not initialized."})

    try:
        # 调用 RAG 链
        answer = rag_chain.invoke(question)
        print(f"RAG chain returned answer: '{answer}'")
        # FastMCP 会自动处理返回的字符串
        return answer
    except Exception as e:
        print(f"Error invoking RAG chain: {e}")
        return json.dumps({"error": str(e)})


if __name__ == "__main__":
    initialize_rag_chain()
    mcp.run(transport='stdio')