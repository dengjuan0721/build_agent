import asyncio
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


# --- 1. 创建一个封装所有逻辑的服务器类 ---
class Neo4jServer:
    def __init__(self):
        """
        构造函数：初始化 FastMCP 实例和应用上下文。
        """
        print("Initializing Neo4jServer...")
        self.mcp = FastMCP("knowledge_graph_agent")
        self.app_context = {}
        self._initialize_services()
        self._register_tools()

    def _initialize_services(self):
        """
        私有方法：初始化所有需要的服务和对象，并存入 self.app_context。
        """
        print("Initializing backend services (LLM, Graph)...")
        self.app_context["kg"] = Neo4jGraph(
            url=os.getenv("NEO4J_URI"),
            username=os.getenv("NEO4J_USERNAME"),
            password=os.getenv("NEO4J_PASSWORD"),
            database=os.getenv("NEO4J_DATABASE"),
        )
        self.app_context["llm"] = ChatOpenAI(
            model="deepseek-chat",
            openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
            openai_api_base=os.getenv("DEEPSEEK_API_BASE"),
            temperature=0
        )
        self.app_context["embeddings"] = DashScopeEmbeddings(
            dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
            model="text-embedding-v2",
        )

        employee_vector_store = Neo4jVector.from_existing_graph(
            embedding=self.app_context["embeddings"],                  # 你的 embedding 实例
            url=os.getenv("NEO4J_URI"),            # Neo4j URL
            username=os.getenv("NEO4J_USERNAME"),  # Neo4j 用户名
            password=os.getenv("NEO4J_PASSWORD"),  # Neo4j 密码
            database=os.getenv("NEO4J_DATABASE"),  # Neo4j 数据库名
            index_name="employee_embeddings",      # 你的向量索引名
            node_label="employee",                 # 【关键】节点标签
            text_node_properties=["bio"],          # 【关键修复!】告诉 LangChain 从 'bio' 属性读取文本
            embedding_node_property="bioEmbedding", # 【关键】包含向量的属性
        )

        retriever = employee_vector_store.as_retriever()

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
                | self.app_context["llm"]
                | StrOutputParser()
        )

        self.app_context["rag_chain"] = rag_chain
        print("Neo4j LCEL RAG Chain initialized successfully.")

    def _register_tools(self):
        """
        私有方法：将类中的方法注册为 MCP 工具。
        """
        # 使用 self.mcp.tool() 作为装饰器来包装我们的方法
        # 这会将它们注册到 self.mcp 实例中
        self.setup_knowledge_graph_index = self.mcp.tool()(self.setup_knowledge_graph)
        self.ask_knowledge_graph = self.mcp.tool()(self.ask_knowledge_graph)
        print("Tools registered.")

    # --- 2. 定义工具逻辑为类方法 ---
    # 注意：这里不再需要 @mcp.tool() 装饰器，因为我们在 __init__ 中手动注册了
    # 但为了可读性，我们可以保留它，并动态应用

    def setup_knowledge_graph(self) -> str:
        """
        Checks if the knowledge graph vector index exists. If not, it generates embeddings and creates the index.
        This is a one-time setup or maintenance tool.
        """
        print(f"Checking for index 'employee_embeddings'...")
        index_name = "employee_embeddings"
        kg = self.app_context["kg"]
        # 步骤 1: 检查索引是否存在
        print(f"Checking for index '{index_name}'...")
        indexes_info = kg.query(f"SHOW VECTOR INDEXES WHERE name = '{index_name}'")

        if indexes_info:
            message = f"Index '{index_name}' already exists. No action needed."
            print(message)
            return json.dumps({"status": "success", "message": message, "details": indexes_info[0]})

    def ask_knowledge_graph(self, question: str) -> str:
        """
        Answers questions about employees, their skills, and expertise by querying a knowledge graph.
        Use this for questions like 'Who is an expert in data science?' or 'Tell me about Michael's background'.
        """
        print(f"Invoking RAG chain with question: '{question}'")

        # 从上下文中获取已初始化的 RAG 链
        rag_chain = self.app_context.get("rag_chain")
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

    def run(self):
        """
        启动 MCP Stdio 服务器。
        """
        print("Starting MCP server...")
        self.mcp.run(transport='stdio')


# --- 3. 修改 main 函数以使用我们的新类 ---
def main():
    load_dotenv(dotenv_path="/Users/dengjuan1/build_agent/.env")
    server = Neo4jServer()
    server.run()


if __name__ == "__main__":
    main()