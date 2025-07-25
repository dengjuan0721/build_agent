import sys

import httpx
from dotenv import load_dotenv
import os
from openai import OpenAI
from langchain_community.graphs import Neo4jGraph
#%%

# Warning control
import warnings
warnings.filterwarnings("ignore")

load_dotenv('.env', override=True)
print("NEO4J_URI =", os.getenv('NEO4J_URI'))
#%%
#数据库连接
NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USERNAME = os.getenv('NEO4J_USERNAME')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
NEO4J_DATABASE = os.getenv('NEO4J_DATABASE')

kg = Neo4jGraph(
    url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD, database=NEO4J_DATABASE
)
#%%
#这部分是cypher test
cypher1 = """
  MATCH (n) 
  RETURN count(n) AS numberOfNodes
  """

result_node = kg.query(cypher1)
print(f"There are {result_node[0]['numberOfNodes']} nodes in this graph.")

cypher2 = """
  MATCH (n:employee) 
  RETURN count(n) AS numberOfEmployees
  """

result_employee = kg.query(cypher2)
print(f"There are {result_employee[0]['numberOfEmployees']} employee nodes in this graph.")

cypher3="""
  MATCH (Alice:employee {employee_name:"Alice"}) 
  RETURN Alice
  """
result_alice = kg.query(cypher3)
print(f"Alice info: {result_alice[0]} ")

cypher4="""
  MATCH (Alice:employee {employee_name:"Alice"}) 
  RETURN Alice.employee_name as name
  """
result_alice_name = kg.query(cypher4)
print(f"Alice name: {result_alice_name[0]['name']} ")

cypher5="""
  MATCH (employee:employee)-[:works_at]->(department:department) 
  RETURN employee.employee_name as name
  """
result_employee2 = kg.query(cypher5)
print(result_employee2)

cypher6="""
  MATCH (employee:employee)-[:works_at]->(eng_department:department {department_name:"Engineering"}) 
  RETURN employee.employee_name as name
  """
result_employee3 = kg.query(cypher6)
print(result_employee3)

#%%
#下面我们需要让LLM学会匹配我们Graph，我们就需要Embedding模型来加入
# 嵌入模型名称（1536维度）
import google.generativeai as genai
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# EMBED_MODEL = "text-embedding-3-small"

# 关键代码：创建一个明确不使用任何代理的httpx客户端
client_with_no_proxy = httpx.Client(proxy=None)

# 将这个客户端传递给gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_embedding_gemini(text, model="models/embedding-001", task_type='RETRIEVAL_QUERY'):
    # 2. 调用Google的embed_content方法
    response = genai.embed_content(
        model=model,  # 使用Google的Embedding模型名
        content=text,
        task_type=task_type
    )
    # 3. 提取embedding向量的方式略有不同
    return response['embedding']

def embed_employees_with_kg(kg: Neo4jGraph):
    """从 Neo4j 中提取 bio，生成 embedding 并写回"""
    query = "MATCH (e:employee) RETURN e.employee_name AS name, e.bio AS bio"
    results = kg.query(query)

    for record in results:
        name = record["name"]
        bio = record["bio"]
        if bio:
            print(f"📎 正在处理：{name}")
            embedding = get_embedding_gemini(bio,task_type="RETRIEVAL_DOCUMENT")
            update_query = """
            MATCH (e:employee {employee_name: $name})
            SET e.bioEmbedding = $embedding
            """
            kg.query(update_query, params={"name": name, "embedding": embedding})
            print(f"✅ 已写入 embedding：{name}")


embed_employees_with_kg(kg)

kg.query("""
  CREATE VECTOR INDEX employee_embeddings IF NOT EXISTS
  FOR (e:employee) ON (e.bioEmbedding) 
  OPTIONS { indexConfig: {
    `vector.dimensions`: 768,
    `vector.similarity_function`: 'cosine'
  }}"""
)
#%%
#相似度计算：我们已经拥有了嵌入向量，现在我们来提出问题
# --- 这是您新的相似度搜索函数 ---
kg = Neo4jGraph(
    url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD, database=NEO4J_DATABASE
)
def get_embedding_gemini(text, model="models/embedding-001", task_type='RETRIEVAL_QUERY'):
    # 2. 调用Google的embed_content方法
    response = genai.embed_content(
        model=model,  # 使用Google的Embedding模型名
        content=text,
        task_type=task_type
    )
    # 3. 提取embedding向量的方式略有不同
    return response['embedding']
def find_similar_movies(question, top_k=5):
    # 步骤一：在Python中为用户问题生成Embedding
    print(f"正在为问题 '{question}' 生成Gemini Embedding...")
    question_embedding = get_embedding_gemini(question, task_type="RETRIEVAL_QUERY")

    # 步骤二：将生成的向量作为参数传入Cypher查询
    print(f"使用生成的向量在Neo4j中进行相似度搜索...")

    # 关键：Cypher查询不再调用genai.vector.encode，而是直接接收一个$question_embedding参数
    cypher_query = """
        CALL db.index.vector.queryNodes(
            'employee_embeddings',  // <--- 1. 使用您为Gemini数据创建的索引名
            $top_k, 
            $question_embedding         // <--- 2. 直接使用传入的向量参数
        ) YIELD node AS employee, score
        MATCH (employee)-[:works_at]->(d:department)
        RETURN 
            employee.employee_name AS employeeName, 
            employee.bio AS bio, 
            d.department_name AS departmentName, // <-- 返回部门名称
            score
    """

    # 准备要传入的参数
    params = {
        "question_embedding": question_embedding,  # <--- 3. 将Python中的向量变量传给Cypher参数
        "top_k": top_k
    }

    # 执行查询
    results = kg.query(cypher_query, params=params)

    return results


# --- 调用示例 ---
my_question = "A human resource professional"
similar_movies = find_similar_movies(my_question)

#%%
#加入langchain来自动编排以上的api调用
# Langchain
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_openai import ChatOpenAI

# 定义参数
INDEX_NAME = "employee_embeddings"  # 您为Gemini数据创建的向量索引名
NODE_LABEL = "employee"                      # 节点的标签 (注意大小写，通常首字母大写)
TEXT_PROPERTY = "bio"                      # 包含源文本的属性名
EMBEDDING_PROPERTY = "bioEmbedding"        # 包含向量的属性名
gemini_embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY") # <-- 关键就在这一行
)
# 使用 from_existing_graph 创建向量存储对象
employee_vector_store = Neo4jVector.from_existing_graph(
    embedding=gemini_embeddings,          # <-- 使用Gemini Embedding模型
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    index_name=INDEX_NAME,                # <-- 填入正确的索引名
    node_label=NODE_LABEL,                # <-- 填入正确的节点标签
    text_node_properties=[TEXT_PROPERTY], # <-- 这是一个列表，填入源文本属性名
    embedding_node_property=EMBEDDING_PROPERTY, # <-- 填入向量属性名
)
#%%
#开始问答
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI

# 1. 将向量存储包装成检索器
retriever = employee_vector_store.as_retriever()

import google.generativeai as genai

# 配置您的API Key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

print("--- 可用的模型列表 ---")
for m in genai.list_models():
  # 检查这个模型是否支持 'generateContent' 方法
  if 'generateContent' in m.supported_generation_methods:
    print(m.name)

# 2. 初始化一个聊天模型 (比如Gemini Pro)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")

# 3. 构建问答链
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff", # 或者 "map_reduce", "refine"
    retriever=retriever,
    return_source_documents=True # 建议开启，方便查看检索到了哪些内容
)

# 4. 进行提问！
question = "Who is an expert in data science and graph technology?"
result = qa_chain.invoke({"query": question})

print("--- Answer ---")
print(result['result'])
#%%
#上面的这个回答差强人意，我们希望回答中可以包含人名
#方式1:让llm去写cypher回答问题
#失败：本质上可以，但是我没有安装genai
from langchain.chains import GraphCypherQAChain
from langchain_community.graphs import Neo4jGraph

# 1. 创建一个Neo4jGraph的实例，它知道图的schema
graph = Neo4jGraph(
    url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD
)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")

# 2. 创建Cypher问答链
cypher_qa_chain = GraphCypherQAChain.from_llm(
    graph=graph,
    llm=llm,
    verbose=True, # 开启verbose可以看到LLM生成的Cypher语句
    allow_dangerous_requests=True
    #但是这个行为非常危险，会触发langchain的安全提示：
    #In order to use this chain, you must acknowledge that it can make dangerous requests by setting `allow_dangerous_requests` to `True`.
)

# 3. 直接提问
question = "Who is an expert in data science and graph technology? Return their name."
result = cypher_qa_chain.invoke({"query": question})

print(result['result'])
#%%
#改Prompt检索meta
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Neo4jVector
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# --- 1. 初始化组件 (和之前一样) ---
employee_vector_store = Neo4jVector.from_existing_graph(
    embedding=gemini_embeddings,          # <-- 使用Gemini Embedding模型
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    index_name=INDEX_NAME,                # <-- 填入正确的索引名
    node_label=NODE_LABEL,                # <-- 填入正确的节点标签
    text_node_properties=[TEXT_PROPERTY], # <-- 这是一个列表，填入源文本属性名
    embedding_node_property=EMBEDDING_PROPERTY, # <-- 填入向量属性名
)
retriever = employee_vector_store.as_retriever()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")

# --- 2. 定义如何格式化检索到的文档 ---
# 这是一个辅助函数，它接收一个Document列表，并将其格式化成我们想要的字符串
def format_docs(docs):
    formatted_docs = []
    for doc in docs:
        # 从metadata中提取人名，如果找不到则用'Unknown'
        employee_name = doc.metadata.get('employee_name', 'Unknown Employee')
        # 拼接成清晰的格式
        formatted_doc = f"Employee Name: {employee_name}\nBio: {doc.page_content}"
        formatted_docs.append(formatted_doc)
    # 用两个换行符将所有格式化后的文档拼接起来
    return "\n\n".join(formatted_docs)


# --- 3. 创建Prompt模板 ---
# 这个模板和之前一样，但它更符合LCEL的风格
template = """
You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
When you answer, you MUST mention the name of the employee who is the expert.

Context:
{context}

Question: {question}

Answer:
"""
prompt = ChatPromptTemplate.from_template(template)


# --- 4. 使用LCEL构建问答链 (核心部分) ---
rag_chain = (
    # 这一部分定义了如何准备Prompt的输入变量
    {
        "context": retriever | RunnableLambda(format_docs), # 关键：先用retriever检索，然后用我们的函数格式化结果
        "question": RunnablePassthrough()                  # 用户的原始问题直接传递过去
    }
    | prompt          # 将准备好的变量填入Prompt模板
    | llm             # 将填充好的Prompt发送给LLM
    | StrOutputParser() # 将LLM的输出解析为字符串
)


# --- 5. 调用链并提问 ---
question = "Who is an expert in data science and graph technology?"
answer = rag_chain.invoke(question) # LCEL链的调用更直接

print("--- Answer ---")
print(answer)