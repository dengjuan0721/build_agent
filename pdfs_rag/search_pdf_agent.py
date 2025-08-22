# 文件: pdfs_rag/agent_builder.py

import os
from pathlib import Path
from typing import List

# LangChain 和 LlamaIndex 的核心 imports
from langchain_deepseek import ChatDeepSeek
from langchain_community.embeddings import DashScopeEmbeddings
from llama_index.core import Settings
from llama_index.core.agent import ReActAgent
from llama_index.llms.langchain import LangChainLLM
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.core.tools.types import BaseTool

# 导入你自己的工具创建函数
from utils.get_doc_tool import get_doc_tools

# 可以在这里定义一个全局变量，只加载一次模型和工具
# 这样可以避免每次调用创建函数时都重新加载
_AGENT_INSTANCE = None


def _initialize_agent():
    """
    一个内部函数，负责所有的初始化工作。
    只在第一次被调用时执行。
    """
    print("--- Initializing RAG Agent for the first time ---")

    # --- 1. 配置全局 LLM 和 Embedding Model ---
    lc_llm = ChatDeepSeek(
        model="deepseek-reasoner",
        api_base=os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1"),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        temperature=0.1
    )
    lc_embed_model = DashScopeEmbeddings(
        model="text-embedding-v4"
    )

    llm = LangChainLLM(llm=lc_llm)
    embed_model = LangchainEmbedding(lc_embed_model)

    Settings.llm = llm
    Settings.embed_model = embed_model
    print("--- LLM and Embedding Model configured ---")

    # --- 2. 加载 PDF 并创建工具 ---
    pdf_directory = "./downloaded_papers/"
    papers = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]

    if not papers:
        raise FileNotFoundError(f"No PDF files found in the directory: {pdf_directory}")

    paper_to_tools_dict = {}
    for paper in papers:
        print(f"Getting tools for paper: {paper}")
        paper_path = os.path.join(pdf_directory, paper)
        # 假设 get_doc_tools 返回的是一个工具列表
        vector_tool, summary_tool = get_doc_tools("./downloaded_papers/" + paper, Path(paper).stem)
        paper_to_tools_dict[paper] = [vector_tool, summary_tool]

    initial_tools = [t for paper in papers for t in paper_to_tools_dict[paper]]
    print(f"--- Loaded {len(initial_tools)} tools from {len(papers)} papers ---")

    # --- 3. 创建并返回 Agent 实例 ---
    agent = ReActAgent(
        tools=initial_tools,
        llm=llm,
        verbose=True
    )
    # 对于新版本
    # agent = ReActAgent(
    #     tools=initial_tools,
    #     llm=llm,
    #     verbose=True
    # )

    return agent


def get_rag_agent() -> ReActAgent:
    """
    获取一个配置好的、基于多个PDF的RAG Agent。
    这个函数使用单例模式，确保 Agent 只被初始化一次。

    Returns:
        An instance of ReActAgent configured with PDF tools.
    """
    global _AGENT_INSTANCE

    if _AGENT_INSTANCE is None:
        _AGENT_INSTANCE = _initialize_agent()

    return _AGENT_INSTANCE

# 注意：`nest_asyncio` 通常应该在主应用程序的入口处调用，而不是在库模块里。
# 因为它会修改全局的 asyncio 事件循环策略。
# 我们暂时把它注释掉，让调用者来决定是否需要它。
# import nest_asyncio
# nest_asyncio.apply()