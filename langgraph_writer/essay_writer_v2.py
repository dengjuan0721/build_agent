from dotenv import load_dotenv
import os
from tavily import TavilyClient
import requests #! 新增
import pypdf #! 新增
from io import BytesIO #! 新增
from langchain_text_splitters import RecursiveCharacterTextSplitter #! 新增

from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List
import operator
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_deepseek import ChatDeepSeek

# load environment variables from .env file
_ = load_dotenv(dotenv_path="/Users/dengjuan1/build_agent/.env")

#%%
class AgentState(TypedDict):
    task: str
    plan: str
    draft: str
    critique: str
    content: List[str]
    revision_number: int
    max_revisions: int
    pdf_summary: str #! 新增：用于存储 PDF 的核心内容摘要

#%%
