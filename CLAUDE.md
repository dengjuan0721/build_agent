# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

这是一个基于LangGraph构建的多智能体AI系统，用于自动化论文写作和内容生成。系统结合了多个专业化智能体，包括规划、写作、编辑和记忆管理，能够生成结构化的学术内容，并集成PDF研究功能。

## 核心架构

项目采用模块化智能体架构，包含以下主要组件：

### 主要工作流智能体 (`langgraph_writer/`)
- **PlannerGraph**: 根据用户提示生成结构化内容计划
- **AiParserGraph**: 将非结构化计划转换为层次化JSON结构
- **SectionWriterGraph**: 基于结构化计划编写各个章节
- **ConnectorWriterGraph**: 创建章节之间的过渡连接
- **EditGraph**: 处理修订和编辑工作流，支持用户反馈

### 记忆系统 (`memory_chat/`)
- **Memory Agent**: 使用mem0和向量存储管理长期记忆
- **Stores**: 使用SQLite和向量相似度搜索处理语义记忆
- **Ingest Functions**: 将笔记本和内容摄取到长期记忆中

### PDF研究模块 (`pdfs_rag/`)
- **Search PDF Agent**: 针对学术论文的检索增强生成
- 集成ArXiv API进行论文发现和分析

### 工具函数 (`utils/`)
- **Recursion Handlers**: 递归处理层次化内容结构
- **Schema Definitions**: 结构化数据的Pydantic模型 (PlanSection, StructuredPlan)
- **File Management**: 将结构化内容保存到有序目录
- **Concurrent Processing**: 并行章节生成的异步工具

## 常用开发命令

### 环境配置
```bash
# 安装依赖
pip install -r requirements.txt

# 或使用uv（推荐）
uv sync
```

### 运行系统
```bash
# 主入口点用于论文生成
python main.py
```

### 依赖项
- **核心**: anthropic, openai, langchain, langgraph
- **记忆**: mem0, 向量数据库 (SQLite with VSS)
- **异步**: nest-asyncio 用于并发处理
- **数据**: pandas, pydantic 用于结构化数据处理

## 关键数据结构

系统使用递归的`PlanSection`模型，支持任意嵌套：
- 每个章节包含标题、详情和可选的子章节
- 计划转换为层次化JSON进行系统性处理
- 内容生成遵循层次结构递归执行

## 文件组织模式

生成的内容按主题组织到目录中：
- 计划保存到 `plan/` 目录
- 结构化JSON保存到 `structured_json/` 目录
- 最终论文保存到主题命名目录（如 `大模型关键技术结点:Tokenizer/`）
- 记忆笔记本保存到 `Notebook4Revise/` 用于修订工作流

## 记忆集成

系统使用以下方式维护持久化记忆：
- SQLite数据库用于结构化存储 (`semantic_memory.db`)
- 向量搜索用于内容相似度匹配
- 基于笔记本的修订历史用于迭代改进

## 开发注意事项

- `main.py`中的主处理循环协调所有智能体
- 每个智能体都设计为上下文管理器使用
- 支持异步处理用于并发章节生成
- 内容生成后自动进行记忆摄取
- 系统内置中文语言支持