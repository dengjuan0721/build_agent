import arxiv
import json
import os
from typing import List
from dotenv import load_dotenv

#%% Tool Functions
PAPER_DIR = "../papers"


def search_papers(topic: str, max_results: int = 5) -> List[str]:
    """
    Search for papers on arXiv based on a topic and store their information.

    Args:
        topic: The topic to search for
        max_results: Maximum number of results to retrieve (default: 5)

    Returns:
        List of paper IDs found in the search
    """

    # Use arxiv to find the papers
    client = arxiv.Client()

    # Search for the most relevant articles matching the queried topic
    search = arxiv.Search(
        query=topic,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )

    papers = client.results(search)

    # Create directory for this topic
    path = os.path.join(PAPER_DIR, topic.lower().replace(" ", "_"))
    os.makedirs(path, exist_ok=True)

    file_path = os.path.join(path, "papers_info.json")

    # Try to load existing papers info
    try:
        with open(file_path, "r") as json_file:
            papers_info = json.load(json_file)
    except (FileNotFoundError, json.JSONDecodeError):
        papers_info = {}

    # Process each paper and add to papers_info
    paper_ids = []
    for paper in papers:
        paper_ids.append(paper.get_short_id())
        paper_info = {
            'title': paper.title,
            'authors': [author.name for author in paper.authors],
            'summary': paper.summary,
            'pdf_url': paper.pdf_url,
            'published': str(paper.published.date())
        }
        papers_info[paper.get_short_id()] = paper_info

    # Save updated papers_info to json file
    with open(file_path, "w") as json_file:
        json.dump(papers_info, json_file, indent=2)

    print(f"Results are saved in: {file_path}")

    return paper_ids

# search_papers("database")


def extract_info(paper_id: str) -> str:
    """
    Search for information about a specific paper across all topic directories.

    Args:
        paper_id: The ID of the paper to look for

    Returns:
        JSON string with paper information if found, error message if not found
    """

    for item in os.listdir(PAPER_DIR):
        item_path = os.path.join(PAPER_DIR, item)
        if os.path.isdir(item_path):
            file_path = os.path.join(item_path, "papers_info.json")
            if os.path.isfile(file_path):
                try:
                    with open(file_path, "r") as json_file:
                        papers_info = json.load(json_file)
                        if paper_id in papers_info:
                            return json.dumps(papers_info[paper_id], indent=2)
                except (FileNotFoundError, json.JSONDecodeError) as e:
                    print(f"Error reading {file_path}: {str(e)}")
                    continue

    return f"There's no saved information related to paper {paper_id}."
#%% Tool Schema
tools = [
    {
        "name": "search_papers",
        "description": "Search for papers on arXiv based on a topic and store their information.",
        "input_schema": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "The topic to search for"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to retrieve",
                    "default": 5
                }
            },
            "required": ["topic"]
        }
    },
    {
        "name": "extract_info",
        "description": "Search for information about a specific paper across all topic directories.",
        "input_schema": {
            "type": "object",
            "properties": {
                "paper_id": {
                    "type": "string",
                    "description": "The ID of the paper to look for"
                }
            },
            "required": ["paper_id"]
        }
    }
]
#%% tool Mapping
mapping_tool_function = {
    "search_papers": search_papers,
    "extract_info": extract_info
}


def execute_tool(tool_name, tool_args_str):
    tool_args = json.loads(tool_args_str)
    result = mapping_tool_function[tool_name](**tool_args)

    if result is None:
        result = "The operation completed but didn't return any results."

    elif isinstance(result, list):
        result = ', '.join(result)

    elif isinstance(result, dict):
        # Convert dictionaries to formatted JSON strings
        result = json.dumps(result, indent=2)

    else:
        # For any other type, convert using str()
        result = str(result)
    return result

#%% 将这两个工具告诉chatbot
import google.generativeai as genai

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

tools = [search_papers, extract_info]

# Gemini通过一个生成模型对象来发起请求
model = genai.GenerativeModel('gemini-2.5-pro', tools=tools)

#%% 调用函数
def process_query(query: str):
    """
    使用最新版google-generativeai (v0.8.x) 处理查询并执行工具调用。
    """
    # 1. 启动一个聊天会话
    chat = model.start_chat()

    # 2. 发送第一条用户消息
    print(f"User Query: {query}")
    response = chat.send_message(query)

    while True:
        # 3. 从最新回复中获取内容
        latest_part = response.candidates[0].content.parts[0]

        # 4. 检查是否有函数调用请求
        if latest_part.function_call:
            function_call = latest_part.function_call
            tool_name = function_call.name
            tool_args = dict(function_call.args)

            print(f"🤖 Gemini wants to call tool: {tool_name} with args: {tool_args}")

            # 5. 执行工具
            result = execute_tool(tool_name, tool_args)
            print(f"Tool Result: {result}")

            # 6. 【关键变化】将工具结果直接以字典形式返回给模型
            #    不再需要手动创建 Part 对象！
            #    库会自动将其包装成正确的格式。
            response = chat.send_message(
                {
                    "function_response": {
                        "name": tool_name,
                        "response": {
                            "content": result
                        }
                    }
                }
            )
            # 循环继续，等待模型基于工具结果生成最终回复
        else:
            # 7. 如果没有函数调用，说明是最终的文本答案
            final_answer = latest_part.text
            print(f"🤖 Gemini's Final Answer: {final_answer}")
            break  # 结束循环

def chat_loop():
    print("Type your queries or 'quit' to exit.")
    while True:
        try:
            query = input("\nQuery: ").strip()
            if query.lower() == 'quit':
                break

            process_query(query)
            print("\n")
        except Exception as e:
            print(f"\nError: {str(e)}")

#%% 现在我们想通过deepseek来回答问题，初始化客户端
import os
from openai import OpenAI # <--- 使用openai库
from dotenv import load_dotenv

load_dotenv()

# --- 关键的初始化修改 ---
client_deepseek = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)
#%% tools deepseek
tools_openai = [
    {
        "type": "function",  # 1. 添加 'type' 字段
        "function": {      # 2. 将所有函数信息包裹在 'function' 对象中
            "name": "search_papers",
            "description": "Search for papers on arXiv based on a topic and store their information.",
            "parameters": {  # 3. 将 'input_schema' 重命名为 'parameters'
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "The topic to search for"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to retrieve",
                        "default": 5
                    }
                },
                "required": ["topic"]
            }
        }
    },
    {
        "type": "function",  # 1. 添加 'type' 字段
        "function": {      # 2. 将所有函数信息包裹在 'function' 对象中
            "name": "extract_info",
            "description": "Search for information about a specific paper across all topic directories.",
            "parameters": {  # 3. 将 'input_schema' 重命名为 'parameters'
                "type": "object",
                "properties": {
                    "paper_id": {
                        "type": "string",
                        "description": "The ID of the paper to look for"
                    }
                },
                "required": ["paper_id"]
            }
        }
    }
]
#%% deepseek
def process_query_deepseek(query: str):

    # 2. 发送第一条用户消息
    messages = [{"role": "user", "content": query}]
    print(f"User Query: {query}")

    while True:
        # 2. 调用DeepSeek的API
        response = client_deepseek.chat.completions.create(
            model="deepseek-chat",  # 使用DeepSeek的聊天模型
            messages=messages,
            tools=tools_openai,
            tool_choice="auto"
        )

        assistant_message = response.choices[0].message
        messages.append(assistant_message)  # 将助手的回复加入历史

        # 4. 检查是否有函数调用请求
        if assistant_message.tool_calls:
            tool_call = assistant_message.tool_calls[0]
            tool_name = tool_call.function.name
            tool_args_str = tool_call.function.arguments

            print(f"🤖 DeepSeek wants to call tool: {tool_name} with args: {tool_args_str}")

            # 5. 执行工具
            result = execute_tool(tool_name, tool_args_str)
            print(f"Tool Result: {result}")

            # 6. 将工具结果返回给模型 (格式也与OpenAI兼容)
            messages.append({
                "role": "tool",
                "content": result,
                "tool_call_id": tool_call.id
            })
            # 循环继续，等待模型基于工具结果生成最终回复
        else:
            # 7. 如果没有函数调用，说明是最终的文本答案
            final_answer = assistant_message.content
            print(f"🤖 DeepSeek's Final Answer: {final_answer}")
            break  # 结束循环
def chat_loop():
    print("Type your queries or 'quit' to exit.")
    while True:
        try:
            query = input("\nQuery: ").strip()
            if query.lower() == 'quit':
                break

            process_query_deepseek(query)
            print("\n")
        except Exception as e:
            print(f"\nError: {str(e)}")

