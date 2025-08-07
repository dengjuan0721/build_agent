from dotenv import load_dotenv
from anthropic import Anthropic
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
from typing import List, Dict, TypedDict
from contextlib import AsyncExitStack
import json
import asyncio
from openai import OpenAI
from openai.types.chat import ChatCompletionToolParam
import os

load_dotenv()

class MCP_ChatBot:
    def __init__(self):
        # Initialize session and client objects
        self.sessions: List[ClientSession] = []  # new
        self.exit_stack = AsyncExitStack()  # new
        self.client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com/v1"
        )
        self.available_tools: List[ChatCompletionToolParam] = []  # new
        self.tool_to_session: Dict[str, ClientSession] = {}  # new

    async def connect_to_server(self, server_name: str, server_config: dict) -> None:
        """Connect to a single MCP server."""
        try:
            server_params = StdioServerParameters(**server_config)
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )  # new
            read, write = stdio_transport
            session = await self.exit_stack.enter_async_context(
                ClientSession(read, write)
            )  # new
            await session.initialize()
            self.sessions.append(session)

            # List available tools for this session
            response = await session.list_tools()
            tools = response.tools
            print(f"\nConnected to {server_name} with tools:", [t.name for t in tools])

            for tool in tools:  # new
                self.tool_to_session[tool.name] = session
                # 在 connect_to_server 函数的 for 循环内部
                self.available_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema  # 注意：键名是 'parameters' 而不是 'input_schema'
                    }
                })
        except Exception as e:
            print(f"Failed to connect to {server_name}: {e}")

    # async def connect_to_servers(self):  # new
    #     """Connect to all configured MCP servers."""
    #     try:
    #         with open("server_config.json.json", "r") as file:
    #             data = json.load(file)
    #
    #         servers = data.get("mcpServers", {})
    #
    #         for server_name, server_config.json in servers.items():
    #             await self.connect_to_server(server_name, server_config.json)
    #     except Exception as e:
    #         print(f"Error loading server configuration: {e}")
    #         raise

    async def connect_to_servers(self):
        """asyncio.gather非阻塞式连接"""
        try:
            with open("server_config.json", "r") as file:
                data = json.load(file)

            servers = data.get("mcpServers", {})

            # 1. 创建一个所有连接任务的列表
            connection_tasks = []
            for server_name, server_config in servers.items():
                task = self.connect_to_server(server_name, server_config)
                connection_tasks.append(task)

            # 2. 使用 asyncio.gather 来并行运行所有任务
            #    并等待它们全部完成
            print("Connecting to all servers in parallel...")
            await asyncio.gather(*connection_tasks)
            print("All servers connected.")

        except Exception as e:
            print(f"Error during parallel server connection: {e}")
            raise

    async def process_query(self, query):
        messages = [{'role': 'user', 'content': query}]

        while True:
            # 1. 调用 DeepSeek API
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                tools=self.available_tools,
                tool_choice="auto",
            )

            response_message = response.choices[0].message
            tool_calls = response_message.tool_calls

            # 2. 检查 LLM 是否要调用工具
            if tool_calls:
                messages.append(response_message)  # 添加助手的回复（包含工具调用请求）

                for tool_call in tool_calls:
                    tool_name = tool_call.function.name
                    tool_args_str = tool_call.function.arguments
                    tool_args = json.loads(tool_args_str)
                    tool_id = tool_call.id

                    print(f"Calling tool {tool_name} with args {tool_args}")

                    # 3. 核心路由逻辑 (这部分是新架构的精髓，需要保留)
                    session = self.tool_to_session[tool_name]
                    result = await session.call_tool(tool_name, arguments=tool_args)

                    # 将工具结果转换为字符串
                    tool_output_str = str(result.content)

                    # 4. 以 OpenAI/DeepSeek 的格式添加工具结果
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_id,
                        "name": tool_name,
                        "content": tool_output_str,
                    })

                # 继续循环，让 LLM 根据工具结果生成最终回复
                continue

            # 5. 如果没有工具调用，直接打印回复并结束
            else:
                final_response = response_message.content
                # 清洗一下可能的非法字符
                if final_response:
                    cleaned_response = final_response.encode('utf-8', 'surrogateescape').decode('utf-8', 'replace')
                    print(cleaned_response)
                break

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Chatbot Started!")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()

                if query.lower() == 'quit':
                    break

                await self.process_query(query)
                print("\n")

            except Exception as e:
                print(f"\nError: {str(e)}")

    async def cleanup(self):  # new
        """Cleanly close all resources using AsyncExitStack."""
        await self.exit_stack.aclose()


async def main():

    chatbot = MCP_ChatBot()
    try:
        await chatbot.connect_to_servers()
        await chatbot.chat_loop()
    except (asyncio.CancelledError, KeyboardInterrupt):
        # 捕获取消信号，这是一种好的实践
        print("\nChatbot interrupted. Cleaning up...")
    finally:
        print("Exiting chat loop. Starting cleanup...")
        # 在 cleanup 之前加一个短暂的 sleep(0)
        # 这会让出控制权，让事件循环处理任何挂起的任务状态变更
        await asyncio.sleep(0)
        await chatbot.cleanup()
        print("Cleanup finished.")


if __name__ == "__main__":
    asyncio.run(main())