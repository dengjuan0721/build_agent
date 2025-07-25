import json

from dotenv import load_dotenv
from openai import OpenAI
import os
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
from typing import List
import asyncio
import nest_asyncio

nest_asyncio.apply()

load_dotenv()

#%% MCP的聊天机器人类
class MCP_ChatBot:

    def __init__(self):
        # 2. 初始化 OpenAI 客户端，指向 DeepSeek 的 API
        self.session: ClientSession = None
        self.client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com/v1"
        )
        self.available_tools: List[dict] = []

    async def process_query(self, query):
        # 3. 这是核心逻辑修改部分
        messages = [{'role': 'user', 'content': query}]

        while True:
            response = self.client.chat.completions.create(
                model="deepseek-chat",  # 使用 DeepSeek 的模型
                messages=messages,
                tools=self.available_tools,
                tool_choice="auto",
            )

            response_message = response.choices[0].message
            tool_calls = response_message.tool_calls

            # 情况一: LLM 决定调用工具
            if tool_calls:
                # 将助手的工具调用决策添加到消息历史中
                messages.append(response_message)

                for tool_call in tool_calls:
                    tool_name = tool_call.function.name
                    tool_args_str = tool_call.function.arguments
                    tool_args = json.loads(tool_args_str)  # DeepSeek/OpenAI 参数是字符串，需解析
                    tool_id = tool_call.id

                    print(f"Calling tool {tool_name} with args {tool_args}")

                    # 通过 MCP session 调用工具
                    result = await self.session.call_tool(tool_name, arguments=tool_args)

                    # 确保工具调用的结果是字符串格式
                    # 如果 result.content 不是字符串，就用 json.dumps 将其序列化为字符串
                    tool_output = result.content
                    print(tool_output)
                    tool_output_str = str(result.content)

                    # 将工具执行结果添加回消息历史
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_id,
                        "name": tool_name,
                        "content": tool_output_str,  # <--- 使用转换后的字符串
                    })

                # 工具调用后，再次请求 LLM 生成最终回复（循环继续）
                continue

            # 情况二: LLM 直接返回文本回复，循环结束
            else:
                final_response = response_message.content
                print(final_response)
                break  # 结束循环

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

    # 这部分代码无需修改
    async def connect_to_server_and_run(self):
        # Create server parameters for stdio connection
        server_params = StdioServerParameters(
            command="uv",  # Executable
            args=["run", "research_server.py"],  # Optional command line arguments
            env=None,  # Optional environment variables
        )
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                self.session = session
                # Initialize the connection
                await session.initialize()

                # List available tools
                response = await session.list_tools()

                tools = response.tools
                print("\nConnected to server with tools:", [tool.name for tool in tools])

                # 格式化工具以适应 OpenAI/DeepSeek 的 API
                self.available_tools = [{
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema
                    }
                } for tool in response.tools]

                await self.chat_loop()


async def main():
    chatbot = MCP_ChatBot()
    await chatbot.connect_to_server_and_run()


if __name__ == "__main__":
    asyncio.run(main())