from dotenv import load_dotenv
from anthropic import Anthropic
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
from typing import List, Dict, TypedDict, Any
from contextlib import AsyncExitStack
import json
import asyncio
from openai import OpenAI
from openai.types.chat import ChatCompletionToolParam
from langchain.memory import ConversationBufferMemory
import os
from langchain_community.chat_message_histories import ChatMessageHistory



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
        # 这个字典将为每个用户会话存储一个独立的记忆对象
        # --- 短期记忆对象 ---
        self.short_term_memory: Dict[str, ChatMessageHistory] = {}

    def get_short_term_history(self, session_id: str) -> ChatMessageHistory:
        """获取或创建当前会话的短期聊天历史。"""
        if session_id not in self.short_term_memory:
            self.short_term_memory[session_id] = ChatMessageHistory()
        return self.short_term_memory[session_id]

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

    async def connect_to_servers(self):  # new
        """Connect to all configured MCP servers."""
        try:
            with open("server_config.json", "r") as file:
                data = json.load(file)

            servers = data.get("mcpServers", {})

            for server_name, server_config in servers.items():
                await self.connect_to_server(server_name, server_config)
        except Exception as e:
            print(f"Error loading server configuration: {e}")
            raise

    async def process_query(self, query: str, session_id: str = "default_session"):
        # --- 步骤 A: 获取短期记忆 ---
        short_term_history = self.get_short_term_history(session_id)
        # 为了 RAG，我们可以将短期历史格式化为字符串
        recent_conversation = "\n".join(
            [f"{msg.type}: {msg.content}" for msg in short_term_history.messages[-4:]]  # 只取最近4条消息
        )

        system_prompt = f"""
        You are a helpful and intelligent assistant.
        Use the following recent conversation history to understand the context, especially for resolving pronouns (like 'he', 'it', 'that') and follow-up questions.

        <Recent Conversation History>
        {recent_conversation}
        </Recent Conversation History>

        Now, continue the conversation and respond to the user's latest message. If you need to use a tool, call it.
        """

        # messages: List[Dict[str, Any]] = [{'role': 'user', 'content': query}]
        # 丰富信息类型
        # 2. 构造初始的 messages 列表
        # 注意：我们不再只放 user message，而是先放 system message
        messages: List[Dict[str, Any]] = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': query}
        ]

        while True:
            print("--- Sending request to DeepSeek ---")
            response = self.client.chat.completions.create(
                model="deepseek-chat",  # 使用 DeepSeek 的模型
                messages=messages,
                tools=self.available_tools,
                tool_choice="auto",
            )

            response_message = response.choices[0].message

            # 检查模型是否决定调用工具
            if response_message.tool_calls:
                # 将模型的回复（包含工具调用请求）添加到历史记录中
                messages.append(response_message)

                # 遍历并执行所有工具调用
                for tool_call in response_message.tool_calls:
                    function_name = tool_call.function.name
                    # 注意: OpenAI/DeepSeek 返回的 arguments 是一个 JSON 字符串，需要解析
                    try:
                        function_args = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError:
                        print(f"Error: Could not decode arguments for {function_name}: {tool_call.function.arguments}")
                        continue

                    print(f"Calling tool '{function_name}' with args {function_args}")

                    # 调用 MCP 工具
                    session = self.tool_to_session[function_name]
                    result = await session.call_tool(function_name, arguments=function_args)
                    print(result)

                    # --- 健壮的工具输出处理逻辑 ---
                    tool_output_string = ""
                    if result.isError:
                        # 情况1: MCP协议层面发生错误
                        tool_output_string = json.dumps({
                            "error": "Tool execution failed at the protocol level.",
                            "details": str(result.content)  # content 可能包含错误详情
                        })
                    elif not result.content:
                        # 情况2: 工具成功执行，但没有返回任何内容
                        tool_output_string = json.dumps({"status": "success", "output": "Tool returned no content."})
                    else:
                        # 情况3: 工具成功执行，并返回了内容
                        # 遍历所有 content blocks，将它们转换为字符串
                        output_parts = []
                        for block in result.content:
                            if block.type == 'text':
                                output_parts.append(block.text)
                            elif block.type == 'structured':
                                # 将结构化数据序列化为 JSON 字符串
                                output_parts.append(json.dumps(block.structured))
                            elif block.type == 'error':
                                # 将错误信息也序列化为 JSON 字符串
                                output_parts.append(json.dumps({
                                    "error": block.name,
                                    "message": block.message
                                }))
                            else:
                                # 对于其他类型（如 image），我们可以提供一个简单的描述
                                output_parts.append(f"[Unsupported content block of type: {block.type}]")

                        # 将所有处理过的部分拼接成一个单一的大字符串
                        tool_output_string = "\n".join(output_parts)

                    # 确保最终结果是非空的字符串
                    if not tool_output_string:
                        tool_output_string = "Tool executed but returned empty content."

                    # --- 将处理好的字符串传递给 LLM ---
                    messages.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": tool_output_string,  # 这里现在总是一个合法的字符串
                    })
            else:
                # 没有工具调用，模型给出了最终答案
                final_answer = response_message.content
                print(final_answer)

                # --- 步骤 D: 在对话结束时，更新短期记忆 ---
                # 这里我们只添加真正的 user query 和 ai response
                short_term_history.add_user_message(query)
                short_term_history.add_ai_message(final_answer)
                print("Short-term memory updated.")

                break  # 结束循环

    async def chat_loop(self):
        """
        运行一个交互式聊天循环，现在支持会话。
        """
        print("\nMCP Chatbot (with Memory) Started!")
        session_id = input(
            "Enter a session ID (e.g., 'user123') or press Enter for default: ").strip() or "default_session"
        print(f"Starting chat session: {session_id}. Type 'quit' to exit.")

        while True:
            try:
                query = input(f"\n[{session_id}] Query: ").strip()
                if query.lower() == 'quit':
                    print("Ending session.")
                    break

                # 将 session_id 传递给 process_query
                await self.process_query(query, session_id)

            except Exception as e:
                print(f"\nError: {str(e)}")

    async def cleanup(self):  # new
        """Cleanly close all resources using AsyncExitStack."""
        await self.exit_stack.aclose()


async def main():
    load_dotenv(dotenv_path="/Users/dengjuan1/build_agent/.env")
    chatbot = MCP_ChatBot()
    try:
        await chatbot.connect_to_servers()
        await chatbot.chat_loop()
    finally:
        await chatbot.cleanup()


if __name__ == "__main__":
    asyncio.run(main())