# debug_deepseek.py

import os
import json
import requests
from dotenv import load_dotenv

# 1. 加载 .env 文件
# 我们用一种更健壮的方式来加载，确保能找到它
dotenv_path = "/Users/dengjuan1/build_agent/.env"
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)
    print(f".env file loaded from: {dotenv_path}")
else:
    print(".env file not found!")

# 2. 从环境变量获取凭证
API_KEY = os.getenv("DEEPSEEK_API_KEY")
API_BASE = os.getenv("DEEPSEEK_API_BASE") # 比如 "https://api.deepseek.com/v1"

# 检查变量是否加载成功
if not API_KEY or not API_BASE:
    print("Error: DEEPSEEK_API_KEY or DEEPSEEK_API_BASE is not set.")
    print("Please check your .env file.")
    exit()

# 3. 构造请求
# 根据 DeepSeek 官方文档，embedding 的端点是 /embeddings
# 所以完整的 URL 是 base_url + /embeddings
# 如果你的 API_BASE 已经包含了 /v1, 那么这里就不需要再加了
# 我们先假设 API_BASE 是 "https://api.deepseek.com"
# 那么完整的 URL 就是 "https://api.deepseek.com/v1/embeddings"
#
# 为了确保路径正确，我们自己拼接
# 先移除 API_BASE 末尾可能存在的斜杠和/v1
clean_base_url = API_BASE.replace("/v1", "").rstrip('/')
url = f"{clean_base_url}/v1/embeddings"


headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}",
}

data = {
    "input": "hello world",
    "model": "embedding-2" # DeepSeek 的 embedding 模型名
}

print("--- Sending Request ---")
print(f"URL: {url}")
print(f"Headers: {{'Content-Type': 'application/json', 'Authorization': 'Bearer sk-...'}}")
print(f"Data: {json.dumps(data, indent=2)}")
print("-----------------------")


# 4. 发送请求
try:
    response = requests.post(url, headers=headers, json=data)

    print("\n--- Response ---")
    print(f"Status Code: {response.status_code}")
    print("Response Headers:")
    print(response.headers)
    print("\nResponse Body (JSON):")
    try:
        # 尝试以 JSON 格式打印响应体
        print(json.dumps(response.json(), indent=2))
    except json.JSONDecodeError:
        # 如果响应不是 JSON，直接打印文本
        print(response.text)
    print("----------------")

except requests.exceptions.RequestException as e:
    print(f"\n--- An Error Occurred ---")
    print(f"Request failed: {e}")
    print("-------------------------")