import json

from dotenv import load_dotenv
# 只需要在主入口加载一次环境变量
load_dotenv(dotenv_path="/Users/dengjuan1/build_agent/.env")
#%%
from langgraph_writer import PlannerGraph
from langgraph_writer import SectionWriterGraph
from langgraph_writer import AiParserGraph
from langgraph_writer import ConnectorWriterGraph
from utils.to_my_file import save_it
from utils.recursion import process_plan_recursively
def main():
    # 2. 定义任务
    task = "Encoder-Only: BERT"

    print(f"--- Starting essay generation for task: '{task}' ---")

    # 3. 运行并处理流式输出
    with PlannerGraph() as planner:
        final_plan = ""
        for s in planner.run(task=task):
            print("---")
            print(s)
            if 'refine_plan' in s:
                final_plan = s['refine_plan']['plan']

    print("\n\n--- Final Plan ---")
    print(final_plan)

    ## 保存
    save_it(final_plan, f"Plan:{task}_chinese.md", "plan")

    with AiParserGraph() as parser:
        final_parser = ""
        for s in parser.run(unstructured_plan=final_plan):
            print("---")
            print(s)
            if ('generate_initial' in s):
                final_parser = s['generate_initial']['parsed_json']
            elif ('revise_parse' in s):
                final_parser = s['revise_parse']['parsed_json']

    if not final_parser:
        raise ValueError("AI Parser failed to produce a structured plan.")

    print("--- Successfully parsed into a structured object. ---")

    save_it(json.dumps(final_parser,indent=4, ensure_ascii=False), f"Content:{task}_chinese.json", "structured_json")


    def section_writer_factory():
        return SectionWriterGraph()
    def connector_writer_factory():
        return ConnectorWriterGraph()

    # 从 'content_structure' 开始递归处理
    essay_parts = []
    # i=0
    for top_level_section in final_parser['content_structure']:
        # i+=1
        # if i == 1:
        #     continue
        section_content = process_plan_recursively(
            top_level_section,
            final_parser,
            section_writer_factory,
            connector_writer_factory,
            level=0  # 顶层 section 从 H2 (##) 开始
        )
        essay_parts.append(section_content)

    first_level = [top_level_section in final_parser['content_structure']]
    for i, part_content in enumerate(essay_parts):
        # 生成一个有意义的文件名。
        # 使用 f-string 格式化，例如：01_part.md, 02_part.md, ...
        # "{i + 1:02d}" 会将数字格式化为两位，不足的前面补0 (例如 1 -> 01, 10 -> 10)
        filename = f"{i + 1:02d}_{first_level[i]}.md"

        save_it(part_content,filename,task)

    print("\n所有部分已成功保存为独立文件！")













#%%
file_path="/Users/dengjuan1/build_agent/plan/Plan/Encoder-Only/ BERT.md"
with open(file_path, 'r', encoding='utf-8') as f:
    final_plan_str = f.read()

from langgraph_writer import AiParserGraph
def main():
    with AiParserGraph() as parser:
        final_parser = ""
        for s in parser.run(unstructured_plan=final_plan_str):
            print("---")
            print(s)
            if ('generate_initial' in s):
                # 当图结束时，'__end__' 键对应的值就是最终的状态
                final_parser = s['generate_initial']['parsed_json']
            elif ('revise_parse' in s):
                # 当图结束时，'__end__' 键对应的值就是最终的状态
                final_parser = s['revise_parse']['parsed_json']

    if not final_parser:
        raise ValueError("AI Parser failed to produce a structured plan.")

    print("--- Plan successfully parsed into a structured object. ---")

#%%
from utils.recursion import process_plan_recursively
from utils.cleaner import create_writing_context
import ast
from langgraph_writer import SectionWriterGraph
from langgraph_writer import ConnectorWriterGraph
file_path="/Users/dengjuan1/build_agent/structured_json/Encoder-Only BERT.json"
with open(file_path, 'r', encoding='utf-8') as f:
    plan_string_representation = f.read()
    structured_plan = json.loads(plan_string_representation)
    # 2. 使用 ast.literal_eval() 来安全地解析这个字符串
    # structured_plan = ast.literal_eval(plan_string_representation)

def main():
    # 创建 SectionWriter 的工厂函数，以便在循环中可以创建新的实例
    def section_writer_factory():
        return SectionWriterGraph()
    def connector_writer_factory():
        return ConnectorWriterGraph()

    # 从 'content_structure' 开始递归处理
    essay_parts = []
    # i=0
    for top_level_section in structured_plan['content_structure']:
        # i+=1
        # if i == 1:
        #     continue

        section_content = process_plan_recursively(
            top_level_section,
            structured_plan,
            section_writer_factory,
            connector_writer_factory,
            level=1  # 顶层 section 从 H2 (##) 开始
        )
        essay_parts.append(section_content)


#%%
import asyncio
if __name__ == "__main__":
    # asyncio.run(main())
    main()
#%%
full_essay_content = "\n\n".join(essay_parts)

# --- 步骤3: 将完整内容写入新的 Markdown 文件 ---
# 定义你想要保存的文件名
import os,sys
output_directory = "transformer"

# --- 步骤3: 创建目录（如果它不存在） ---
try:
    # os.makedirs() 可以创建多层目录
    # exist_ok=True 表示如果目录已经存在，不要抛出错误
    os.makedirs(output_directory, exist_ok=True)
    print(f"目录 '{output_directory}' 已创建或已存在。")
except OSError as e:
    # 如果因为权限等问题创建失败，打印错误并退出
    print(f"错误：创建目录 '{output_directory}' 失败。原因: {e}")
    sys.exit(1) # 退出脚本

for i, part_content in enumerate(essay_parts):
    # 生成一个有意义的文件名。
    # 使用 f-string 格式化，例如：01_part.md, 02_part.md, ...
    # "{i + 1:02d}" 会将数字格式化为两位，不足的前面补0 (例如 1 -> 01, 10 -> 10)
    filename = f"{i + 1:02d}_part.md"

    # 使用 os.path.join() 来安全地拼接目录和文件名，它会自动处理斜杠/反斜杠
    file_path = os.path.join(output_directory, filename)

    try:
        # 使用 'with open' 安全地写入每个文件
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(part_content)

        print(f"  - 成功保存: {file_path}")

    except IOError as e:
        print(f"  - 错误：无法写入文件 {file_path}。原因: {e}")

print("\n所有部分已成功保存为独立文件！")