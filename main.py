import json

from dotenv import load_dotenv
# 只需要在主入口加载一次环境变量
load_dotenv(dotenv_path="/Users/dengjuan1/build_agent/.env", override=True)
#%%
from langgraph_writer import PlannerGraph
from langgraph_writer import SectionWriterGraph
from langgraph_writer import AiParserGraph
from langgraph_writer import ConnectorWriterGraph
from utils.to_my_file import save_it
from utils.recursion import process_plan_recursively
from langgraph_writer import EditGraph
from memory_chat import ingest_notebook_to_long_term_memory
from utils.concurrent_recursion import process_plan_recursively_async
from pdfs_rag import search_pdf_agent
import asyncio
#%%
async def main():
    # 2. 定义任务
    task = "大模型关键技术结点:Tokenizer"

    print(f"--- Starting essay generation for task: '{task}' ---")

    # 3. 运行并处理流式输出
    with PlannerGraph() as planner:
        final_plan = ""
        for s in planner.run(task=task):
            print("---")
            print(s)
            if 'refine_plan' in s:
                final_plan = s['refine_plan']['plan']

    # print("\n\n--- Final Plan ---")
    # print(final_plan)

    ## 保存
    save_it(final_plan, f"Plan_{task}_chinese.md", "plan")

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

    with open('/Users/dengjuan1/build_agent/parsed_json.json', 'r', encoding='utf-8') as f:
        # 3. 尝试将文件内容解析为 JSON
        final_parser = json.load(f)

    def section_writer_factory():
        return SectionWriterGraph()
    def connector_writer_factory():
        return ConnectorWriterGraph()

    # 从 'content_structure' 开始递归处理
    essay_parts = []
    notebook_entries = []
    i=0
    for top_level_section in final_parser['content_structure']:
        i+=1
        if i > 3:
            continue
        section_content, section_notebook_entries = process_plan_recursively(
            top_level_section,
            final_parser,
            section_writer_factory,
            connector_writer_factory,
            level=0  # 顶层 section 从 H2 (##) 开始
        )
        essay_parts.append(section_content)
        notebook_entries.extend(section_notebook_entries)


    # 并发
    # top_level_tasks = [
    #     process_plan_recursively_async(
    #         top_level_section,
    #         final_parser,
    #         section_writer_factory,
    #         connector_writer_factory,
    #         level=0
    #     )
    #     for top_level_section in final_parser['content_structure']
    # ]
    #
    # print(f"--- Starting ASYNC generation of {len(top_level_tasks)} top-level sections ---")
    #
    # # 2. 使用 asyncio.gather 并发执行所有顶层任务
    # essay_parts = await asyncio.gather(*top_level_tasks)
    #
    # print("\n--- All sections have been processed asynchronously. ---")


    # 保存
    first_level = [section['title'] for section in final_parser['content_structure']]
    for i, part_content in enumerate(essay_parts):
        # 生成一个有意义的文件名。
        # 使用 f-string 格式化，例如：01_part.md, 02_part.md, ...
        # "{i + 1:02d}" 会将数字格式化为两位，不足的前面补0 (例如 1 -> 01, 10 -> 10)
        filename = f"{i + 1:02d}_{first_level[i]}.md"

        save_it(part_content, filename, task)

    print("\n所有部分已成功保存为独立文件！")

    save_it(json.dumps(notebook_entries,indent=4, ensure_ascii=False), f"Notebook:{task}_chinese.json", "Notebook4Revise")

    #回顾和修改

    notebook_path = "Notebook4Revise/Tokenizer_chinese.json"
    file_to_edit_path = "大模型关键技术结点:Tokenizer/01_1. 引言.md"
    notebook_mark="leafSection: '核心问题引出'" #input("给出修改段落在修订记事本上所属的的段落标签：type+location_description")
    original_content = "权威观点指出，Tokenizer是模型预处理中的核心环节，其优化能显著提升模型整体表现。" #input("给出需要修订的内容")
    user_prompt="这句话提到了`权威观点指出`,可是并没有一个具体的引用，如果无法找到确切的引用情况应该避免用`权威观点`这种模糊的代指"#input("给出修订指示")

    initial_state = {
        "notebook_path": notebook_path,
        "file_to_edit_path": file_to_edit_path,
        "notebook_mark": notebook_mark, #type+location_description
        "original_content": original_content,
        "initial_user_prompt": user_prompt,
        "proposed_edit":[],
        "user_feedback":"",
        "revision_history":""
    }
    with EditGraph() as editor:
        for s in editor.run(initial_state):
            print(s)
            print("---")

    notebook_file = "/Users/dengjuan1/build_agent/Notebook4Revise/Tokenizer_chinese.json"
    user = "dengjuan"

    await ingest_notebook_to_long_term_memory(notebook_path=notebook_file, user_id=user)



#%%
#concurrent




