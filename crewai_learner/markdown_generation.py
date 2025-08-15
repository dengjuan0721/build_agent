import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool, ScrapeWebsiteTool, DirectoryReadTool

# 自定义一个文件写入工具
# from langchain.tools import tool
#
#
# class FileWriterTool:
#     @tool("Write File Tool")
#     def write_file(filename: str, content: str) -> str:
#         """
#         A tool that can be used to write a file to the local directory.
#         Args:
#             filename (str): The name of the file to write, including extension.
#             content (str): The content to write to the file.
#         Returns:
#             str: A confirmation message indicating the file was written.
#         """
#         # 确保输出目录存在
#         output_path = os.path.join("output", filename)
#         os.makedirs(os.path.dirname(output_path), exist_ok=True)
#
#         with open(output_path, "w", encoding="utf-8") as f:
#             f.write(content)
#         return f"Successfully wrote content to {output_path}."
#

# 1. 导入官方的 FileWriterTool
from crewai_tools import FileWriterTool

# 2. 实例化一个通用的写入工具
#    我们不需要在此时指定任何参数
writer_tool = FileWriterTool()

# 加载环境变量
_ = load_dotenv(dotenv_path="/Users/dengjuan1/agent_crew/.env")

#%%
# --- 1. 定义 LLM (保持不变) ---
from langchain_deepseek import ChatDeepSeek
from langchain_community.llms.moonshot import Moonshot

llm = ChatDeepSeek(
    model="deepseek-chat",
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
    openai_api_base=os.getenv("DEEPSEEK_API_BASE"),
    temperature=0.0
)

coder_llm = Moonshot(
    api_key=os.getenv("MOONSHOT_API_KEY"),
    base_url="https://api.moonshot.cn/v1",
    model= "kimi-k2-0711-preview"
)

#%%
# --- 2. 实例化工具 ---
search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()
# 读取范式文件的工具
paradigm_reader_tool = DirectoryReadTool(directory='./instructions')
# 写入最终产出的工具
# file_writer_tool = FileWriterTool().write_file

# 尝试从它的具体模块路径导入
from crewai_tools import ArxivPaperTool

# 1. 实例化专用的 arXiv 工具
arxiv_tool = ArxivPaperTool()

# 2. 将通用研究员升级为配备了专用工具的专家
arxiv_specialist = Agent(
    role='Expert Academic Researcher for arXiv',
    goal=(
        "Utilize the dedicated arXiv tool to conduct a deep dive into a given scientific topic. "
        "Extract key information about the most influential papers, including their abstracts, authors, and publication dates. "
        "Compile a comprehensive literature review summary, highlighting the main research questions and findings."
    ),
    backstory=(
        "You are an expert academic researcher specializing in literature reviews using the arXiv database. "
        "You are highly proficient with the specialized arXiv search tool, enabling you to go beyond simple keyword searches. "
        "You can pinpoint seminal works and trace the evolution of ideas within your field. "
        "Your mission is to provide a solid, evidence-based foundation for new research projects by summarizing the current state-of-the-art."
    ),
    # 关键：使用专用工具！
    tools=[arxiv_tool],
    llm=llm,
    verbose=True,
    allow_delegation=False, # 通常专家 Agent 不需要委派
)

#%%
# --- 3. 定义你的 Agent Crew ---

# Agent 1: 总主题大纲撰写人
chief_outliner = Agent(
    role="Paradigm-aware Chief Outliner",
    goal=("From the provided 'Anthropic's Prompt Engineering Documentation Paradigm' document, you must EXTRACT and APPLY ONLY the core principles related to "
        "structural decomposition and information hierarchy. Your primary objective is to use these extracted principles to break down "
        "a large, complex topic into a logical, hierarchical series of smaller sub-topics. "
        "The final output must be a structured list of these sub-topics, each paired with a suggested, descriptive filename."
          ),
    backstory=(
        "You are a master strategist and information architect. You have deeply internalized the 'Anthropic's Prompt Engineering Documentation Paradigm'. "
        "Your job is to create the master plan for a documentation suite, ensuring every piece aligns with the grand vision. "
        "You don't write the content, you design the skeleton that others will build upon."
    ),
    tools=[paradigm_reader_tool],
    llm=llm,
    verbose=True,
)


# Agent 3: 子主题大纲撰写人
sub_topic_outliner = Agent(
    role='Detailed Sub-topic Outliner',
    goal='Create a detailed, well-structured outline for a single sub-topic, based on provided research.',
    backstory=(
        "You are a meticulous planner. You take a focused topic and a research brief, and you craft a perfect, "
        "step-by-step outline that a writer can follow to create a complete article. Your outlines are clear, logical, and comprehensive."
    ),
    llm=llm,
    verbose=True,
)
# Coder Agent 定义
coder = Agent(
    role='Python Code Integration Specialist',
    goal=(
        "Review a given article draft, identify sections where a code example would clarify the concept, "
        "and insert a clear, correct, and well-commented Python code snippet in that location."
    ),
    backstory=(
        "You are an expert Python developer with a talent for teaching. You are not a writer, but a 'code illustrator'. "
        "You receive text written by others and your job is to enhance it with practical code examples. "
        "You look for placeholders like '[code example here]' or phrases like 'for example, the code would look like this...' "
        "You can also add code area as you like"
        "Your code is always clean, follows best practices, and directly demonstrates the point made in the surrounding text."
    ),
    llm=coder_llm, # 或者你可以为 coder 指定一个更擅长代码的 LLM，比如 coder_llm
    verbose=True,
    allow_delegation=False,
)

# Agent 4: Coder / 内容撰写人
content_writer = Agent(
    role='Technical Content Writer',
    goal='Write an insightful and factually accurate article on {topic}',
    backstory=(
        "You are a skilled technical writer who can break down complex topics into easy-to-understand prose. "
        "You work from an outline provided by a strategist. You focus on clear explanations. "
        "When a concept requires a code example, you will insert a clear placeholder like '[INSERT CODE EXAMPLE FOR: concept name]' "
        "for the Coder to fill in later."
    ),
    llm=llm,
    verbose=True,
)

#%%
# --- 4. 定义动态创建的任务 ---

# 我们可以定义一个函数来为每个子主题创建任务流程
def create_writing_tasks_for_subtopic(sub_topic_info):
    """
    为单个子主题创建研究、大纲和写作任务。
    sub_topic_info 是一个字典，例如: {'sub_topic': '...', 'filename': '...'}
    """
    sub_topic = sub_topic_info['sub_topic']
    filename = sub_topic_info['filename']

    # 子主题的研究任务
    task_research_sub = Task(
        description=f"Conduct thorough research on the sub-topic: '{sub_topic}'. Gather all key information, concepts, and examples.",
        expected_output=f"A comprehensive research report on '{sub_topic}', including links to sources.",
        agent=arxiv_specialist,
        async_execution=True  # 可以异步执行以提高效率
    )

    # 子主题的大纲撰写任务
    task_outline_sub = Task(
        description=f"Based on the research for '{sub_topic}', create a detailed, structured outline for the article.",
        expected_output=f"A complete Markdown-formatted outline for an article about '{sub_topic}'.",
        agent=sub_topic_outliner,
        context=[task_research_sub]  # 依赖于研究任务
    )

    # 子主题的最终写作任务
    task_write_sub = Task(
        description=f"Using the detailed outline for '{sub_topic}', write a complete, high-quality, chinese article in Markdown. The final filename must be '{filename}'.",
        expected_output=f"A confirmation message that the file '{filename}' has been successfully written to the 'output' directory.",
        agent=content_writer,
        context=[task_outline_sub],  # 依赖于大纲任务
        tools=[writer_tool]
    )

    return [task_research_sub, task_outline_sub, task_write_sub]

#%%

# --- 5. 组建并启动 Crew ---

# 假设用户输入的主题
main_topic = "How to write an Neural Language Model RNN tutorial for beginners whose career is data engineer"

# 任务 0.1: 读取范式文件 (隐式地由 Agent 在 Task 1 中使用)
# 任务 0.2: 对大主题进行初步研究
task_initial_research = Task(
    description=f"Conduct initial high-level research on the main topic: '{main_topic}'. Identify key areas, common structures, and important concepts.",
    expected_output="A summary report of the initial findings for the main topic.",
    agent=arxiv_specialist
)

# 任务 1: 创建总大纲，分解子主题
task_create_main_outline = Task(
    description=(
        f"Read the writing paradigm from './instructions/anthropic_style.md'.\n"
        f"Based on this paradigm and the initial research on '{main_topic}', break down the main topic into a series of sub-topics. "
        "For each sub-topic, provide a clear title and a suitable filename (e.g., '01_introduction.md').\n"
        "Your final output MUST be a JSON list of dictionaries. Each dictionary should have 'sub_topic' and 'filename' keys. "
        "Example: [{'sub_topic': 'Introduction to Python', 'filename': '01_introduction.md'}, {'sub_topic': 'Variables and Data Types', 'filename': '02_variables.md'}]"
    ),
    expected_output="A JSON formatted string representing a list of sub-topics and their filenames.",
    agent=chief_outliner,
    context=[task_initial_research]
)

# 初始的 Crew，只为了生成子主题列表
planning_crew = Crew(
    agents=[arxiv_specialist, chief_outliner],
    tasks=[task_initial_research, task_create_main_outline],
    process=Process.sequential,
    verbose=True,
)

#%%
print("🚀 Starting the Planning Crew to generate sub-topics...")
sub_topics_json = planning_crew.kickoff()
print("✅ Planning Crew finished.")
print(f"Generated sub-topics plan:\n{sub_topics_json}")

#%%
# 解析 JSON 输出
import json

try:
    sub_topics_list = json.loads(sub_topics_json.raw)
except json.JSONDecodeError:
    print("Error: The output from the planning crew was not valid JSON. Cannot proceed.")
    sub_topics_list = []
#%%
# --- 现在，为每个子主题动态创建并执行写作任务 ---
if sub_topics_list:
    all_writing_tasks = []
    for sub_topic_info in sub_topics_list:
        tasks = create_writing_tasks_for_subtopic(sub_topic_info)
        all_writing_tasks.extend(tasks)

    # 组建最终的写作 Crew
    writing_crew = Crew(
        agents=[arxiv_specialist, sub_topic_outliner, content_writer],
        tasks=all_writing_tasks,
        # process=Process.hierarchical,  # 使用层级模式让 manager agent 协调
        verbose=True,
    )

    print("\n🚀 Starting the Writing Crew to create articles for each sub-topic...")
    writing_result = writing_crew.kickoff()
    print("\n✅ Writing Crew finished its work!")
    print(f"Final result from writing crew:\n{writing_result}")