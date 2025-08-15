import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool, ScrapeWebsiteTool
from langchain_deepseek import ChatDeepSeek
from langchain_openai import OpenAIEmbeddings

# 加载环境变量
_ = load_dotenv(dotenv_path="/Users/dengjuan1/agent_crew/.env")
#%%

# --- 1. 定义你的 LLM ---
# 使用你提供的 DeepSeek 模型配置
# 注意：CrewAI v0.28.8+ 与 LangChain 的集成方式有变化
# 我们使用 LangChain 的 ChatOpenAI 类来包装它
llm = ChatDeepSeek(
    model="deepseek-chat",
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
    openai_api_base=os.getenv("DEEPSEEK_API_BASE"),
    temperature=0.1 # 对于分析任务，使用较低的温度以确保事实性
)
if hasattr(llm, 'openai_api_key') and hasattr(llm.openai_api_key, 'get_secret_value'):
    llm.openai_api_key = llm.openai_api_key.get_secret_value()

# --- 1.1. 定义你的 Embedding 模型 (这是新增的关键部分) ---
# 新增：定义 embedder 的配置字典
embedder_config = {
    "provider": "openai",  # 因为 DashScope 是 OpenAI 兼容的，所以 provider 用 'openai'
    "config": {
        "model": "text-embedding-v4",
        "api_key": os.getenv("DASHSCOPE_API_KEY"),
        "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    }
}

#%%
# --- 2. 实例化工具 ---
# 搜索工具，用于寻找外部框架
search_tool = SerperDevTool()
# 网页抓取工具，用于读取URL内容
scrape_tool = ScrapeWebsiteTool()

#%%
# --- 3. 定义你的 Agent Crew ---

# Agent 1: my_url内容描述人
url_content_summarizer = Agent(
    role='URL Content Summarizer',
    goal='Accurately summarize the key points, structure, tone, and examples of a single given URL about prompt engineering.',
    backstory=(
        "You are a meticulous content analyst. Your specialty is distilling the essence of a document, "
        "focusing on how it's structured, the core message it conveys, the tone it uses (e.g., academic, practical, formal), "
        "and the types of examples it provides. You output a clear, structured summary for each URL you are given."
    ),
    tools=[scrape_tool],
    llm=llm,
    verbose=True,
    allow_delegation=False
)

# Agent 2: 多my_url间关系总结人
anthropic_pattern_synthesizer = Agent(
    role='Anthropic Writing Pattern Synthesizer',
    goal='Identify and synthesize the common writing patterns, structure, and rhetorical strategies across multiple summaries of Anthropic\'s prompt engineering guides.',
    backstory=(
        "You are an expert in meta-analysis and synthesis. You receive multiple summaries of documents from the same source (Anthropic) "
        "and your talent is to see the forest for the trees. You identify recurring structural elements (e.g., 'Principle -> Example -> Anti-pattern'), "
        "consistent terminology, a shared authorial voice, and the overall didactic framework they use to teach."
    ),
    llm=llm,
    verbose=True,
    allow_delegation=False
)

# Agent 3: 类似内容的其他url的范式/框架要点总结人
external_framework_researcher = Agent(
    role='External Prompting Framework Researcher',
    goal='Find and summarize other popular prompt engineering frameworks (e.g., from OpenAI, Cohere, or community-driven ones like CO-STAR or APE) to provide a basis for comparison.',
    backstory=(
        "You are a savvy market and academic researcher. You know how to quickly find the canonical resources for any given technical topic. "
        "Your goal is to provide a clear, concise summary of 2-3 alternative prompt engineering frameworks, highlighting their core principles and structure."
    ),
    tools=[search_tool, scrape_tool],
    llm=llm,
    verbose=True,
    allow_delegation=False
)

# Agent 4: 文档撰写范式基础理论研究者
paradigm_theorist = Agent(
    role='Documentation Paradigm Theorist',
    goal=(
        "Provide a structured analytical framework based on the core theories of "
        "Technical Communication and Information Architecture. This framework will serve "
        "as the intellectual toolkit for the final analysis of the writing paradigm."
    ),
    backstory=(
        "You are a distinguished academic in Technical Communication and Information Architecture, "
        "deeply familiar with the works of Hackos, Morville, Rosenfeld, and Carroll. "
        "Your role is not to analyze the specific Anthropic content, but to provide the timeless, "
        "fundamental principles and structured vocabulary needed to deconstruct and evaluate any "
        "documentation paradigm. Your output empowers other agents to move from simple observation "
        "to a profound, theory-backed analysis."
    ),
    llm=llm,
    verbose=True,
    allow_delegation=False
)

# Agent 5: 对比人
comparative_analyst = Agent(
    role='Comparative Analyst for Writing Paradigms',
    goal='Compare and contrast the identified Anthropic-specific patterns with the external prompting frameworks. Highlight what makes the Anthropic style unique.',
    backstory=(
        "You are a critical thinker with a keen eye for nuance. You receive an analysis of the 'Anthropic way' and an analysis of 'other ways'. "
        "Your job is to create a structured comparison, pointing out similarities, differences, and unique value propositions of the Anthropic style. For example: 'While OpenAI focuses on X, Anthropic emphasizes Y'."
    ),
    llm=llm,
    verbose=True,
    allow_delegation=False
)

# Agent 6: 范式/框架要点最终生成人
final_outline_generator = Agent(
    role='Final Paradigm Outline Generator',
    goal='Create the final, structured outline of the "Anthropic Prompt Engineering Writing Paradigm" based on all prior analyses.',
    backstory=(
        "You are the lead editor and strategist. Your job is to take all the pieces of research—the internal patterns, the external comparisons, and the theoretical framework—and forge them into a single, coherent, and actionable document. "
        "The output should be a well-structured markdown document that clearly explains the Anthropic paradigm, making it easy for others to learn and adopt."
    ),
    llm=llm,
    verbose=True,
    allow_delegation=False
)

#%%
# --- 4. 定义你的 Tasks ---

# 任务输入：在这里放入你要分析的 Anthropic URL 列表
# 示例 URL (请替换成你自己的真实URL)
anthropic_urls = [
    "https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview.md",
    "https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/claude-4-best-practices.md",
    "https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/prompt-generator.md",
    "https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/prompt-templates-and-variables.md",
    "https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/prompt-improver.md",
    "https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/be-clear-and-direct.md",
    "https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/multishot-prompting.md",
    "https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/chain-of-thought.md",
    "https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/use-xml-tags.md",
    "https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/system-prompts.md",
    "https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/prefill-claudes-response.md",
    "https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/chain-prompts.md",
    "https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/long-context-tips.md",
    "https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/extended-thinking-tips.md"
    # 你可以添加更多URL
]

# 动态为每个 URL 创建一个总结任务
summarization_tasks = []
for url in anthropic_urls:
    summarization_tasks.append(
        Task(
            description=f"Read and meticulously summarize the content, structure, and tone of the following URL: {url}",
            expected_output="A structured summary in markdown, covering key concepts, examples, and the overall writing style of the page.",
            agent=url_content_summarizer,
        )
    )

# Task 2: 关系总结任务 (接收 Agent 1 的内容)
task_synthesize_patterns = Task(
    description="Analyze the provided summaries of Anthropic's documentation. Identify and synthesize the common writing patterns, structural elements, and core principles that define their style.",
    expected_output="A markdown document detailing the recurring patterns in Anthropic's writing. For example: '1. Use of analogies. 2. Structure: Definition -> Good Example -> Bad Example. 3. Consistent use of terminology like 'constitutional AI'.'",
    agent=anthropic_pattern_synthesizer,
    context=summarization_tasks # 关键：依赖所有总结任务
)

# Task 3: 外部研究任务
task_research_external = Task(
    description="Find and summarize 2-3 prominent, non-Anthropic prompt engineering frameworks. Focus on their core ideas and structure.",
    expected_output="A report summarizing the key principles and structure of at least two other prompt engineering frameworks (e.g., from OpenAI, Cohere, or community standards).",
    agent=external_framework_researcher
)

# Task 4: 理论基础任务
task_provide_theory = Task(
    description="Provide a generic, theoretical framework for how to analyze any documentation or writing paradigm. What are the essential components to look for?",
    expected_output="A clear, educational guide or checklist on the fundamental components of a writing paradigm (e.g., Audience, Purpose, Structure, Tone, Core Principles).",
    agent=paradigm_theorist
)

# Task 5: 对比分析任务 (接收 Agent 2 和 3 的内容)
task_compare_frameworks = Task(
    description="Take the synthesized Anthropic patterns and the summary of external frameworks, then perform a comparative analysis. What makes the Anthropic style stand out?",
    expected_output="A comparative analysis in markdown, using a table or bullet points to highlight the similarities and, more importantly, the unique differences of the Anthropic paradigm.",
    agent=comparative_analyst,
    context=[task_synthesize_patterns, task_research_external] # 关键：依赖内部总结和外部研究
)

# Task 6: 最终生成任务 (接收 Agent 4, 2, 5 的内容)
task_generate_final_outline = Task(
    description="Synthesize all the inputs—the theoretical framework, the identified Anthropic patterns, and the comparative analysis—into a single, final outline that defines the Anthropic Prompt Engineering Writing Paradigm.",
    expected_output="A comprehensive, well-structured markdown document titled 'The Anthropic Prompt Engineering Writing Paradigm'. It should clearly outline the paradigm's principles, structure, tone, and key characteristics, making it easy for a new writer to understand and apply.",
    agent=final_outline_generator,
    context=[task_provide_theory, task_synthesize_patterns, task_compare_frameworks] # 关键：依赖理论、内部总结和对比分析
)

#%%
# --- 5. 组建并启动你的 Crew ---

# 整合所有 agents 和 tasks
all_agents = [
    url_content_summarizer,
    anthropic_pattern_synthesizer,
    external_framework_researcher,
    paradigm_theorist,
    comparative_analyst,
    final_outline_generator
]

all_tasks = [
    *summarization_tasks, # 解包所有总结任务
    task_synthesize_patterns,
    task_research_external,
    task_provide_theory,
    task_compare_frameworks,
    task_generate_final_outline
]

#%%
paradigm_crew = Crew(
    agents=all_agents,
    tasks=all_tasks,
    process=Process.sequential, # 任务必须按顺序执行，因为存在依赖关系
    verbose=True, # 使用 verbose=2 可以看到每个 Agent 的思考过程和行动
    memory=True, # 开启记忆功能，让 agent 可以在任务间共享更丰富的上下文
    embedder=embedder_config
)
#%%
# 启动 Crew！
print("🚀 Starting the Paradigm Analysis Crew...")
result = paradigm_crew.kickoff()

#%%
print("\n\n✅ Crew finished its work!")
print("--- Final Result ---")
print(result)