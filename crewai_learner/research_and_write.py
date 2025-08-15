from crewai import Agent, Task, Crew
import os
from langchain_deepseek import ChatDeepSeek
from langchain_community.llms.moonshot import Moonshot
from dotenv import load_dotenv
_ = load_dotenv(dotenv_path="/Users/dengjuan1/agent_crew/.env")
#%%
# 1. 导入官方的 FileWriterTool
from crewai_tools import FileWriterTool

# 2. 实例化一个通用的写入工具
#    我们不需要在此时指定任何参数
writer_tool = FileWriterTool()

#%% custom tool
from crewai_tools import ArxivPaperTool

# 1. 实例化专用的 arXiv 工具
arxiv_tool = ArxivPaperTool()

from crewai_tools import ScrapeWebsiteTool

# 实例化网页抓取工具
scrape_tool = ScrapeWebsiteTool()

#%% llm

llm = ChatDeepSeek(
    model="deepseek-chat",
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
    openai_api_base=os.getenv("DEEPSEEK_API_BASE"),
    temperature=0.0,
    max_tokens=8192
)

coder_llm = Moonshot(
    moonshot_api_key=os.getenv("MOONSHOT_API_KEY"),
    # base_url="https://api.moonshot.cn/v1",
    model= "moonshot/kimi-k2-0711-preview",
    max_tokens=16384
)
#%%
if hasattr(llm, 'openai_api_key') and hasattr(llm.openai_api_key, 'get_secret_value'):
    llm.openai_api_key = llm.openai_api_key.get_secret_value()
llm.api_key = llm.openai_api_key

#%% agent

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

#planner
planner = Agent(
    role="Content Planner",
    goal="Plan engaging and factually accurate content on {topic}."
          "Your plan MUST be heavily inspired by the structure and intuitive approach of Andrej Karpathy's famous blog post, 'The Unreasonable Effectiveness of Recurrent Neural Networks'. "
          ,
    backstory="You're working on planning a blog article "
              "about the topic: {topic}."
              "You collect information that helps the "
              "audience learn something "
              "and make informed decisions. "
              "You firmly believe in the teaching philosophy of Andrej Karpathy: show the magic first, then explain how it works. "
              "Your primary task is to analyze his classic RNN blog post to understand its narrative structure, its use of compelling examples (like generating Shakespeare or code), "
              "Your work is the basis for "
              "the Content Writer to write an article on this topic.",
    allow_delegation=False,
	verbose=True,
    llm=llm,
    tools=[scrape_tool]
)
#writer
writer = Agent(
    role="Content Writer",
    goal="Write insightful and factually accurate "
         "opinion piece about the topic: {topic}",
    backstory="You're working on a writing "
              "a new opinion piece about the topic: {topic}. "
              "You base your writing on the work of "
              "the Content Planner, who provides an outline "
              "and relevant context about the topic. "
              "You follow the main objectives and "
              "direction of the outline, "
              "as provide by the Content Planner. "
              "You also provide objective and impartial insights "
              "and back them up with information "
              "provide by the Content Planner. "
              "You acknowledge in your opinion piece "
              "when your statements are opinions "
              "as opposed to objective statements.",
    allow_delegation=False,
    verbose=True,
    llm=llm
)
coder = Agent(
    role="Python Code Expert",
    goal="Enhance the blog post on {topic} by adding clear, correct, and relevant Python code examples.",
    backstory=(
        "You are an expert Python developer with a passion for education. "
        "You specialize in breaking down complex technical topics, like {topic}, into understandable code snippets. "
        "You receive drafts from the Content Writer and your mission is to identify the most crucial points "
        "that can be clarified with a practical code example. Your code is always clean, well-commented, "
        "and directly illustrates the concept being discussed in the text, "
        "making the article more valuable and practical for the reader."
    ),
    allow_delegation=False,
    verbose=True,
    llm=coder_llm

)

#editor
editor = Agent(
    role="Editor",
    goal="Edit a given blog post to align with "
         "the writing style of the organization. ",
    backstory="You are an editor who receives a blog post "
              "from the Content Writer. "
              "Your goal is to review the blog post "
              "to ensure that it follows journalistic best practices,"
              "provides balanced viewpoints "
              "when providing opinions or assertions, "
              "and also avoids major controversial topics "
              "or opinions when possible.",
    allow_delegation=False,
    verbose=True,
    llm=llm
)
#%% task
# 定义 Karpathy 的博客 URL
karpathy_rnn_url = "http://karpathy.github.io/2015/05/21/rnn-effectiveness/"

plan = Task(
    description=(
        "1. Prioritize the latest trends, key players, "
            "and noteworthy news on {topic}.\n"
        "2. Identify the target audience, considering "
            "their interests and pain points.\n"
        "3. Develop a detailed content outline including "
            "an introduction, key points, and a call to action.\n"
        "4. Include SEO keywords and relevant data or sources."
        "5.  Your MOST IMPORTANT first step is to use the scrape_tool to read and deeply analyze the content of this specific URL: "
        f"'{karpathy_rnn_url}'. Extract its key themes, teaching structure, and the types of examples used.\n"
    ),
    expected_output="A comprehensive content plan document "
        "with an outline, audience analysis, "
        "SEO keywords, and resources.\n"
        "The outline must contain publication-ready headings and subheadings. \n"
    ,
    agent=planner,
)

write = Task(
    description=(
        "1. Use the content plan to craft a compelling "
            "blog post on {topic}."
        "This means you will strictly use the exact headings and subheadings as they are written in the plan.\n"
        "2. Incorporate SEO keywords naturally.\n"
		"3. Sections/Subtitles are properly and slightly adjusted "
            "in an engaging manner.\n"
        "4. Ensure the post is structured with an "
            "engaging introduction, insightful body, "
            "and a summarizing conclusion.\n"
        "5. Proofread for grammatical errors and "
            "alignment with the brand's voice.\n"
        "6. Strongly refer to Andrej Karpathy's "
        f"'{karpathy_rnn_url}'The Unreasonable Effectiveness of Recurrent Neural Networks'."
    ),
    expected_output="A well-written chinese blog post "
        "in markdown format, ready for publication, "
        "each section should have 2 or 3 paragraphs.",
    agent=writer,
    context=[plan]
)

add_link = Task(
    description=(
        "You are tasked with enriching the provided blog post draft about {topic} with academic insights from arXiv papers. "
        "Your goal is not just to insert information, but to **seamlessly integrate** it into the existing narrative.\n\n"
        "Follow these steps meticulously:\n"
        "1.  **Analyze the Draft**: Read the entire blog post to understand its structure, tone, and logical flow.\n"
        "2.  **Identify Integration Points**: Find the most suitable sections in the draft to introduce a key concept or finding from a relevant arXiv paper. These should be points where an academic reference can strengthen an argument or provide a foundational definition.\n"
        "3.  **Craft a Smooth Transition**: Before inserting the reference block, write a transitional sentence or two. This sentence MUST connect the preceding text with the upcoming quote. For example, instead of just dropping a quote, you could write: 'This idea was formally introduced in a seminal paper that stated:'\n"
        "4.  **Insert the Reference Block**: Use a Markdown Blockquote (`>`) to present the most relevant phrase from the paper, followed by the citation. The format should be:\n"
        "    > (Translated, relevant phrase from the paper). — Author, et al. (Year). *Title of Paper*. arXiv:[link to paper]\n"
        "5.  **Ensure Logical Cohesion**: After the blockquote, if necessary, add another sentence to connect the quote back to the main argument of your article. Ensure the entire section reads as a single, coherent piece.\n\n"
        "**Crucial Requirement**: The final text must flow naturally. Avoid abrupt insertions. The reader should feel that the reference is a natural and essential part of the explanation."
    ),
    expected_output=(
        "The original blog post, now enhanced with seamlessly integrated arXiv references. "
        "Each reference should be introduced with a proper transition and formatted as a Markdown Blockquote, "
        "maintaining a smooth and logical narrative flow throughout the article."
    ),
    agent=writer, # 或者一个专门的 'editor' agent
    context=[write], # 依赖于 writer 的初稿
    tools=[arxiv_tool] # Agent 需要用这个工具来查找论文的详细信息
)


# 新增的 Code Task
code = Task(
    description=(
        "1. Review the blog post draft about {topic} provided by the Content Writer.\n"
        "2. Identify the best sections to insert Python code examples to illustrate key concepts (e.g., how to create model, calculate probabilities).\n"
        "3. Write simple, clear, and correct Python code snippets for these sections.\n"
        "4. Ensure each code snippet is well-commented to explain the logic.\n"
        "5. Integrate the code examples seamlessly into the post using Markdown code blocks."
    ),
    expected_output=(
        "The original blog post, now enhanced with relevant, well-commented Python code examples in Markdown format. "
        "The code should be seamlessly integrated into the text to clarify the technical explanations."
    ),
    agent=coder,
    context=[add_link] # 关键：此任务依赖write任务的输出
)

edit = Task(
    description=("Proofread the given blog post for "
                 "grammatical errors and "
                 "alignment with the brand's voice."),
    expected_output="A well-written blog post in markdown format, "
                    "ready for publication, "
                    "each section should have 2 or 3 paragraphs.\n"
                    "remove text beyond '```markdown```'.\n"
    ,
    agent=editor,
    context=[code]
)

#%% crew
crew = Crew(
    agents=[planner, writer, coder, editor],
    tasks=[plan, write, add_link, code, edit],
    verbose=True,
    # memory=True # L2
)

#%%
result = crew.kickoff(inputs={"topic": "神经语言模型RNN"})

#%%
planning_crew = Crew(agents=[planner], tasks=[plan], verbose=True,)

planner_output = planning_crew.kickoff(inputs={"topic": "神经语言模型RNN"})

#%%
import re

def parse_planner_outline(markdown_text: str) -> list[dict]:
    """
    Parses the markdown outline from the planner into a structured list of sections.
    """
    sections = []
    # This regex is designed to capture main sections like "### 1. Introduction: ..."
    # and all the content (including sub-points) until the next main section.
    pattern = r"###\s*([\d\.]+\.\s*.*?)\n([\s\S]*?)(?=\n###\s*[\d\.]+\.\s*|\Z)"

    matches = re.findall(pattern, markdown_text)

    for match in matches:
        title = match[0].strip()
        content_points = match[1].strip()

        # We can create a more detailed prompt for the writer here
        section_details = f"Title: {title}\nKey points to cover:\n{content_points}"

        sections.append({
            "title": title,
            "details": section_details
        })

    return sections

sections_to_write = parse_planner_outline(planner_output.raw)
print("✅ Planner output parsed. Sections to write:")
for i, sec in enumerate(sections_to_write):
    print(f"  {i+1}. {sec['title']}")

#%%

# 整个大纲作为全局上下文
full_context_outline = planner_output.raw
#%%
write_section_task = Task(
    description=(
        "Your task is to write ONE specific section of a larger blog post. \n\n"
        "**Global Context (The Full Outline):**\n---\n"
        f"{full_context_outline}\n---\n\n"
        "**Your Specific Section to Write:**\n---\n"
        f"{sections_to_write[6]['details']}\n---\n\n"
        "**Instructions:**\n"
        "1. Focus ONLY on writing the content for your assigned section. "
        "2. Strictly follow the key points listed for your section. "
        "3. Ensure your writing style is engaging and follows the Karpathy philosophy mentioned in the global context. "
        "4. Your final output should be ONLY the markdown content for this one section, starting with its title as a heading."
    ),
    expected_output="The complete markdown content for the assigned section, starting with its title as a heading (e.g., '### 2. What Are RNNs?').",
    agent=writer
)

code_section_task = Task(
    description=(
        "1. Review the section draft provided by the section writer.\n"
        "2. Identify the best sections to insert Python code examples to illustrate key concepts (e.g., how to create model, calculate probabilities).\n"
        "3. Write clear, and correct Python code snippets for these sections.\n"
        "4. Ensure each code snippet is well-commented to explain the logic.\n"
        "5. Integrate the code examples seamlessly into the post using Markdown code blocks."
        "WARNING: YOU ONLY INSERT, NEVER DELETE"
    ),
    expected_output=(
        "The original blog post, now enhanced with relevant, well-commented Python code examples in Markdown format. "
        "The code should be seamlessly integrated into the text by paragraphs to clarify the technical explanations."
    ),
    agent=coder,
    context=[write_section_task] # 关键：此任务依赖write任务的输出
)

section_writing_crew = Crew(
    agents=[writer, coder], #, coder, link_adder, editor],
    tasks=[write_section_task, code_section_task], #, code_section_task, ...],
    verbose=True # 可以设置成 1 或 0，避免过多日志
)
#%%
section_result = section_writing_crew.kickoff()

#%%
