from crewai import Agent, Task, Crew
import os
from langchain_deepseek import ChatDeepSeek
from langchain_community.llms.moonshot import Moonshot
from dotenv import load_dotenv
_ = load_dotenv(dotenv_path="/Users/dengjuan1/build_agent/.env")
#%% tools
from crewai_tools import ScrapeWebsiteTool

# 实例化网页抓取工具
scrape_tool = ScrapeWebsiteTool()

from crewai_tools import PDFSearchTool

# LeNet-5 PDF 的 URL
lenet_pdf_url = "http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf"
alex_net_nrl = "https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf"

# 实例化 PDF 工具，并在创建时就指定要处理的 PDF 文件
# 工具会在内部下载并加载这个 PDF
pdf_tool = PDFSearchTool()

#%%
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

if hasattr(llm, 'openai_api_key') and hasattr(llm.openai_api_key, 'get_secret_value'):
    llm.openai_api_key = llm.openai_api_key.get_secret_value()
llm.api_key = llm.openai_api_key
#%% agent
#planner
planner = Agent(
    role="Content Planner",
    goal="Plan engaging and factually accurate content on {topic}."
          "Your plan MUST be heavily inspired by given research paper and its theory, "
          f"url:{lenet_pdf_url}, {alex_net_nrl}"
          ,
    backstory="You're working on planning an article collection "
              "about the topic: {topic}."
              "You collect information that helps the "
              "audience learn something "
              "and make informed decisions. "
              "You equipe the knowledge from scraping website and reading famous paper."
              "Your primary task is to analyze given website content structure, "
              "Your work is the basis for "
              "the Content Writer to write an article on this topic.",
    allow_delegation=False,
	verbose=True,
    llm=llm,
    tools=[scrape_tool, pdf_tool]
)
#%% task

plan = Task(
    description=(
        "1. Prioritize the latest trends, key players, "
            "and noteworthy news on {topic}.\n"
        "2. Identify the target audience, considering "
            "their interests and pain points.\n"
        "3. Develop a detailed content outline including "
            "an introduction, key points, and a call to action.\n"
        "4. Include SEO keywords and relevant data or sources."
        "5. Your MOST IMPORTANT first step is to use the pdf_tool to read and deeply analyze the content of given specific URLs. "
        "Extract its key themes, teaching structure, or the types of examples used.\n"
    ),
    expected_output="A comprehensive content plan document "
        "with an outline, audience analysis, "
        "SEO keywords, and resources.\n"
        "The outline must contain publication-ready headings and subheadings. \n"
    ,
    agent=planner,
)
#%%
planning_crew = Crew(agents=[planner], tasks=[plan], verbose=True,memory=True)

planner_output = planning_crew.kickoff(inputs={"topic": "神经语言模型CNN"})


