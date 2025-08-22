import nest_asyncio
nest_asyncio.apply()
#%%
# urls = [
#     "https://openreview.net/pdf?id=VtmBAGCN7o",
#     "https://openreview.net/pdf?id=6PmJoRfdaK",
#     "https://openreview.net/pdf?id=hSyW5go0v8",
# ]

papers = [
    "metagpt.pdf",
    "longlora.pdf",
    "selfrag.pdf",
]
#%%
from utils.get_doc_tool import get_doc_tools
from pathlib import Path

paper_to_tools_dict = {}
for paper in papers:
    print(f"Getting tools for paper: {paper}")
    vector_tool, summary_tool = get_doc_tools(paper, Path(paper).stem)
    paper_to_tools_dict[paper] = [vector_tool, summary_tool]