# 文件: pdfs_rag/__init__.py

# 从我们的 agent_builder 模块中，导入主创建函数
from .essay_builder import EssayWriterGraph
from .planner_graph import PlannerGraph
from .sub_phrase import SectionWriterGraph
from .plan_parser import AiParserGraph
from .phrase_connector import ConnectorWriterGraph
from .edit_graph import EditGraph

# (可选) 你可以在这里定义包的元数据
__version__ = "0.1.0"