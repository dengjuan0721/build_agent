# 文件: utils/schemas.py (创建一个新文件来存放数据结构)

from pydantic import BaseModel, Field
from typing import List, Optional

class SubPoint(BaseModel):
    """Represents a single, detailed point within a larger section."""
    title: str = Field(description="The clear, concise title of this sub-point.")
    details: str = Field(description="The detailed instructions or content for this sub-point.")

class EssaySection(BaseModel):
    """Represents a major section of the essay (e.g., Introduction, a body paragraph, Conclusion)."""
    title: str = Field(description="The main, reader-facing title of this section.")
    goal: Optional[str] = Field(None, description="The specific objective or goal for this section.")
    details: str = Field(description="The complete, unstructured text content of this section's plan.")
    sub_points: List[SubPoint] = Field(description="A list of structured sub-points within this section.")

# class StructuredPlan(BaseModel):
#     """The root model for a fully parsed, structured essay plan."""
#     objective: str = Field(description="The main, overarching objective of the entire essay.")
#     target_audience: str = Field(description="The intended audience for the essay.")
#     tone: str = Field(description="The desired writing tone (e.g., authoritative, engaging).")
#     introduction: EssaySection = Field(description="The structured plan for the introduction.")
#     body_sections: List[EssaySection] = Field(description="A list of the main body sections of the essay.")
#     conclusion: EssaySection = Field(description="The structured plan for the conclusion.")


# schemas.py (新版本)

class PlanSection(BaseModel):
    """
    Represents a single, potentially nested, section of a content plan.
    This model is recursive, allowing sections to contain other sections.
    """
    title: str = Field(description="The clear, reader-facing title of this section or sub-point.")
    details: Optional[str] = Field(None,description="The detailed instructions, content, or explanation for this specific section.")

    # 关键！sub_sections 是一个包含 PlanSection 自身的列表
    sub_sections: Optional[List['PlanSection']] = Field(None,description="A list of nested sub-sections. Can be empty or None if there are no further sub-levels.")


# Pydantic v2 会自动处理前向引用 'PlanSection'
# 如果在旧版本或遇到问题，可以添加下面这行
# PlanSection.model_rebuild()


class StructuredPlan(BaseModel):
    """The root model for a fully parsed, structured essay plan."""
    objective: str = Field(description="The main, overarching objective of the entire essay.")
    target_audience: str = Field(description="The intended audience for the essay.")
    tone: str = Field(description="The desired writing tone (e.g., authoritative, engaging).")

    # 现在，我们使用统一的、可递归的 PlanSection
    content_structure: List[PlanSection] = Field(
        description="The main, hierarchical structure of the essay, starting with the Introduction.")

# from pydantic import BaseModel, Field

class ProposedEdit(BaseModel):
    """A structured representation of a proposed edit."""
    revised_content: str = Field(description="The full, revised text of the paragraph.")
    ai_reasoning: str = Field(description="A brief explanation of the changes made and why, based on the user's prompt.")