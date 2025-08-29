# utils/notebook_utils.py (可以创建一个新文件来存放这个工具函数)

from typing import Dict, Any, Literal


def create_notebook_entry(
        paragraph_type: Literal["leaf", "intro", "conclusion"],
        node_data: Dict[str, Any],
        content: str
) -> Dict[str, Any]:
    """
    根据段落内容和节点数据，创建一个标准格式的记事本条目。

    Args:
        paragraph_type: 段落类型 ('leaf', 'intro', 'conclusion').
        node_data: 当前章节的节点数据 (来自结构化JSON).
        content: 已生成的段落文本内容.

    Returns:
        一个代表记事本条目的字典.
    """
    if not content:
        # 如果内容为空，返回一个标记条目，避免后续处理出错
        return {
            "type": paragraph_type,
            "location_description": f"Failed to generate content for section: '{node_data.get('title', 'Untitled')}'",
            "content_preview": "...",
            "edit_history": []
        }

    # 生成内容预览
    # 取前15个字符和后15个字符
    preview_length = 15
    if len(content) <= preview_length * 2:
        preview = content
    else:
        preview = f"{content[:preview_length]}...{content[-preview_length:]}"

    # 清理换行符，让预览更整洁
    preview = preview.replace('\n', ' ').strip()

    entry = {
        "type": paragraph_type,
        "location_description": f"Section: '{node_data.get('title', 'Untitled')}'",
        "content_preview": preview,
        "edit_history": []  # 初始化为空列表
    }

    return entry