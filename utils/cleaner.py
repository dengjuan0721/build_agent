import json


def clean_none_values(d):
    """
    递归地从字典和列表中移除值为 None 的键。
    """

    if isinstance(d, dict):
        return {k: clean_none_values(v) for k, v in d.items() if v is not None}
    elif isinstance(d, list):
        return [clean_none_values(i) for i in d]
    else:
        return d


def create_writing_context(structured_plan: dict, section_to_write: dict) -> dict:
    """
    为一个子章节写作任务，构建一个智能的、上下文感知的精简上下文。
    """

    # 1. 提取顶层元数据
    context = {
        "objective": structured_plan.get("objective"),
        "target_audience": structured_plan.get("target_audience"),
        "tone": structured_plan.get("tone"),
    }

    # 2. 创建一个“智能目录”(Smart Table of Contents)
    #    这个新的 TOC 生成器会根据当前任务，动态地决定显示细节的层级。
    current_section_title = section_to_write.get("title")

    def generate_smart_toc(sections: list, parent_is_active=False, level=1) -> list:
        toc_lines = []
        for section in sections:
            # 检查当前 section 是否是我们要写的 section
            is_current_section = (section.get("title") == current_section_title)

            # 检查当前 section 是否是我们要写的 section 的祖先
            # (这是一个简化的检查，通过 parent_is_active 标志位向下传递)
            is_ancestor_or_active = parent_is_active or is_current_section

            # --- 决定是否展开 ---
            should_expand = is_ancestor_or_active

            # 添加当前行的标题
            prefix = ">>>" if is_current_section else "-"
            toc_lines.append(f"{'  ' * (level - 1)}{prefix} {section['title']}")

            # 如果需要展开，并且存在子章节，则递归深入
            if should_expand and section.get("sub_sections"):
                # 将 is_ancestor_or_active 状态传递给子节点
                # 如果父节点是 active 的，那么所有子节点都应该被认为是 active 路径的一部分
                toc_lines.extend(generate_smart_toc(section["sub_sections"], parent_is_active=is_ancestor_or_active,
                                                    level=level + 1))
        return toc_lines

    # 我们需要一个更鲁棒的方式来找到当前 section 的“家族树”
    # 让我们重写这个逻辑

    # --- V2: 更健壮的智能目录生成器 ---

    # a. 先找到当前活动节点的路径
    active_path = []

    def find_active_path(sections, target_title):
        for section in sections:
            if section.get("title") == target_title:
                return [target_title]  # 找到了
            if section.get("sub_sections"):
                path = find_active_path(section["sub_sections"], target_title)
                if path:  # 如果在子树中找到了
                    return [section["title"]] + path
        return None

    active_path = find_active_path(structured_plan.get("content_structure", []), current_section_title) or []

    # b. 再根据路径生成目录
    def generate_toc_with_path(sections, path_set, level=1):
        toc_lines = []
        for section in sections:
            is_active = section["title"] in path_set

            prefix = ">>>" if is_active and section["title"] == current_section_title else "-"

            toc_lines.append(f"{'  ' * (level - 1)}{prefix} {section['title']}")

            if is_active and section.get("sub_sections"):
                toc_lines.extend(generate_toc_with_path(section["sub_sections"], path_set, level + 1))
        return toc_lines

    toc_str = "\n".join(generate_toc_with_path(structured_plan.get("content_structure", []), set(active_path)))
    context["table_of_contents"] = toc_str

    # 3. 清理当前要写的子章节信息，移除 None 值 (保持不变)
    cleaned_section_to_write = clean_none_values(section_to_write)
    context["section_to_write"] = cleaned_section_to_write

    return context