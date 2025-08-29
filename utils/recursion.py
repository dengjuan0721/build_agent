from utils.cleaner import create_writing_context
from typing import Dict, Any, List, Tuple
from utils.notebook import create_notebook_entry
def write_leaf_section(writer_factory, structured_plan: dict, leaf_node_data: dict) -> str:
    """
    负责调用 SectionWriterGraph 来撰写单个叶子节点的内容。
    """
    print(f"\n{'=' * 20} Starting Leaf Writer for: [{leaf_node_data['title']}] {'=' * 20}")

    # 1. 为当前任务构建精简的上下文
    writing_context = create_writing_context(structured_plan, leaf_node_data)

    # 2. 准备 SectionWriter 的输入
    writer_inputs = {
        "writing_context": writing_context,
        "max_revisions": 1,  # 每个子章节迭代1次
    }

    final_draft = ""
    # 3. 使用 'with' 语句来正确管理 SectionWriterGraph 的生命周期
    with writer_factory() as writer:
        # 4. 为每次独立的写作任务创建一个新的线程ID
        thread_config = {"configurable": {"thread_id": '1'}}
        # 5. 使用 .invoke() 来获取最终结果，避免 stream 的复杂性
        final_state = writer.graph.invoke(writer_inputs, thread_config)

        if final_state and 'draft' in final_state:
            final_draft = final_state.get('draft', '')
            print(f"--- Successfully generated draft for: [{leaf_node_data['title']}] ---")
        else:
            print(f"!!! Warning: Failed to generate draft for: [{leaf_node_data['title']}]")
            final_draft = f"### {leaf_node_data['title']}\n\n[Content generation failed for this section.]"

    return final_draft


def process_plan_recursively(
        node_data: dict,
        structured_plan: dict,
        writer_factory,  # 传递工厂函数
        connector_factory,
        level=1
) -> Tuple[str, List[Dict]]:
    """
    递归地遍历结构化计划，生成内容，并同时创建记事本元数据。

    Returns:
        一个元组: (生成的文本内容, 记事本条目列表)
    """

    content_parts = []

    # 1. 组装当前节点的标题
    #    我们只在最终返回时添加标题，避免在 Connector 的上下文中重复
    node_title = f"{'#' * (level+1)} {node_data['title']}"

    # 如果是叶子节点 (有 details)，直接调用写手
    if not node_data.get("sub_sections"):
        # write_leaf_section 负责调用 LeafWriterGraph 并返回内容字符串
        leaf_content = write_leaf_section(writer_factory, structured_plan, node_data)
        # ✨ 生成记事本条目
        notebook_entry = create_notebook_entry("leaf", node_data, leaf_content)
        # 最终返回的内容是：标题 + 生成的正文
        full_content = f"{node_title}\n\n{leaf_content}"
        return full_content, [notebook_entry]

    # 如果是枝干节点 (有 sub_sections)，递归处理子节点
    elif node_data.get("sub_sections"):
        all_notebook_entries = []  # 用于收集所有子节点的记事本条目

        #先写引言
        intro_paragraph = ""
        print(f"{'  ' * level}-> Calling Connector for INTRO of [{node_data['title']}]")
        with connector_factory() as beginner:
            # 准备上下文和输入
            connector_context = create_writing_context(  # 假设有这个函数
                structured_plan,
                node_data
            )
            connector_inputs = {
                "writing_context": connector_context,
                "max_revisions": 1,
                "mode": 'intro'
            }
            final_state = beginner.graph.invoke(connector_inputs, {"configurable": {"thread_id": "1"}})
            intro_paragraph = final_state.get('draft', f"[Introduction for {node_data['title']} failed to generate.]")

        # ✨ 生成引言的记事本条目
        intro_entry = create_notebook_entry("intro", node_data, intro_paragraph)
        all_notebook_entries.append(intro_entry)

        children_content_parts = []

        for sub_node in node_data["sub_sections"]:
            child_content, child_notebook_entries = process_plan_recursively(
                sub_node, structured_plan, writer_factory, connector_factory, level + 1
            )
            children_content_parts.append(child_content)
            all_notebook_entries.extend(child_notebook_entries)

        full_children_text = "\n\n".join(children_content_parts)


        # (这里调用 ConnectorWriter 来写总结)
        conclusion_paragraph = ""
        print(f"{'  ' * level}-> Calling Connector for CONCLUSION of [{node_data['title']}]")
        with connector_factory() as ender:
            # 关键修改！复用 create_writing_context
            ender_context = create_writing_context(structured_plan, node_data)

            # 关键修改！追加 sub_sections_content，对于总结，它是已生成的子内容
            ender_context['sub_sections_content'] = full_children_text
            connector_inputs = {
                "writing_context": ender_context,
                "max_revisions": 1,
                "mode": 'conclude'
            }
            final_state = ender.graph.invoke(connector_inputs, {"configurable": {"thread_id": "1"}})
            conclusion_paragraph = final_state.get('draft',
                                                   f"[Conclusion for {node_data['title']} failed to generate.]")
        # ✨ 生成结论的记事本条目
        conclusion_entry = create_notebook_entry("conclusion", node_data, conclusion_paragraph)
        all_notebook_entries.append(conclusion_entry)
    # d. 组装这个枝干节点的完整内容
    full_content = (
        f"{node_title}\n\n"
        f"{intro_paragraph}\n\n"
        f"{full_children_text}\n\n"
        f"{conclusion_paragraph}"
    )

    return full_content, all_notebook_entries