import asyncio
from utils.cleaner import create_writing_context


# --- 1. 改造叶子节点写入函数 ---
async def write_leaf_section_async(writer_factory, structured_plan: dict, leaf_node_data: dict) -> str:
    """
    [异步版] 负责调用 SectionWriterGraph 来撰写单个叶子节点的内容。
    """
    print(f"\n{'=' * 20} Starting Leaf Writer for: [{leaf_node_data['title']}] {'=' * 20}")

    # 1. 为当前任务构建精简的上下文 (同步操作，没问题)
    writing_context = create_writing_context(structured_plan, leaf_node_data)

    # 2. 准备 SectionWriter 的输入
    writer_inputs = {
        "writing_context": writing_context,
        "max_revisions": 1,
    }

    final_draft = ""
    # 3. 使用 'with' 语句
    with writer_factory() as writer:
        # 4. 准备线程ID
        thread_config = {"configurable": {"thread_id": '1'}}

        # 5. 【核心改动】使用 .ainvoke() 并 await 其结果
        final_state = await writer.graph.ainvoke(writer_inputs, thread_config)

        if final_state and 'draft' in final_state:
            final_draft = final_state.get('draft', '')
            print(f"--- Successfully generated draft for: [{leaf_node_data['title']}] ---")
        else:
            print(f"!!! Warning: Failed to generate draft for: [{leaf_node_data['title']}]")
            final_draft = f"### {leaf_node_data['title']}\n\n[Content generation failed for this section.]"

    return final_draft


# --- 2. 改造递归处理函数 ---
async def process_plan_recursively_async(
        node_data: dict,
        structured_plan: dict,
        writer_factory,
        connector_factory,
        level=1
) -> str:
    """
    [异步版] 递归地遍历结构化计划，为叶子节点调用写手，并组装内容。
    """
    node_title = f"{'#' * (level + 1)} {node_data['title']}"

    # 如果是叶子节点 (有 details)
    if not node_data.get("sub_sections"):
        # 【核心改动】await 异步的叶子节点写入函数
        leaf_content = await write_leaf_section_async(writer_factory, structured_plan, node_data)
        return f"{node_title}\n\n{leaf_content}"

    # 如果是枝干节点 (有 sub_sections)
    elif node_data.get("sub_sections"):
        # a. [并发执行] 写引言和处理所有子节点可以同时进行
        print(f"{'  ' * level}-> Preparing tasks for INTRO and SUB-SECTIONS of [{node_data['title']}]")

        # 准备引言任务
        with connector_factory() as beginner:
            connector_context = create_writing_context(structured_plan, node_data)
            intro_inputs = {
                "writing_context": connector_context, "max_revisions": 1, "mode": 'intro'
            }
            intro_task = beginner.graph.ainvoke(intro_inputs, {"configurable": {"thread_id": "1"}})

        # 准备所有子节点的递归处理任务
        children_tasks = [
            process_plan_recursively_async(
                sub_node, structured_plan, writer_factory, connector_factory, level + 1
            )
            for sub_node in node_data["sub_sections"]
        ]

        # 【核心改动】使用 asyncio.gather 并发执行引言和所有子节点的生成
        all_results = await asyncio.gather(intro_task, *children_tasks)

        # 解包结果
        intro_state = all_results[0]
        children_content_parts = all_results[1:]

        intro_paragraph = intro_state.get('draft', f"[Introduction for {node_data['title']} failed to generate.]")
        full_children_text = "\n\n".join(children_content_parts)

        # b. [单独执行] 写总结 (因为它依赖于子节点的内容)
        conclusion_paragraph = ""
        print(f"{'  ' * level}-> Calling Connector for CONCLUSION of [{node_data['title']}]")
        with connector_factory() as ender:
            ender_context = create_writing_context(structured_plan, node_data)
            ender_context['sub_sections_content'] = full_children_text
            conclusion_inputs = {
                "writing_context": ender_context, "max_revisions": 1, "mode": 'conclude'
            }
            # 【核心改动】await 异步调用
            final_state = await ender.graph.ainvoke(conclusion_inputs, {"configurable": {"thread_id": "1"}})
            conclusion_paragraph = final_state.get('draft',
                                                   f"[Conclusion for {node_data['title']} failed to generate.]")

        # d. 组装这个枝干节点的完整内容
        return (
            f"{node_title}\n\n"
            f"{intro_paragraph}\n\n"
            f"{full_children_text}\n\n"
            f"{conclusion_paragraph}"
        )

    # 如果节点既不是叶子也不是枝干，返回空字符串
    return ""