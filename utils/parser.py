# 我们可以把这个函数放在一个新的 utils/parser.py 文件里
import re


def parse_final_plan(markdown_text: str) -> dict:
    """
    Parses the final, detailed plan into a structured dictionary with sections and sub-sections.
    """
    plan = {"objective": "", "tone": "", "sections": []}

    # Extract Objective and Tone
    obj_match = re.search(r"Objective:\s*([\s\S]*?)(?=\n\nTone:)", markdown_text)
    if obj_match:
        plan["objective"] = obj_match.group(1).strip()

    tone_match = re.search(r"Tone:\s*([\s\S]*?)(?=\n\n---)", markdown_text)
    if tone_match:
        plan["tone"] = tone_match.group(1).strip()

    # Extract sections and their subsections
    section_pattern = r"(Section \d+:[\s\S]*?)(?=\n\n---|\Z)"
    sections_raw = re.findall(section_pattern, markdown_text)

    for sec_raw in sections_raw:
        title_match = re.search(r"Section \d+: (.*?)\n", sec_raw)
        goal_match = re.search(r"Goal: (.*?)\n", sec_raw)

        if not title_match:
            continue

        section_data = {
            "title": title_match.group(1).strip(),
            "goal": goal_match.group(1).strip() if goal_match else "",
            "sub_sections": []
        }

        # Extract subsections
        subsection_pattern = r"\*\s*(\d+\.\d+.*?):\s*([\s\S]*?)(?=\n\s*\*\s*\d+\.\d+|\Z)"
        subsections_raw = re.findall(subsection_pattern, sec_raw)

        for sub_raw in subsections_raw:
            sub_title = sub_raw[0].strip()
            details = sub_raw[1].strip()
            section_data["sub_sections"].append({
                "title": sub_title,
                "details": details
            })

        plan["sections"].append(section_data)

    return plan