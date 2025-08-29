import os
import json
from langgraph_writer import EditGraph


def get_full_content_from_file(file_path: str, notebook_entry: dict) -> str:
    """A helper to extract the full paragraph text from a file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    return content.split(notebook_entry['location_description'].split("'")[1])[1].strip()


def run_editor(task_name:str, notebook_path, content_path):

    if not os.path.exists(notebook_path):
        print(f"Error: Notebook file not found at {notebook_path}")
        return

    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    # --- User Interaction Loop ---
    while True:
        print("\n\n--- 📖 Available Sections to Edit ---")
        for i, entry in enumerate(notebook):
            print(f"[{i}] {entry['location_description']}: {entry['content_preview']}")

        try:
            choice = input("\nEnter the number of the section to edit (or 'exit'): ")
            if choice.lower() == 'exit':
                break

            selected_index = int(choice)
            selected_entry = notebook[selected_index]

            title = selected_entry['location_description'].split("'")[1]
            md_filename = ""
            for f in os.listdir(content_path):
                if title in f:
                    md_filename = f
                    break

            if not md_filename:
                print(f"Error: Could not find .md file for section '{title}'")
                continue

            file_to_edit_path = os.path.join(content_path, md_filename)

            # Here we need to get the full original content.
            # This is a critical step. A robust solution would parse markdown.
            # For simplicity, let's assume the file *is* the content for leaf nodes.
            with open(file_to_edit_path, 'r', encoding='utf-8') as f:
                # Naive implementation: read everything after the title markdown
                lines = f.readlines()
                original_content = "".join(lines[2:])  # Skip title and newline

            user_prompt = input("What change would you like to make? ")

            # --- Initialize and run the graph ---
            editor_graph = EditGraph()
            initial_state = {
                "notebook_path": notebook_path,
                "file_to_edit_path": file_to_edit_path,
                "location_description": selected_entry['location_description'],
                "original_content": original_content,
                "user_prompt": user_prompt,
                "user_feedback": "",  # Start with empty feedback
                "proposed_edit": None,
                "revision_count": 0
            }

            final_state = editor_graph.graph.invoke(initial_state)
            print("\n--- ✅ Edit session for this section is complete! ---")

        except (ValueError, IndexError):
            print("Invalid input. Please enter a valid number.")
        except Exception as e:
            print(f"An error occurred: {e}")


if __name__ == "__main__":
    run_editor()