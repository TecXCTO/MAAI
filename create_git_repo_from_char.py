import os
from pathlib import Path

# --- Your Flowchart Logic ---
# Syntax: Parent --> Child
flowchart_input = """
Root --> src
Root --> docs
Root --> tests
src --> main.py
src --> utils.py
docs --> README.md
"""

def generate_structure_from_flow(flow_text):
    lines = flow_text.strip().split('\n')
    
    for line in lines:
        if "-->" in line:
            # Split 'Parent --> Child' into usable names
            parts = line.split("-->")
            parent = parts[0].strip()
            child = parts[1].strip()
            
            # Logic: If it has an extension (like .py), create a file. 
            # Otherwise, create a directory.
            path = Path(parent) / child
            
            if "." in child:
                # Ensure the parent directory exists before creating the file
                os.makedirs(parent, exist_ok=True)
                path.touch(exist_ok=True)
                print(f"Created file: {path}")
            else:
                os.makedirs(path, exist_ok=True)
                print(f"Created directory: {path}")

if __name__ == "__main__":
    generate_structure_from_flow(flowchart_input)
