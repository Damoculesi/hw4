import json
import os
import re
from typing import List

# Reuse your existing helper functions
from homework.generate_qa import extract_kart_objects, extract_track_info, generate_qa_pairs

def parse_image_filename(image_filename: str):
    """
    Parse image filename to get info JSON path and view index.
    Example: "valid/00048_09_im.jpg" → ("valid/00048_info.json", 9)
    """
    match = re.match(r"(.*)/([0-9a-f]+)_(\d+)_im\.jpg", image_filename)
    if not match:
        raise ValueError(f"Invalid image filename format: {image_filename}")
    folder, base_id, view_index = match.groups()
    info_path = os.path.join(folder, f"{base_id}_info.json")
    return info_path, int(view_index)


def generate_balanced_qa_from_image_file(image_file: str, img_width=150, img_height=100) -> List[dict]:
    """
    Generate QA pairs from a single image file path.
    """
    info_path, view_index = parse_image_filename(image_file)
    
    # Ensure file exists
    if not os.path.exists(info_path):
        print(f"Warning: info file {info_path} not found.")
        return []
    
    karts = extract_kart_objects(info_path, view_index, img_width, img_height)
    track = extract_track_info(info_path)
    ego_kart = next((k for k in karts if k["is_ego_kart"]), None)

    qa_pairs = []

    if ego_kart:
        qa_pairs.append({
            "question": "What kart is the ego car?",
            "answer": ego_kart["kart_name"],
            "image_file": image_file
        })

    qa_pairs.append({
        "question": "How many karts are there in the scenario?",
        "answer": str(len(karts)),
        "image_file": image_file
    })

    qa_pairs.append({
        "question": "What track is this?",
        "answer": track,
        "image_file": image_file
    })

    return qa_pairs


def generate_all_balanced_qas(image_list_file: str, output_file: str):
    """
    Generate QA pairs for all image files listed in a given file.
    """
    with open(image_list_file) as f:
        images = json.load(f)

    all_qas = []
    for entry in images:
        image_file = entry["image_file"]
        all_qas.extend(generate_balanced_qa_from_image_file(image_file))

    with open(output_file, "w") as f:
        json.dump(all_qas, f, indent=2)
    print(f"✅ Saved {len(all_qas)} QA pairs to {output_file}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python generate_qas.py <image_list.json> <output.json>")
        exit(1)

    image_list_file = sys.argv[1]
    output_file = sys.argv[2]
    generate_all_balanced_qas(image_list_file, output_file)
