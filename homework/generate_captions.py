from pathlib import Path

import fire
from matplotlib import pyplot as plt

from .generate_qa import draw_detections, extract_frame_info, extract_kart_objects, extract_track_info

#TODO
def generate_caption(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
    """
    Generate caption for a specific view.
    """
    # 1. Ego car
    # {kart_name} is the ego car.

    # 2. Counting
    # There are {num_karts} karts in the scenario.

    # 3. Track name
    # The track is {track_name}.

    # 4. Relative position
    # {kart_name} is {position} of the ego car.

    karts = extract_kart_objects(info_path, view_index, img_width, img_height)
    track_name = extract_track_info(info_path)
    
    if not karts:
        return [f"An empty scene on the {track_name} track."]

    ego_kart = next((k for k in karts if k["is_ego_kart"]), None)
    other_karts = [k for k in karts if not k.get("is_ego_kart", False)]

    caption_parts = []

    # 1. Ego car
    if ego_kart:
        caption_parts.append(f"{ego_kart['kart_name']} is the ego car.")
    
    # 2. Track name
    caption_parts.append(f"The track is {track_name}.")

    # 3. Counting
    num_karts = len(karts)
    if num_karts == 1:
        caption_parts.append("There is 1 kart in the scene.")
    else:
        caption_parts.append(f"There are {num_karts} karts in the scene.")

    # 4. Relative position of one other kart (if available)
    if ego_kart and other_karts:
        # Describe the closest other kart for variety
        closest_kart = min(other_karts, key=lambda k: 
            ((k['center'][0] - ego_kart['center'][0])**2 + (k['center'][1] - ego_kart['center'][1])**2)**0.5
        )
        
        other_center_x, other_center_y = closest_kart['center']
        ego_center_x, ego_center_y = ego_kart['center']
        
        horizontal = "right" if other_center_x > ego_center_x else "left"
        vertical = "front" if other_center_y < ego_center_y else "behind"
        
        caption_parts.append(f"{closest_kart['kart_name']} is to the {vertical} and {horizontal} of the ego car.")

    return [" ".join(caption_parts)]


def check_caption(info_file: str, view_index: int):
    captions = generate_caption(info_file, view_index)

    print("\nCaption:")
    print("-" * 50)
    for i, caption in enumerate(captions):
        print(f"{i + 1}. {caption}")
        print("-" * 50)

    info_path = Path(info_file)
    base_name = info_path.stem.replace("_info", "")
    image_file = list(info_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0]

    annotated_image = draw_detections(str(image_file), info_file)

    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title(f"Frame {extract_frame_info(str(image_file))[0]}, View {view_index}")
    plt.show()


"""
Usage Example: Visualize QA pairs for a specific file and view:
   python generate_captions.py check --info_file ../data/valid/00000_info.json --view_index 0

You probably need to add additional commands to Fire below.
"""


def main():
    fire.Fire({"check": check_caption})


if __name__ == "__main__":
    main()
