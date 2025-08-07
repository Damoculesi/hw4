# used gemini to generate the code
import json
from pathlib import Path
from tqdm import tqdm
from .generate_captions import generate_caption

def create_full_caption_dataset():
    """
    Generates a full caption dataset by iterating through all
    info files and views in the training data directory.
    For each image, it creates as many pairs as there are captions returned.
    """
    base_data_dir = Path(__file__).parent.parent / 'data'
    train_dir = base_data_dir / 'train'
    output_file = train_dir / 'autogen_captions.json'

    print(f"Looking for data in: {train_dir}")
    print(f"Output will be saved to: {output_file}")

    info_files = list(train_dir.glob('*_info.json'))

    if not info_files:
        print(f"Error: No '_info.json' files found in {train_dir}.")
        return

    all_caption_pairs = []

    print(f"Found {len(info_files)} info files. Processing all views...")
    for info_path in tqdm(info_files, desc="Processing info files"):
        base_name = info_path.stem.replace("_info", "")

        for view_index in range(10):
            image_file_name = f"train/{base_name}_{view_index:02d}_im.jpg"

            # generate_caption returns a list[str]; create one pair per sentence
            sentences = generate_caption(str(info_path), view_index) or []
            for s in sentences:
                s = (s or "").strip()
                if not s:
                    continue
                all_caption_pairs.append({
                    'image_file': image_file_name,
                    'caption': s
                })

    print(f"\nGenerated a total of {len(all_caption_pairs)} image-caption pairs.")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_caption_pairs, f, indent=2, ensure_ascii=False)

    print(f"Successfully saved the dataset to {output_file}")

if __name__ == '__main__':
    create_full_caption_dataset()
