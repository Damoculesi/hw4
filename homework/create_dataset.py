# used gemini to generate the code
import json
from pathlib import Path
from tqdm import tqdm
from .generate_qa import generate_qa_pairs

def create_full_dataset():
    """
    Generates a full Question-Answer dataset by iterating through all
    info files and views in the training data directory.
    """
    # Define the base data directory relative to this script's location
    base_data_dir = Path(__file__).parent.parent / 'data'
    train_dir = base_data_dir / 'train'
    output_file = train_dir / 'autogen_qa_pairs.json'

    print(f"Looking for data in: {train_dir}")
    print(f"Output will be saved to: {output_file}")

    # Find all the _info.json files in the training directory
    info_files = list(train_dir.glob('*_info.json'))

    if not info_files:
        print(f"Error: No '_info.json' files found in {train_dir}.")
        print("Please ensure you have downloaded and unzipped the data correctly.")
        return

    all_qa_pairs = []
    
    # Use tqdm for a nice progress bar
    print(f"Found {len(info_files)} info files. Processing all views...")
    for info_path in tqdm(info_files, desc="Processing info files"):
        # Each info file corresponds to 10 image views (00 to 09)
        base_name = info_path.stem.replace("_info", "")
        
        for view_index in range(10):
            # We need to construct the image file path to link it in the QA pair
            image_file_name = f"train/{base_name}_{view_index:02d}_im.jpg"
            
            # Generate the QA pairs for this specific view
            qa_pairs = generate_qa_pairs(str(info_path), view_index)
            
            # Add the image_file path to each QA pair
            for pair in qa_pairs:
                pair['image_file'] = image_file_name
                all_qa_pairs.append(pair)

    # Save all the collected QA pairs into a single file
    print(f"\nGenerated a total of {len(all_qa_pairs)} QA pairs.")
    with open(output_file, 'w') as f:
        json.dump(all_qa_pairs, f, indent=2)

    print(f"Successfully saved the dataset to {output_file}")

if __name__ == '__main__':
    create_full_dataset()