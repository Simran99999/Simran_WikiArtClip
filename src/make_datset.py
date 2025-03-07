# src/data/make_dataset.py
import os
import shutil
import json

def create_dirs(base_path):
    dirs = ['raw', 'interim', 'processed', 'external']
    for d in dirs:
        os.makedirs(os.path.join(base_path, d), exist_ok=True)

def load_captions(captions_dir):
    captions = {}
    for file in os.listdir(captions_dir):
        if file.endswith(".txt"):
            with open(os.path.join(captions_dir, file), 'r', encoding='utf-8') as f:
                captions[file] = f.readlines()
    return captions

def save_captions_json(captions, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(captions, f, indent=4)

if __name__ == "__main__":
    base_path = "data"
    create_dirs(base_path)
    captions = load_captions("data/interim/captions")
    save_captions_json(captions, "data/processed/captions.json")
    print("Dataset processed and saved.")
