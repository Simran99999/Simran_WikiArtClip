import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import re
from tqdm import tqdm
import random
import argparse
from pathlib import Path

class WikiArtCLIPDataset(Dataset):
    """
    Dataset for WikiArt CLIP fine-tuning
    """
    def __init__(self, dataframe, root_dir='', transform=None, max_caption_length=77):
        """
        Args:
            dataframe: DataFrame with image paths and captions
            root_dir: Root directory to prepend to image paths if needed
            transform: Optional image transformations
            max_caption_length: Maximum length of captions (for CLIP tokenizer)
        """
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.transform = transform
        self.max_caption_length = max_caption_length
        
        # Clean image paths if necessary
        self.dataframe['Image_Path'] = self.dataframe['Image_Path'].apply(self._clean_path)
        
    def _clean_path(self, path):
        """
        Cleans the image path to make it relative or use the correct separators
        """
        # Extract the relevant part of the path (after "WikiArtCLIP" or similar directories)
        # This makes paths more portable across systems
        if "/Users/" in path:
            # Extract just the filename for simplicity
            return os.path.basename(path)
        return path
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_path = self.dataframe.iloc[idx]['Image_Path']
        caption = self.dataframe.iloc[idx]['Caption']
        
        # Handle relative or absolute paths
        if not os.path.isabs(img_path) and self.root_dir:
            img_path = os.path.join(self.root_dir, img_path)
        
        try:
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
                
            return {
                'image': image, 
                'caption': caption,
                'image_path': img_path
            }
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a placeholder in case of errors
            return {
                'image': torch.zeros((3, 224, 224)),
                'caption': caption,
                'image_path': img_path
            }

def preprocess_wikiart_dataset(input_csv, output_dir, img_root_dir=None, val_split=0.1, test_split=0.1):
    """
    Preprocesses the WikiArt dataset for CLIP fine-tuning
    
    Args:
        input_csv: Path to the merged_captions.csv file
        output_dir: Directory to save the processed data
        img_root_dir: Root directory for images (if None, uses absolute paths)
        val_split: Fraction of data to use for validation
        test_split: Fraction of data to use for testing
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the dataset
    print(f"Loading dataset from {input_csv}")
    df = pd.read_csv("/Users/sikumari/Desktop/WikiArtCLIP/data/processed/merged_captions.csv")
    
    # Check for duplicates in captions
    print("Checking for duplicate captions...")
    duplicate_captions = df['Caption'].value_counts()
    common_captions = duplicate_captions[duplicate_captions > 100].index.tolist()
    
    print(f"Found {len(common_captions)} very common captions (appearing >100 times)")
    print("Most common captions:")
    for caption, count in df['Caption'].value_counts().head(10).items():
        print(f"  - '{caption}': {count} occurrences")
    
    # Optional: Remove or mark images with generic captions
    print("Processing captions...")
    
    # Extract art style information from paths where possible
    def extract_style(path):
        # Pattern matches standard WikiArt organization by style
        style_match = re.search(r'/([\w-]+)/', path)
        if style_match:
            style = style_match.group(1).replace('-', ' ').title()
            return style
        return None
    
    df['art_style'] = df['Image_Path'].apply(extract_style)
    
    # Enhance captions with style information where generic
    def enhance_caption(row):
        caption = row['Caption']
        style = row['art_style']
        
        # If caption is one of the very common generic ones and we have style info
        if caption in common_captions and style:
            return f"{caption} in the {style} style"
        return caption
    
    df['Enhanced_Caption'] = df.apply(enhance_caption, axis=1)
    
    # Create train/val/test splits
    print("Creating dataset splits...")
    # Shuffle the dataframe
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Calculate split indices
    total_samples = len(df)
    val_size = int(val_split * total_samples)
    test_size = int(test_split * total_samples)
    train_size = total_samples - val_size - test_size
    
    # Split the dataset
    train_df = df[:train_size]
    val_df = df[train_size:train_size+val_size]
    test_df = df[train_size+val_size:]
    
    print(f"Dataset split: {len(train_df)} train, {len(val_df)} validation, {len(test_df)} test")
    
    # Save processed splits
    train_df.to_csv(os.path.join(output_dir, 'wikiart_train.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'wikiart_val.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'wikiart_test.csv'), index=False)
    
    # Create a metadata file with dataset statistics
    metadata = {
        'total_samples': total_samples,
        'train_samples': len(train_df),
        'val_samples': len(val_df),
        'test_samples': len(test_df),
        'unique_captions': df['Caption'].nunique(),
        'enhanced_captions': df['Enhanced_Caption'].nunique(),
        'common_captions': common_captions
    }
    
    metadata_df = pd.DataFrame([metadata])
    metadata_df.to_csv(os.path.join(output_dir, 'wikiart_metadata.csv'), index=False)
    
    print(f"Preprocessing complete. Files saved to {output_dir}")
    
    return train_df, val_df, test_df


def create_training_samples(train_df, output_dir, sample_size=None):
    """
    Creates a jsonl file with training samples in the format expected by CLIP
    
    Args:
        train_df: DataFrame with training data
        output_dir: Directory to save the output
        sample_size: Optional number of samples to generate (for testing)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if sample_size:
        train_df = train_df.sample(n=min(sample_size, len(train_df)))
    
    # Create CLIP-compatible format
    clip_samples = []
    for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Creating CLIP samples"):
        sample = {
            "image_path": row['Image_Path'],
            "text": row['Enhanced_Caption'] if 'Enhanced_Caption' in row else row['Caption']
        }
        clip_samples.append(sample)
    
    # Save as a jsonl file
    import json
    with open(os.path.join(output_dir, 'wikiart_clip_samples.jsonl'), 'w') as f:
        for sample in clip_samples:
            f.write(json.dumps(sample) + '\n')
    
    print(f"Created {len(clip_samples)} CLIP training samples")


def main():
    parser = argparse.ArgumentParser(description='Preprocess WikiArt dataset for CLIP fine-tuning')
    parser.add_argument('--input_csv', type=str, required=True, help='Path to the merged_captions.csv file')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the processed data')
    parser.add_argument('--img_root_dir', type=str, default=None, help='Root directory for images')
    parser.add_argument('--val_split', type=float, default=0.1, help='Fraction of data to use for validation')
    parser.add_argument('--test_split', type=float, default=0.1, help='Fraction of data to use for testing')
    parser.add_argument('--sample_size', type=int, default=None, help='Number of samples to use (for testing)')
    
    args = parser.parse_args()
    
    # Process the dataset
    train_df, val_df, test_df = preprocess_wikiart_dataset(
        args.input_csv, 
        args.output_dir,
        args.img_root_dir,
        args.val_split,
        args.test_split
    )
    
    # Create CLIP training samples
    create_training_samples(train_df, args.output_dir, args.sample_size)
    
    print("Done!")


if __name__ == "__main__":
    main()