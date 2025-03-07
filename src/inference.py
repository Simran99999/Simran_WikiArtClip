import os
import argparse
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import open_clip
from open_clip import tokenize

# Art-specific vocabulary to enhance captions
ART_STYLES = [
    "Impressionism", "Expressionism", "Cubism", "Surrealism", "Abstract", 
    "Renaissance", "Baroque", "Neoclassicism", "Romanticism", "Realism",
    "Art Nouveau", "Art Deco", "Pop Art", "Minimalism", "Ukiyo-e",
    "Fauvism", "Pointillism", "Rococo", "Gothic", "Naive Art",
    "Symbolism", "Post-Impressionism", "Magic Realism"
]

ART_TECHNIQUES = [
    "oil painting", "watercolor", "acrylic", "charcoal", "pastel",
    "sketch", "etching", "lithograph", "woodcut", "fresco",
    "tempera", "collage", "mixed media", "ink wash", "gouache",
    "digital art", "pencil drawing", "linocut", "engraving"
]

ART_SUBJECTS = [
    "landscape", "portrait", "still life", "abstract composition", "religious scene",
    "historical event", "mythology", "allegory", "genre painting", "self-portrait",
    "cityscape", "seascape", "figure study", "interior scene", "battle scene",
    "narrative", "symbolic imagery", "architectural study"
]

class ArtImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        """
        Dataset for inference on art images
        
        Args:
            image_paths: List of image paths
            transform: Optional image transform
        """
        self.image_paths = image_paths
        
        # Default transform if none provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                                     (0.26862954, 0.26130258, 0.27577711))
            ])
        else:
            self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = torch.zeros((3, 224, 224))
            
        return image, img_path


def generate_candidate_descriptions():
    """Generate a list of art-specific candidate descriptions"""
    candidates = []
    
    # Basic structure templates
    templates = [
        "a {technique} of {subject}",
        "a {style} painting of {subject}",
        "a {style} {technique} depicting {subject}",
        "{subject} in {style} style",
        "an artwork depicting {subject} in the style of {style}",
        "a {technique} showing {subject} with {style} elements",
        "{subject} rendered in {technique}",
        "a {style} interpretation of {subject}",
        "a {technique} with {style} influences showing {subject}"
    ]
    
    # Generate descriptions using templates and vocabulary
    for template in templates:
        for style in ART_STYLES:
            for technique in ART_TECHNIQUES:
                for subject in ART_SUBJECTS:
                    description = template.format(
                        style=style.lower(),
                        technique=technique,
                        subject=subject
                    )
                    candidates.append(description)
    
    # Add some more specific descriptions
    additional_descriptions = [
        "a painting with vibrant colors and bold brushstrokes",
        "a monochromatic artwork with strong contrast",
        "a detailed painting with fine brushwork",
        "an artwork with geometric patterns and shapes",
        "a painting with dramatic lighting and shadows",
        "an artwork with dreamlike imagery and symbolism",
        "a painting with layered textures and mixed media",
        "an artwork with traditional motifs and symbolism",
        "a painting with atmospheric perspective and depth",
        "an artwork with stylized figures and exaggerated forms",
        "a painting with delicate lines and subtle color transitions",
        "an abstract artwork with emotional expression",
        "a painting with narrative elements and storytelling",
        "an artwork with cultural and historical references",
        "a painting with meticulous attention to detail",
        "an artwork with spiritual and mystical themes",
        "a painting with rhythmic composition and movement"
    ]
    
    candidates.extend(additional_descriptions)
    
    # Remove duplicates and limit the total number of candidates
    candidates = list(set(candidates))
    
    print(f"Generated {len(candidates)} candidate descriptions")
    return candidates


@torch.no_grad()
def generate_captions(model, image_paths, device, batch_size=32, top_k=5):
    """
    Generate captions for images using a fine-tuned CLIP model
    
    Args:
        model: Fine-tuned CLIP model
        image_paths: List of image paths
        device: Device to run inference on
        batch_size: Batch size for inference
        top_k: Number of top captions to return
        
    Returns:
        Dictionary mapping image paths to captions
    """
    model.eval()
    
    # Create dataset and dataloader
    _, _, preprocess = open_clip.create_model_and_transforms(
        model.model_name,
        pretrained=False
    )
    
    dataset = ArtImageDataset(image_paths, transform=preprocess)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Generate candidate descriptions
    candidates = generate_candidate_descriptions()
    
    # Tokenize all candidates at once
    candidate_tokens = tokenize(candidates).to(device)
    
    # Encode all candidate texts
    text_features_list = []
    for i in range(0, len(candidates), batch_size):
        batch_tokens = candidate_tokens[i:i+batch_size]
        with torch.no_grad():
            batch_features = model.encode_text(batch_tokens)
            batch_features = F.normalize(batch_features, dim=1)
        text_features_list.append(batch_features)
    
    # Concatenate all text features
    text_features = torch.cat(text_features_list)
    
    # Generate captions for all images
    results = {}
    
    for images, paths in tqdm(dataloader, desc="Generating captions"):
        images = images.to(device)
        
        # Encode images
        image_features = model.encode_image(images)
        image_features = F.normalize(image_features, dim=1)
        
        # Calculate similarity between images and all candidate texts
        similarity = image_features @ text_features.T
        
        # Get top-k captions for each image
        top_indices = similarity.topk(top_k, dim=1).indices
        
        # Store results
        for i, path in enumerate(paths):
            top_captions = [candidates[idx.item()] for idx in top_indices[i]]
            top_scores = [similarity[i, idx].item() for idx in top_indices[i]]
            
            results[path] = {
                'captions': top_captions,
                'scores': top_scores
            }
    
    return results


def visualize_results(image_path, caption_results, output_dir, orig_caption=None):
    """
    Visualize an image with its original and generated captions
    
    Args:
        image_path: Path to the image
        caption_results: Results from generate_captions for this image
        output_dir: Directory to save the visualization
        orig_caption: Original caption if available
    """
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load and display the image
        img = Image.open(image_path).convert('RGB')
        plt.figure(figsize=(10, 8))
        plt.imshow(img)
        plt.axis('off')
        
        # Create caption text
        caption_text = f"File: {os.path.basename(image_path)}\n\n"
        
        if orig_caption:
            caption_text += f"Original Caption: {orig_caption}\n\n"
        
        caption_text += "Generated Captions:\n"
        for i, (caption, score) in enumerate(zip(
            caption_results['captions'], 
            caption_results['scores']
        )):
            caption_text += f"{i+1}. {caption} (score: {score:.4f})\n"
        
        plt.figtext(0.5, 0.01, caption_text, wrap=True, horizontalalignment='center')
        
        # Save the visualization
        output_path = os.path.join(
            output_dir, 
            f"{os.path.splitext(os.path.basename(image_path))[0]}_captioned.jpg"
        )
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        
    except Exception as e:
        print(f"Error visualizing results for {image_path}: {e}")