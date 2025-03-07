import os
import argparse
import json
import math
import random
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image

import open_clip
from open_clip import tokenize

# Custom CLIP loss function (since we can't import CLIPLoss directly)
class CustomClipLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, image_features, text_features, logit_scale):
        # Normalize features
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        # Compute cosine similarity
        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text = logits_per_image.T
        
        batch_size = image_features.shape[0]
        labels = torch.arange(batch_size, device=image_features.device)
        
        # Compute loss
        loss_img = F.cross_entropy(logits_per_image, labels)
        loss_txt = F.cross_entropy(logits_per_text, labels)
        loss = (loss_img + loss_txt) / 2
        
        return loss


class WikiArtDataset(Dataset):
    def __init__(self, csv_file, root_dir=None, transform=None):
        """
        Args:
            csv_file: Path to the CSV file with image paths and captions
            root_dir: Optional root directory to prepend to image paths
            transform: Optional transforms to apply to images
        """
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        
        # Use enhanced captions if available, otherwise use original captions
        self.caption_col = 'Enhanced_Caption' if 'Enhanced_Caption' in self.data.columns else 'Caption'
        
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
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_path = self.data.iloc[idx]['Image_Path']
        caption = self.data.iloc[idx][self.caption_col]
        
        # Handle relative or absolute paths
        if self.root_dir and not os.path.isabs(img_path):
            img_path = os.path.join(self.root_dir, img_path)
        
        try:
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return placeholder on error
            image = torch.zeros((3, 224, 224))
        
        return image, caption, img_path


def train_one_epoch(model, data_loader, optimizer, scheduler, device, epoch, args):
    model.train()
    loss_fn = CustomClipLoss()
    
    total_loss = 0
    total_samples = 0
    
    progress = tqdm(data_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
    
    for batch_idx, (images, texts, _) in enumerate(progress):
        images = images.to(device)
        text_tokens = tokenize(texts).to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        image_features, text_features, logit_scale = model(images, text_tokens)
        loss = loss_fn(image_features, text_features, logit_scale)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        
        # Update progress bar
        progress.set_postfix({
            'loss': loss.item(),
            'avg_loss': total_loss / total_samples,
            'lr': optimizer.param_groups[0]['lr']
        })
        
        # Free up memory
        del images, text_tokens, image_features, text_features
        torch.cuda.empty_cache()
    
    scheduler.step()
    avg_loss = total_loss / total_samples
    
    return avg_loss


@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()
    loss_fn = CustomClipLoss()
    
    total_loss = 0
    total_samples = 0
    all_image_features = []
    all_text_features = []
    all_image_paths = []
    all_texts = []
    
    for images, texts, img_paths in tqdm(data_loader, desc="Evaluating"):
        images = images.to(device)
        text_tokens = tokenize(texts).to(device)
        
        # Forward pass
        image_features, text_features, logit_scale = model(images, text_tokens)
        
        # Save features for later analysis
        all_image_features.append(image_features.cpu())
        all_text_features.append(text_features.cpu())
        all_image_paths.extend(img_paths)
        all_texts.extend(texts)
        
        # Compute loss
        loss = loss_fn(image_features, text_features, logit_scale)
        
        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        
        # Free up memory
        del images, text_tokens, image_features, text_features
        torch.cuda.empty_cache()
    
    # Concatenate all features
    all_image_features = torch.cat(all_image_features)
    all_text_features = torch.cat(all_text_features)
    
    # Compute metrics like accuracy
    avg_loss = total_loss / total_samples
    
    return avg_loss, {
        'image_features': all_image_features,
        'text_features': all_text_features,
        'image_paths': all_image_paths,
        'texts': all_texts
    }


def fine_tune_clip(args):
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print(f"Loading model: {args.model}")
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model,
        pretrained=args.pretrained
    )
    model = model.to(device)
    
    # Create datasets and dataloaders
    print("Creating datasets...")
    train_dataset = WikiArtDataset(
        args.train_csv,
        root_dir=args.img_root_dir,
        transform=preprocess
    )
    
    val_dataset = WikiArtDataset(
        args.val_csv,
        root_dir=args.img_root_dir,
        transform=preprocess
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Set up optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.min_lr
    )
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        # Train
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            epoch,
            args
        )
        
        # Evaluate
        val_loss, val_data = evaluate(model, val_loader, device)
        
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        
        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }
            torch.save(
                checkpoint,
                os.path.join(args.output_dir, f"wikiart_clip_best.pt")
            )
            print(f"  Saved new best model checkpoint!")
        
        # Save latest model
        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }
            torch.save(
                checkpoint,
                os.path.join(args.output_dir, f"wikiart_clip_epoch_{epoch+1}.pt")
            )
    
    print("Training complete!")
    return model


def main():
    parser = argparse.ArgumentParser(description='Fine-tune CLIP model on WikiArt dataset')
    
    # Data arguments
    parser.add_argument('--train_csv', type=str, required=True, help='Path to training CSV file')
    parser.add_argument('--val_csv', type=str, required=True, help='Path to validation CSV file')
    parser.add_argument('--img_root_dir', type=str, default=None, help='Root directory for images')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for checkpoints')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='ViT-B-32', help='CLIP model architecture')
    parser.add_argument('--pretrained', type=str, default='openai', help='Pretrained model source')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='Minimum learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.1, help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of dataloader workers')
    parser.add_argument('--save_every', type=int, default=1, help='Save checkpoint every N epochs')
    parser.add_argument('--cpu', action='store_true', help='Force CPU usage')
    
    args = parser.parse_args()
    
    # Fine-tune the model
    model = fine_tune_clip(args)
    
    print("Done!")


if __name__ == "__main__":
    main()