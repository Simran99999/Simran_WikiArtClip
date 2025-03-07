# helpers.py

import os
import json
import logging
import torch
import random
import numpy as np
from pathlib import Path
from typing import Dict, Any

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a JSON file.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        Dict[str, Any]: Loaded configuration as a dictionary.
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def save_config(config: Dict[str, Any], config_path: str):
    """
    Save configuration to a JSON file.

    Args:
        config (Dict[str, Any]): Configuration dictionary to save.
        config_path (str): Path to save the configuration file.
    """
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

def setup_logging(log_dir: str, log_level: str = "INFO"):
    """
    Set up logging to file and console.

    Args:
        log_dir (str): Directory to save log files.
        log_level (str): Logging level (e.g., "INFO", "DEBUG").
    """
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "training.log")
    
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def get_device(use_cpu: bool = False) -> torch.device:
    """
    Get the appropriate device (CPU or GPU) for training.

    Args:
        use_cpu (bool): Whether to force CPU usage.

    Returns:
        torch.device: The device to use.
    """
    if use_cpu or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device("cuda")

def set_seed(seed: int = 42):
    """
    Set the random seed for reproducibility.

    Args:
        seed (int): Random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_output_dir(output_dir: str):
    """
    Create the output directory if it doesn't exist.

    Args:
        output_dir (str): Path to the output directory.
    """
    os.makedirs(output_dir, exist_ok=True)

def save_checkpoint(state: Dict[str, Any], filename: str):
    """
    Save a model checkpoint.

    Args:
        state (Dict[str, Any]): State dictionary containing model and optimizer states.
        filename (str): Path to save the checkpoint.
    """
    torch.save(state, filename)
    logging.info(f"Checkpoint saved to {filename}")

def load_checkpoint(filename: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer = None):
    """
    Load a model checkpoint.

    Args:
        filename (str): Path to the checkpoint file.
        model (torch.nn.Module): Model to load the state into.
        optimizer (torch.optim.Optimizer, optional): Optimizer to load the state into.

    Returns:
        int: The epoch at which the checkpoint was saved.
    """
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    logging.info(f"Checkpoint loaded from {filename}")
    return checkpoint.get('epoch', 0)

def log_metrics(metrics: Dict[str, Any], epoch: int, prefix: str = ""):
    """
    Log training or evaluation metrics.

    Args:
        metrics (Dict[str, Any]): Dictionary of metrics to log.
        epoch (int): Current epoch.
        prefix (str, optional): Prefix to add to the metric names.
    """
    log_str = f"Epoch {epoch}: "
    for key, value in metrics.items():
        log_str += f"{prefix}{key}: {value:.4f} "
    logging.info(log_str)