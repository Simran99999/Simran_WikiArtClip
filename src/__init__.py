# src/__init__.py

# Import preprocessing functions
from src.preprocessing_script import (
    WikiArtCLIPDataset,
    preprocess_wikiart_dataset,
    create_training_samples
)

# Import training functions
from src.finetuning_script import (
    WikiArtDataset,
    CustomClipLoss,
    train_one_epoch,
    evaluate,
    fine_tune_clip
)

# Import helper functions
from src.helpers import (
    load_config,
    save_config,
    setup_logging,
    get_device,
    set_seed,
    create_output_dir,
    save_checkpoint,
    load_checkpoint,
    log_metrics
)

# Package version
__version__ = "0.1.0"