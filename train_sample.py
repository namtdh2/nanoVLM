import os
import sys
from pathlib import Path
from dataclasses import dataclass
import torch
import wandb
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoImageProcessor

# Add the parent directory to Python path
sys.path.append(str(Path(__file__).parent))

from data.datasets import VQADataset
from data.collators import VQACollator
from models.vision_language_model import VisionLanguageModel
import models.config as cfg
from train import train, get_dataloaders, init_dist, destroy_dist, is_dist, is_master

@dataclass
class VLMConfig:
    vit_hidden_dim: int = 768
    vit_inter_dim: int = 4 * vit_hidden_dim
    vit_patch_size: int = 16
    vit_img_size: int = 224
    vit_n_heads: int = 12
    vit_dropout: float = 0.0
    vit_n_blocks: int = 12
    vit_ln_eps: float = 1e-6
    vit_cls_flag: bool = False
    vit_model_type: str = 'google/siglip-base-patch16-224'

    lm_hidden_dim: int = 576
    lm_inter_dim: int = 1536
    lm_rms_eps: float = 1e-5
    lm_re_base: int = 100000
    lm_max_position_embeddings: int = 8192
    lm_vocab_size: int = 49152
    lm_n_heads: int = 9
    lm_n_kv_heads: int = 3
    lm_dropout: float = 0.0
    lm_n_blocks: int = 30
    lm_attn_scaling: float = 1.0
    lm_eos_token_id: int = 0
    lm_max_length: int = 128 - 49  # Deduct the image token length to achieve a 'nice number'
    lm_use_tokens: bool = False
    lm_tie_weights: bool = True
    lm_model_type: str = 'HuggingFaceTB/SmolLM2-135M'
    lm_tokenizer: str = 'HuggingFaceTB/cosmo2-tokenizer'

    mp_pixel_shuffle_factor: int = 2

    vlm_load_backbone_weights: bool = True
    vlm_checkpoint_path: str = 'checkpoints/nanoVLM-sample'

@dataclass
class TrainConfig:
    lr_mp: float = 1e-3
    lr_backbones: float = 5e-5
    val_ratio: float = 0.2
    compile: bool = False
    data_cutoff_idx: int = 10000  # Use all samples
    batch_size: int = 64  # Small batch size for sample dataset
    gradient_accumulation_steps: int = 1
    mmstar_batch_size: int = 2
    max_grad_norm: float = None
    eval_in_epochs: bool = True
    eval_interval: int = 125
    epochs: int = 20
    compile: bool = False
    resume_from_vlm_checkpoint: bool = False
    # train_dataset_path: str = 'HuggingFaceM4/the_cauldron'
    # train_dataset_name: tuple[str, ...] = ("tqa", "vsr")
    train_dataset_path: str = 'playground/windoor_dataset'
    # train_dataset_path: str = 'playground/sample_dataset'
    train_dataset_name: tuple[str, ...] = ("default",)  # Use default configuration
    test_dataset_path: str = "Lin-Chen/MMStar"
    log_wandb: bool = False  # Disable wandb logging for sample training

def main():
    # Initialize distributed training if needed
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        init_dist()

    # Initialize configurations
    vlm_cfg = VLMConfig()
    train_cfg = TrainConfig()

    # Verify dataset size
    dataset = load_from_disk(train_cfg.train_dataset_path)
    print(f"Dataset size: {len(dataset['train'])} samples")

    # Print configurations
    print("--- VLM Config ---")
    print(vlm_cfg)
    print("--- Train Config ---")
    print(train_cfg)

    # Train the model
    train(train_cfg, vlm_cfg)

    # Clean up distributed training if used
    if is_dist():
        destroy_dist()

if __name__ == "__main__":
    main() 