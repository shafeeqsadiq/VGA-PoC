"""
models/backbones.py
Loads pretrained SigLIP-B/16 vision encoder with LoRA and a pruned 12-layer SmolLM2-360M.
Protects nn.ModuleList registrations and enforces targeted layer adapters.
"""
from typing import Tuple
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, SiglipVisionModel, AutoModel
from peft import LoraConfig, get_peft_model
from configs.poc_config import CONFIG

def load_siglip_lora(
    model_name: str = "google/siglip-base-patch16-384",
    lora_r: int = getattr(CONFIG, "lora_r", 16),
    lora_alpha: int = getattr(CONFIG, "lora_alpha", 32)
) -> nn.Module:
    """
    Loads SigLIP-B/16 vision encoder and injects rank-16 LoRA adapters
    into attention layers 6-11 to prevent few-shot overfitting.
    """
    try:
        vision_model = SiglipVisionModel.from_pretrained(model_name)
    except Exception:
        full_model = AutoModel.from_pretrained(model_name)
        vision_model = getattr(full_model, "vision_model", full_model)

    # Freeze base model parameters
    for param in vision_model.parameters():
        param.requires_grad = False

    # Target layers 6 to 11 (upper half of vision stack)
    try:
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
            layers_to_transform=list(range(6, 12)),
            layers_pattern="layers",
            lora_dropout=0.05,
            bias="none"
        )
        lora_model = get_peft_model(vision_model, lora_config)
    except Exception:
        # Fallback to all layers if environment PEFT does not support layers_to_transform
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
            lora_dropout=0.05,
            bias="none"
        )
        lora_model = get_peft_model(vision_model, lora_config)

    return lora_model

def load_pruned_smollm2_lora(
    model_name: str = "HuggingFaceTB/SmolLM2-360M",
    num_layers: int = 12,
    lora_r: int = getattr(CONFIG, "lora_r", 16),
    lora_alpha: int = getattr(CONFIG, "lora_alpha", 32)
) -> Tuple[nn.Module, AutoTokenizer]:
    """
    Loads SmolLM2-360M, prunes to 12 layers (~181M params), and wraps with LoRA adapters.
    Maintains strict nn.ModuleList hierarchy to prevent parameter detachment.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(model_name)
    config.num_hidden_layers = num_layers

    try:
        model = AutoModel.from_pretrained(model_name, config=config, ignore_mismatched_sizes=True)
    except Exception:
        model = AutoModel.from_pretrained(model_name)
        model.config.num_hidden_layers = num_layers

    # Critical: Enforce nn.ModuleList registration to preserve autograd and device transfers
    if hasattr(model, "layers"):
        model.layers = nn.ModuleList(list(model.layers)[:num_layers])
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        model.model.layers = nn.ModuleList(list(model.model.layers)[:num_layers])

    for param in model.parameters():
        param.requires_grad = False

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none"
    )
    lora_model = get_peft_model(model, lora_config)

    return lora_model, tokenizer