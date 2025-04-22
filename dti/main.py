import os
import torch
from transformers import AdamW, get_cosine_schedule_with_warmup

from config import Config
from model import AffinityPredictor, DrugTargetInteractionLoss
from data_loader import get_data_loaders
from trainer import train_model


def setup_environment():
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.cuda.empty_cache()


def main():
    setup_environment()
    
    config = Config()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_loader, test_loader = get_data_loaders(config)
    print("Data loaders created")
    
    model = AffinityPredictor(
        protein_model_name=config.protein_model_name,
        molecule_model_name=config.molecule_model_name,
        hidden_sizes=config.hidden_sizes,
        dropout=config.dropout
    ).to(device)
    
    try:
        print("Loading checkpoint...")
        model.load_state_dict(torch.load(config.model_save_path))
        print("Checkpoint loaded successfully")
    except FileNotFoundError:
        print("No checkpoint found, starting from scratch")
    
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.lr,
        weight_decay=config.weight_decay
    )
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.num_warmup_steps,
        num_training_steps=config.num_training_steps
    )
    
    loss_fn = DrugTargetInteractionLoss()
    
    train_losses, val_losses = train_model(
        model,
        train_loader,
        test_loader,
        optimizer,
        scheduler,
        loss_fn,
        device,
        config
    )
    
    print("Training completed")


if __name__ == "__main__":
    main()