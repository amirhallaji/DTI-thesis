import sys
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from lifelines.utils import concordance_index


def train_one_epoch(model, dataloader, optimizer, scheduler, loss_fn, device, accumulation_steps):
    model.train()
    total_loss = 0
    scaler = GradScaler()
    optimizer.zero_grad()
    
    for i, batch in enumerate(tqdm(dataloader, desc="Training", leave=False)):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        true_labels = batch["label"]
        
        with autocast(dtype=torch.float16):
            pred = model(batch)
            loss = loss_fn(pred.view(-1), true_labels)
            
        scaler.scale(loss).backward()
        
        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(dataloader):
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            
        total_loss += loss.item() * accumulation_steps
        
    return total_loss / len(dataloader)


def validate_one_epoch(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0
    all_labels = []
    all_preds = []
    
    with autocast():
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation", leave=False):
                batch = {k: v.to(device) for k, v in batch.items()}
                pred = model(batch)
                loss = F.mse_loss(pred.view(-1), batch["label"])
                total_loss += loss.item()
                all_preds.extend(pred.view(-1).cpu().numpy())
                all_labels.extend(batch["label"].cpu().numpy())
                torch.cuda.empty_cache()
                
    return total_loss / len(dataloader), concordance_index(all_labels, all_preds)


def save_model(model, path):
    torch.save(model.state_dict(), path)


def train_model(model, train_loader, val_loader, optimizer, scheduler, loss_fn, device, config):
    log_file = open(config.train_log_file, "a")
    original_stdout = sys.stdout
    sys.stdout = log_file
    
    best_loss = float("inf")
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    for epoch in range(config.epochs):
        print(f"Epoch {epoch + 1}")
        
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, loss_fn, device, config.accumulation_steps
        )
        train_losses.append(train_loss)
        print(f"Train Loss: {train_loss:.4f}")
        
        val_loss, c_index = validate_one_epoch(model, val_loader, loss_fn, device)
        val_losses.append(val_loss)
        print(f"Validation Loss: {val_loss:.4f} | CI score: {c_index:.4f}")
        
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            save_model(model, config.model_save_path)
            print(f"Best model saved at epoch {epoch + 1}")
        else:
            patience_counter += 1
            print(f"Did not improve. Patience: {patience_counter}")
            if patience_counter >= config.patience:
                print("Early stopping triggered.")
                break
                
    save_model(model, config.final_model_path)
    print("Final model saved.")
    
    sys.stdout = original_stdout
    log_file.close()
    
    return train_losses, val_losses