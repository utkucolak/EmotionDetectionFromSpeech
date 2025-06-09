from model import build_transformer
from config import get_config, get_weights_file_path

import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from utils.dataset import RAVDESSDataset
from pathlib import Path
import torch
import torch.nn as nn

from tqdm import tqdm
import warnings

from utils.visualize import *

config = get_config()

def get_dataloaders(csv_path: str, batch_size: int = 16, max_len: int = 300):
    df = pd.read_csv(csv_path)
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["emotion"], random_state=42)
    
    train_df.to_csv("train_temp.csv", index=False)
    val_df.to_csv("val_temp.csv", index=False)

    train_dataset = RAVDESSDataset("train_temp.csv", max_len=max_len)
    val_dataset = RAVDESSDataset("val_temp.csv", max_len=max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    return train_loader, val_loader

train_loader, val_loader = get_dataloaders(config["csv_path"],
                                           batch_size=config["batch_size"],
                                           max_len=config["max_len"])

def get_model(config, time_steps):
    return build_transformer(time_steps, config["d_model"])

def train_model(config):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    best_val_accuracy = 0.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = get_dataloaders(
        config["csv_path"], 
        batch_size=config["batch_size"], 
        max_len=config["max_len"]
    )

    model = get_model(config, config["max_len"])
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], eps=1e-9)

    initial_epoch = 0
    global_step = 0

    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f"Preloading model {model_filename}")

        state = torch.load(model_filename)
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state["optimizer_state_dict"])
        global_step = state["global_step"]

    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)
    
    for epoch in range(initial_epoch, config["num_epochs"]):
        model.train()

        total_train_loss = 0
        correct_train = 0
        total_train = 0

        batch_iterator = tqdm(train_loader, desc=f"Epoch {epoch:02d}")

        for mel, labels in batch_iterator:
            mel = mel.to(device)           # (B, T, n_mels)
            labels = labels.to(device)     # (B,)

            output = model(mel)            # (B, num_classes)
            loss = loss_fn(output, labels)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1
            batch_iterator.set_postfix(loss=loss.item())

            total_train_loss += loss.item() * mel.size(0)  # sum over batch
            _, preds = torch.max(output, dim=1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)
        
        model_filename = get_weights_file_path(config, f'{epoch:02d}')
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        train_accuracies.append(correct_train / total_train)
        # Validation

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        

        with torch.no_grad():
            for encoder_input, labels in val_loader:
                encoder_input = encoder_input.to(device)
                labels = labels.to(device)

                outputs = model(encoder_input)
                loss = loss_fn(outputs, labels)

                val_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = correct / total
        print(f"Epoch {epoch} validation loss: {avg_val_loss:.4f}, accuracy: {val_accuracy:.4f}")
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)
        if epoch == 0 or val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            save_path = get_weights_file_path(config, f"best")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "global_step": global_step
            }, save_path)
        if (epoch) % 500 == 0:
            torch.save({
                "epoch": epoch+1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "global_step": global_step
            }, model_filename)
            print(f"Checkpoint saved at epoch {epoch} → {model_filename}")
    plot_training_curves(train_losses, 
                         val_losses, 
                         train_accuracies, 
                         val_accuracies)
if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    config = get_config()
    train_model(config)