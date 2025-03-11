# train.py

import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import wandb  # Ensure wandb is installed: pip install wandb

from dataloader import create_dataloaders
from model import UNet  # Your modified ResNet-based U-Net from model.py


def get_loss_fn(loss_type):
    """
    Returns the loss function based on loss_type.
    Options: 'mse', 'l1', 'mse+l1'
    """
    if loss_type == "mse":
        return nn.MSELoss()
    elif loss_type == "l1":
        return nn.L1Loss()
    elif loss_type == "mse+l1":
        mse = nn.MSELoss()
        l1 = nn.L1Loss()
        return lambda output, target: 0.5 * mse(output, target) + 0.5 * l1(output, target)
    else:
        raise ValueError("Invalid loss type. Choose from 'mse', 'l1', 'mse+l1'.")


def train_model(model, train_loader, val_loader, device, epochs, lr, loss_type, weight_decay):
    """
    Trains the U-Net model on the rotation task.
    """
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = get_loss_fn(loss_type)
    model = model.to(device)

    best_val_loss = float('inf')
    best_checkpoint_path = os.path.join(wandb.run.dir, "best_checkpoint.pth")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for inputs, targets, _ in train_loader:
            inputs = inputs.to(device)    # (B, 3, 32, 32)
            targets = targets.to(device)  # (B, 3, 32, 32)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        val_loss = validate_model(model, val_loader, device, criterion)

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_checkpoint_path)
            wandb.run.summary["best_val_loss"] = best_val_loss

        wandb.log({
            "epoch": epoch,
            "train_loss": epoch_loss,
            "val_loss": val_loss,
            "lr": optimizer.param_groups[0]['lr']
        })
        print(f"Epoch [{epoch}/{epochs}], Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Visualize predictions every 20 epochs
        if epoch % 20 == 0:
            visualize_predictions(model, val_loader, device, num_samples=8)

    wandb.log({"final_val_loss": best_val_loss})
    wandb.save(best_checkpoint_path)
    return best_val_loss, best_checkpoint_path


def validate_model(model, val_loader, device, criterion):
    """
    Validates the model on the given data loader.
    """
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets, _ in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item() * inputs.size(0)
    val_loss /= len(val_loader.dataset)
    return val_loss


def visualize_predictions(model, loader, device, num_samples=8):
    """
    Logs a grid of input, target, and predicted images to wandb.
    """
    import torchvision.utils as vutils

    model.eval()
    inputs, targets, _ = next(iter(loader))
    inputs = inputs.to(device)
    with torch.no_grad():
        preds = model(inputs)
    inputs = inputs.cpu()
    preds = preds.cpu()
    targets = targets.cpu()

    inputs_grid = vutils.make_grid(inputs[:num_samples], nrow=num_samples, normalize=True, scale_each=True)
    preds_grid = vutils.make_grid(preds[:num_samples], nrow=num_samples, normalize=True, scale_each=True)
    targets_grid = vutils.make_grid(targets[:num_samples], nrow=num_samples, normalize=True, scale_each=True)

    wandb.log({
        "inputs": [wandb.Image(inputs_grid, caption="Input Images")],
        "predictions": [wandb.Image(preds_grid, caption="Model Predictions")],
        "targets": [wandb.Image(targets_grid, caption="Target Images")]
    })

def none_or_int(value):
    if value.lower() == "none":
        return None
    return int(value)


def main():
    parser = argparse.ArgumentParser(description="Train U-Net on CIFAR rotation task with flexible experiments.")
    parser.add_argument("--project", type=str, default="cifar-rotation", help="W&B project name.")
    parser.add_argument("--entity", type=str, default=None, help="W&B entity (team or username).")
    parser.add_argument("--epochs", type=int, default=300, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for AdamW.")
    parser.add_argument("--loss", type=str, default="mse", choices=["mse", "l1", "mse+l1"], help="Loss function to use.")
    parser.add_argument(
        "--max_samples",
        type=none_or_int,
        default=None,
        help="Max samples to load for train/val/test. Use 'None' to load the entire dataset."
    )
    parser.add_argument("--dataset_dir", type=str, default="/pub0/smnair/stat946/shape-rotator/dataset",
                        help="Path to the dataset folder containing CSVs.")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of dataloader workers.")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use.")
    args = parser.parse_args()

    # Initialize wandb
    wandb.init(
        project=args.project,
        entity=args.entity,
        config=vars(args),
        name=f"exp_gpu_{args.gpu_id}_{args.loss}",
        reinit=True
    )

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Create dataloaders (the entire dataset is now in memory)
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset_dir=args.dataset_dir,
        batch_size=args.batch_size,
        val_split=0.1,
        num_workers=args.num_workers,
        max_samples=args.max_samples
    )

    # Instantiate the model
    model = UNet(n_channels=3, n_classes=3)

    # Train the model and get best checkpoint path
    best_val_loss, best_checkpoint_path = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        loss_type=args.loss,
        weight_decay=args.weight_decay
    )

    print("Training complete!")
    print(f"Final Validation Loss: {best_val_loss:.4f}")
    wandb.log({"final_val_loss": best_val_loss})

    # Load the best checkpoint before final visualization
    print("Loading best checkpoint for visualization...")
    model.load_state_dict(torch.load(best_checkpoint_path, map_location=device))

    # Final visualization using the best checkpoint
    visualize_predictions(model, val_loader, device, num_samples=8)

    wandb.finish()


if __name__ == "__main__":
    main()
