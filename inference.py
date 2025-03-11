# inference.py

import os
import argparse
import numpy as np
import pandas as pd
import torch

from dataloader import create_dataloaders
from model import UNet  # Your U-Net (or ResNet-based U-Net) from model.py

def none_or_int(value):
    """
    Helper function to allow 'None' or integer for max_samples.
    """
    if value.lower() == "none":
        return None
    return int(value)

def main():
    parser = argparse.ArgumentParser(description="Generate predictions on the test set and save as CSV.")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to the best checkpoint (.pth file).")
    parser.add_argument("--dataset_dir", type=str, default="/pub0/smnair/stat946/shape-rotator/dataset",
                        help="Path to the dataset folder containing CSVs.")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size for inference.")
    parser.add_argument("--num_workers", type=int, default=2,
                        help="Number of workers for DataLoader.")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="GPU ID to use for inference.")
    parser.add_argument("--max_samples", type=none_or_int, default=None,
                        help="Max samples to load for test. Use 'None' for all.")
    args = parser.parse_args()

    # Decide which device to use
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create the test loader (we only need test_loader)
    # We can ignore train_loader, val_loader
    _, _, test_loader = create_dataloaders(
        dataset_dir=args.dataset_dir,
        batch_size=args.batch_size,
        val_split=0.1,
        num_workers=args.num_workers,
        max_samples=args.max_samples
    )

    # Instantiate the model and load checkpoint
    model = UNet(n_channels=3, n_classes=3).to(device)
    print(f"Loading checkpoint from {args.checkpoint_path}...")
    model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
    model.eval()

    # Prepare lists to store predictions and IDs
    all_preds = []
    all_ids = []

    # Inference loop
    with torch.no_grad():
        for inputs, batch_ids in test_loader:
            inputs = inputs.to(device)  # shape: (B, 3, 32, 32)
            outputs = model(inputs)     # shape: (B, 3, 32, 32)
            outputs = outputs.cpu().numpy()

            # Flatten each image from (3,32,32) to (3072,)
            # so we can store them as rows in a CSV
            B = outputs.shape[0]
            outputs_flat = outputs.reshape(B, -1)  # (B, 3072)

            all_preds.append(outputs_flat)
            all_ids.append(batch_ids.numpy())  # shape: (B,)

    # Concatenate all predictions and IDs
    all_preds = np.concatenate(all_preds, axis=0)  # shape: (N, 3072)
    all_ids = np.concatenate(all_ids, axis=0)      # shape: (N,)

    # Convert to DataFrame
    # The original input CSV format is: "ID" + 3072 pixel columns
    # We'll name them pixel_1, pixel_2, ..., pixel_3072
    num_pixels = 3 * 32 * 32
    columns = ["ID"] + [f"p{i}" for i in range(1, num_pixels + 1)]

    df = pd.DataFrame(
        data=np.column_stack([all_ids, all_preds]),
        columns=columns
    )

    # Save to CSV in the same folder as the checkpoint
    out_dir = os.path.dirname(args.checkpoint_path)
    out_csv_path = os.path.join(out_dir, "test_dataset_predicted_images.csv")
    df.to_csv(out_csv_path, index=False)
    print(f"Predictions saved to {out_csv_path}")

if __name__ == "__main__":
    main()
