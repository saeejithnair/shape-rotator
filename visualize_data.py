#%% 
import os
import matplotlib.pyplot as plt
import torch
import sys
import os

from dataloader import create_dataloaders, CIFARRotationDataset
#%%
def visualize_batch(inputs, outputs=None, ids=None, title="Batch Visualization"):
    """
    Visualize a batch of input (and optionally output) images side-by-side.
    
    Args:
        inputs: A batch of input images as a tensor of shape (B, 3, 32, 32).
        outputs: (Optional) A batch of output images of shape (B, 3, 32, 32).
        ids: (Optional) A list of IDs corresponding to each sample in the batch.
        title: Title for the figure.
    """
    # We'll show up to 8 images from the batch to avoid overly large figures
    max_images = min(inputs.size(0), 8)
    
    # If we have both inputs and outputs, we create 2 rows in our subplot
    rows = 2 if outputs is not None else 1
    fig, axes = plt.subplots(rows, max_images, figsize=(max_images * 2, 4))
    if rows == 1:
        axes = [axes]  # Make the indexing consistent
    
    for i in range(max_images):
        # Convert the input image from (3, 32, 32) to (32, 32, 3) for plotting
        inp = inputs[i].permute(1, 2, 0).numpy()
        axes[0][i].imshow(inp, cmap="gray" if inp.shape[2] == 1 else None)
        axes[0][i].axis("off")
        if ids is not None:
            axes[0][i].set_title(f"ID: {ids[i].item() if torch.is_tensor(ids[i]) else ids[i]}")

        # If we have outputs, show them in the second row
        if outputs is not None:
            out = outputs[i].permute(1, 2, 0).numpy()
            axes[1][i].imshow(out, cmap="gray" if out.shape[2] == 1 else None)
            axes[1][i].axis("off")
    
    fig.suptitle(title)
    plt.show()

#%%
def main():
    # Create the dataloaders (adjust dataset_dir, batch_size, etc. as needed)
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset_dir="/pub0/smnair/stat946/shape-rotator/dataset",
        batch_size=8,        # Number of samples per batch
        val_split=0.1,       # 10% of the data for validation
        num_workers=2,       # Number of workers for data loading
        max_samples=64       # Limit total samples for quick testing
    )
    
    # --- Visualize a single batch from the training set ---
    train_batch = next(iter(train_loader))
    inputs, outputs, ids = train_batch
    visualize_batch(inputs, outputs, ids, title="Train Batch")
    
    # --- Visualize a single batch from the validation set ---
    val_batch = next(iter(val_loader))
    inputs, outputs, ids = val_batch
    visualize_batch(inputs, outputs, ids, title="Validation Batch")
    
    # --- Visualize a single batch from the test set ---
    test_batch = next(iter(test_loader))
    # Test set only returns (input_tensor, ids)
    inputs, ids = test_batch
    visualize_batch(inputs, outputs=None, ids=ids, title="Test Batch")

#%%
main()
j# %%
