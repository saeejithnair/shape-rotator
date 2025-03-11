# dataloader.py
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm

class CIFARRotationDataset(Dataset):
    """
    Dataset for CIFAR rotation task.

    This version preloads the entire dataset into memory.

    Args:
        input_csv_path: Path to the input images CSV file
        output_csv_path: Path to the output images CSV file (optional, for training data)
        transform: PyTorch transforms to apply to the images
        is_test: Whether this is test data (no output images)
        max_samples: Maximum number of samples to load (None for all)
    """
    def __init__(self, input_csv_path, output_csv_path=None, transform=None, is_test=False, max_samples=None):
        self.transform = transform
        self.is_test = is_test

        # Load input CSV into memory
        print(f"Loading input images from {input_csv_path}...")
        input_df = pd.read_csv(input_csv_path)
        
        # Optionally limit the number of samples
        if max_samples is not None and max_samples < len(input_df):
            input_df = input_df.sample(max_samples, random_state=42)
        
        # Extract IDs for reference
        self.ids = input_df['ID'].values
        
        # Preload and convert input images to tensor
        # Assumes the CSV columns are: "ID", then pixel values
        input_array = input_df.iloc[:, 1:].values.astype(np.float32)
        self.input_images = torch.from_numpy(input_array.reshape(-1, 3, 32, 32))
        
        # Preload output images if not test data
        if not is_test and output_csv_path is not None:
            print(f"Loading output images from {output_csv_path}...")
            output_df = pd.read_csv(output_csv_path)
            # Filter output_df to include only rows with IDs in our input
            output_df = output_df[output_df['ID'].isin(self.ids)]
            # Ensure alignment by reindexing using the same IDs order
            output_df = output_df.set_index('ID').loc[self.ids].reset_index()
            output_array = output_df.iloc[:, 1:].values.astype(np.float32)
            self.output_images = torch.from_numpy(output_array.reshape(-1, 3, 32, 32))
        else:
            self.output_images = None

        print(f"Loaded {len(self.input_images)} samples into memory.")

    def __len__(self):
        return len(self.input_images)
    
    def __getitem__(self, idx):
        # Retrieve input tensor
        input_tensor = self.input_images[idx]
        if self.transform:
            input_tensor = self.transform(input_tensor)
        
        if self.is_test:
            return input_tensor, self.ids[idx]
        
        # Retrieve target tensor for training/validation
        output_tensor = self.output_images[idx]
        if self.transform:
            output_tensor = self.transform(output_tensor)
        
        return input_tensor, output_tensor, self.ids[idx]


def create_dataloaders(dataset_dir, batch_size=32, val_split=0.1, num_workers=4, max_samples=None):
    """
    Create train, validation, and test dataloaders.
    
    Args:
        dataset_dir: Directory containing the CSV files
        batch_size: Batch size for the dataloaders
        val_split: Fraction of training data to use for validation
        num_workers: Number of workers for the dataloaders
        max_samples: Maximum number of samples to load (None for all)
        
    Returns:
        train_loader, val_loader, test_loader
    """
    # Define file paths
    train_input_path = os.path.join(dataset_dir, "train_dataset_input_images.csv")
    train_output_path = os.path.join(dataset_dir, "train_dataset_output_images.csv")
    test_input_path = os.path.join(dataset_dir, "test_dataset_input_images.csv")
    
    # Define transforms (minimal for this task)
    transform = transforms.Compose([
        # Add any additional transforms if needed
    ])
    
    # Create the full training dataset (preloaded in memory)
    full_train_dataset = CIFARRotationDataset(
        input_csv_path=train_input_path,
        output_csv_path=train_output_path,
        transform=transform,
        is_test=False,
        max_samples=max_samples
    )
    
    # Split into train and validation sets
    val_size = int(len(full_train_dataset) * val_split)
    train_size = len(full_train_dataset) - val_size
    
    train_dataset, val_dataset = random_split(
        full_train_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )
    
    # Create test dataset (preloaded in memory)
    test_dataset = CIFARRotationDataset(
        input_csv_path=test_input_path,
        transform=transform,
        is_test=True,
        max_samples=max_samples
    )
    
    # Create DataLoaders with prefetch_factor for improved performance (PyTorch 1.7+)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2
    )
    
    print(f"Created dataloaders with {len(train_dataset)} training samples, "
          f"{len(val_dataset)} validation samples, and {len(test_dataset)} test samples.")
    
    return train_loader, val_loader, test_loader

# Example usage
if __name__ == "__main__":
    # Create dataloaders with a small subset for quick testing
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset_dir="dataset",
        batch_size=32,
        val_split=0.1,
        num_workers=2,
        max_samples=1000  # Limit to 1000 samples for quick testing
    )
    
    # Test the train loader
    print("\nTesting train loader...")
    for inputs, targets, ids in tqdm(train_loader, total=len(train_loader)):
        pass
    print(f"Train batch shape - Input: {inputs.shape}, Target: {targets.shape}")
    
    # Test the validation loader
    print("\nTesting validation loader...")
    for inputs, targets, ids in tqdm(val_loader, total=len(val_loader)):
        pass
    print(f"Validation batch shape - Input: {inputs.shape}, Target: {targets.shape}")
    
    # Test the test loader
    print("\nTesting test loader...")
    for inputs, ids in tqdm(test_loader, total=len(test_loader)):
        pass
    print(f"Test batch shape - Input: {inputs.shape}")
    
    print("\nDataloaders working correctly!")
