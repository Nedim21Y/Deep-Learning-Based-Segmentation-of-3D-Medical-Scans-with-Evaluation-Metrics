import os
import glob
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import time

# config.py content
DATA_DIR = "/content/drive/MyDrive/Colab.data/tubitk folder"
OUTPUT_FOLDER_NAME = "SimpleOutput_SegmentationVersionThree" # New: Folder name for output in Google Drive
OUTPUT_DIR = os.path.join("/content/drive/MyDrive", OUTPUT_FOLDER_NAME) # Dynamically create Google Drive output path

# --- 2. MODEL & TRAINING PARAMETERS ---
# The shape of your .npy arrays (Height, Width, Depth)
INPUT_SHAPE = (128, 128, 64)
BATCH_SIZE = 1
EPOCHS = 100
LEARNING_RATE = 1e-4
NUM_CLASSES = 1 # Set to 1 for binary segmentation (background vs. foreground).
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_NAME = "light_unet_best.pth"

# metrics.py content (New Section)

def dice_score(preds, targets, smooth=1e-6):
    """
    Calculates the Dice Similarity Coefficient (DSC).
    Args:
        preds (torch.Tensor): Predicted masks (binarized: 0 or 1).
        targets (torch.Tensor): Ground truth masks (binarized: 0 or 1).
        smooth (float): A small constant to avoid division by zero.
    Returns:
        float: Dice score.
    """
    preds = preds.float().flatten()
    targets = targets.float().flatten()

    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum()

    dice = (2. * intersection + smooth) / (union + smooth)
    return dice

def iou_score(preds, targets, smooth=1e-6):
    """
    Calculates the Intersection Over Union (IoU) or Jaccard Index.
    Args:
        preds (torch.Tensor): Predicted masks (binarized: 0 or 1).
        targets (torch.Tensor): Ground truth masks (binarized: 0 or 1).
        smooth (float): A small constant to avoid division by zero.
    Returns:
        float: IoU score.
    """
    preds = preds.float().flatten()
    targets = targets.float().flatten()

    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection # Union = A + B - Intersection

    iou = (intersection + smooth) / (union + smooth)
    return iou

# You can add more metrics here if needed, e.g., accuracy, sensitivity, specificity
def calculate_metrics(outputs, masks):
    """
    Calculates Dice and IoU scores for a batch of predictions and ground truths.
    Args:
        outputs (torch.Tensor): Raw model outputs (logits).
        masks (torch.Tensor): Ground truth masks.
    Returns:
        tuple: (average_dice, average_iou)
    """
    preds = (torch.sigmoid(outputs) > 0.5).float()

    batch_dice = []
    batch_iou = []

    for i in range(preds.shape[0]): # Iterate over batch dimension
        d = dice_score(preds[i], masks[i])
        j = iou_score(preds[i], masks[i])
        batch_dice.append(d.item())
        batch_iou.append(j.item())

    return np.mean(batch_dice), np.mean(batch_iou)



# dataset.py content
def get_data_splits(data_dir):
    """
    Finds image and mask .npy files, pairs them, and splits them into
    training, validation, and test sets.

    Args:
        data_dir (str): The path to the directory containing the .npy files.

    Returns:
        tuple: A tuple containing three lists: train_files, val_files, test_files.
               Each list contains (image_path, mask_path) tuples.
    """
    print("\n--- Finding and splitting data files ---")
    print(f"DEBUG: 'data_dir' received: '{data_dir}'")

    #  Verify the data directory exists
    if not os.path.isdir(data_dir):
        print(f"ERROR: The directory '{data_dir}' does not exist.")
        raise FileNotFoundError(f"Directory '{data_dir}' not found. Please ensure the path is correct and exists on your system.")

    #  Find all image files using a glob pattern
    search_pattern = os.path.join(data_dir, "*_cropped_img.npy")
    print(f"DEBUG: Glob search pattern: '{search_pattern}'")

    all_image_files = sorted(glob.glob(search_pattern))
    print(f"Found {len(all_image_files)} image files.")
    print(f"DEBUG: All image files found by glob: {all_image_files[:5]}")

    if not all_image_files:
        raise FileNotFoundError(f"No *_cropped_img.npy files found in '{data_dir}' using pattern '{search_pattern}'. Check file names or directory.")

    #  Generate expected mask file paths from image file paths
    all_mask_files = [f.replace("_img.npy", "_mask.npy") for f in all_image_files]
    print(f"Generated {len(all_mask_files)} expected mask file paths.")

    print(" Sample image files found:", all_image_files[:2])
    print(" Sample mask files expected:", all_mask_files[:2])

    #  Pair image and mask files, checking for mask file existence
    paired_files = []
    for img, mask in zip(all_image_files, all_mask_files):
        if os.path.exists(mask):
            paired_files.append((img, mask))
        else:
            print(f" Missing mask file: {mask}")

    print(f"DEBUG: Number of valid image-mask pairs found: {len(paired_files)}")

    if not paired_files:
        raise FileNotFoundError(f"No valid image-mask pairs found in '{data_dir}'. Check file names.")

    #  Shuffle the paired files and split into train, validation, and test sets
    random.seed(42) # For reproducibility
    random.shuffle(paired_files)

    total_files = len(paired_files)
    train_split_idx = int(total_files * 0.8)
    val_split_idx = int(total_files * 0.9)

    train_files = paired_files[:train_split_idx]
    val_files = paired_files[train_split_idx:val_split_idx]
    test_files = paired_files[val_split_idx:]

    print(f"✅ Total pairs: {total_files} | Train: {len(train_files)} | Val: {len(val_files)} | Test: {len(test_files)}")
    return train_files, val_files, test_files


class SimpleNpyDataset(Dataset):
    """
    A PyTorch Dataset class to load pre-processed .npy image and mask files.
    It handles loading, squeezing (if necessary), casting to float32,
    and reshaping to PyTorch's (C, D, H, W) format for 3D data, or (C, H, W) for 2D.
    """
    def __init__(self, file_pairs):
        """
        Initializes the dataset with lists of image and mask file paths.

        Args:
            file_pairs (list): A list of (image_path, mask_path) tuples.
        """
        self.image_paths = [pair[0] for pair in file_pairs]
        self.mask_paths = [pair[1] for pair in file_pairs]
        print(f"🔍 Initialized dataset with {len(self.image_paths)} samples.")

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Loads and processes an image-mask pair at a given index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the processed image tensor and mask tensor.
                   Expected output shape for 3D data is (C, D, H, W).
        """
        # Load numpy arrays
        img = np.load(self.image_paths[idx])
        mask = np.load(self.mask_paths[idx])

        # Squeeze out singleton dimensions, e.g., (1, D, H, W) -> (D, H, W)
        if img.ndim == 4 and img.shape[0] == 1:
            img = np.squeeze(img, axis=0)
        if mask.ndim == 4 and mask.shape[0] == 1:
            mask = np.squeeze(mask, axis=0)

        # Binarize mask and ensure float32 type
        mask = (mask > 0).astype(np.float32)

        # Convert numpy arrays to PyTorch tensors and ensure float type
        img_tensor = torch.from_numpy(img).float()
        mask_tensor = torch.from_numpy(mask).float()

        # Reshape for PyTorch: (D, H, W) -> (C, D, H, W) for 3D grayscale
        if img_tensor.ndim == 3: # Assuming it's (Depth, Height, Width) for 3D grayscale
            img_tensor = img_tensor.unsqueeze(0)
        elif img_tensor.ndim == 2: # Assuming (H, W) for 2D grayscale
            img_tensor = img_tensor.unsqueeze(0)

        if mask_tensor.ndim == 3:
            mask_tensor = mask_tensor.unsqueeze(0)
        elif mask_tensor.ndim == 2:
            mask_tensor = mask_tensor.unsqueeze(0)

        return img_tensor, mask_tensor

# model.py content
def double_conv_block(in_channels, out_channels):
    """A block of two 3D convolutions with instance norm and ReLU."""
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.InstanceNorm3d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.InstanceNorm3d(out_channels),
        nn.ReLU(inplace=True),
    )

class LightUNet3D(nn.Module):
    """A lighter version of the 3D U-Net for faster training."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Encoder
        self.enc1 = double_conv_block(in_channels, 16)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = double_conv_block(16, 32)
        self.pool2 = nn.MaxPool3d(2)
        # Bottleneck
        self.bottleneck = double_conv_block(32, 64)
        # Decoder
        self.up2 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.dec2 = double_conv_block(64, 32)
        self.up1 = nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2)
        self.dec1 = double_conv_block(32, 16)
        # Output
        self.out_conv = nn.Conv3d(16, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))

        d2 = self.up2(b)
        if d2.shape[-3:] != e2.shape[-3:]:
            diffD = e2.shape[2] - d2.shape[2]
            diffH = e2.shape[3] - d2.shape[3]
            diffW = e2.shape[4] - d2.shape[4]
            e2 = e2[:, :, diffD // 2 : e2.shape[2] - diffD + diffD // 2,
                       diffH // 2 : e2.shape[3] - diffH + diffH // 2,
                       diffW // 2 : e2.shape[4] - diffW + diffW // 2]
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        if d1.shape[-3:] != e1.shape[-3:]:
            diffD = e1.shape[2] - d1.shape[2]
            diffH = e1.shape[3] - d1.shape[3]
            diffW = e1.shape[4] - d1.shape[4]
            e1 = e1[:, :, diffD // 2 : e1.shape[2] - diffD + diffD // 2,
                       diffH // 2 : e1.shape[3] - diffH + diffH // 2,
                       diffW // 2 : e1.shape[4] - diffW + diffW // 2]
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.out_conv(d1)

# train.py content

def train_model(model, train_loader, val_loader, optimizer, criterion, epochs, device, model_save_path):
    """Main training and validation loop."""
    print("\n--- Starting Model Training ---")
    best_val_loss = float('inf')

    for epoch in range(epochs):
        start_time = time.time()

        # --- Training phase ---
        model.train()
        total_train_loss = 0

        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # --- Validation phase ---
        model.eval()
        total_val_loss = 0
        total_val_dice = 0
        total_val_iou = 0

        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, masks)
                total_val_loss += loss.item()

                # Calculate metrics for validation
                dice, iou = calculate_metrics(outputs, masks)
                total_val_dice += dice
                total_val_iou += iou

        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_dice = total_val_dice / len(val_loader)
        avg_val_iou = total_val_iou / len(val_loader)

        # --- Logging and Saving ---
        elapsed_time = time.time() - start_time
        print(f"Epoch {epoch+1:02}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val Dice: {avg_val_dice:.4f} | "
              f"Val IoU: {avg_val_iou:.4f} | "
              f"Time: {elapsed_time:.2f}s")

        # Save the best model based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"  🎉 New best model saved with Val Loss: {best_val_loss:.4f}")

    print("\n--- Model Training Finished ---")


# predict.py content

def predict_and_visualize(model, loader, device, model_path, output_dir):
    """Runs prediction on a single test sample and saves a visualization."""
    print("\n--- Running Prediction and Visualization ---")

    # Load the best model weights
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Successfully loaded model from: {model_path}")
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}. Cannot perform prediction.")
        return
    except Exception as e:
        print(f"Error loading model state_dict: {e}")
        return

    model.eval() # Set model to evaluation mode

    total_test_dice = 0
    total_test_iou = 0
    num_samples_tested = 0

    with torch.no_grad():
        # Iterate through the entire test loader for comprehensive metric calculation
        for i, (current_img_batch, current_mask_batch) in enumerate(loader): # Renamed for clarity
            # Move tensors to the specified device (CPU/GPU)
            current_img_batch, current_mask_batch = current_img_batch.to(device), current_mask_batch.to(device)

            output_logits = model(current_img_batch)

            # Calculate metrics
            dice, iou = calculate_metrics(output_logits, current_mask_batch)
            total_test_dice += dice
            total_test_iou += iou
            num_samples_tested += 1

            # Only visualize the first sample
            if i == 0:
                # Apply sigmoid to get probabilities and binarize for the mask
                pred_probs = torch.sigmoid(output_logits)
                pred_mask = (pred_probs > 0.5).float() # Threshold at 0.5

                # Convert tensors to numpy arrays for visualization
                # Use current_img_batch and current_mask_batch directly
                img_np = current_img_batch.squeeze().cpu().numpy()
                mask_np = current_mask_batch.squeeze().cpu().numpy()
                pred_np = pred_mask.squeeze().cpu().numpy()

                # Ensure img_np, mask_np, pred_np are 3D (D, H, W) for slicing.
                while img_np.ndim > 3:
                    img_np = np.squeeze(img_np, axis=0)
                while mask_np.ndim > 3:
                    mask_np = np.squeeze(mask_np, axis=0)
                while pred_np.ndim > 3:
                    pred_np = np.squeeze(pred_np, axis=0)

                # Select a middle slice for visualization
                if img_np.ndim == 3:
                    slice_idx = img_np.shape[0] // 2
                elif img_np.ndim == 2:
                    slice_idx = None
                    print("Warning: Image array is 2D. Displaying the whole image without slicing.")
                else:
                    print(f"Error: Unexpected image dimensions ({img_np.shape}) for visualization. Expected 2D or 3D.")
                    continue # Skip visualization if dimensions are wrong

                # Plotting
                fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor='black')
                fig.suptitle('Segmentation Result Comparison', fontsize=16, color='white')

                # Original Image
                if slice_idx is not None:
                    axes[0].imshow(img_np[slice_idx], cmap='gray')
                    axes[0].set_title(f'Original Image (Slice {slice_idx})', color='white')
                else:
                    axes[0].imshow(img_np, cmap='gray')
                    axes[0].set_title('Original Image', color='white')
                axes[0].axis('off')

                # Ground Truth Mask (overlayed)
                if slice_idx is not None:
                    axes[1].imshow(img_np[slice_idx], cmap='gray')
                    axes[1].imshow(mask_np[slice_idx], cmap='Greens', alpha=0.5)
                else:
                    axes[1].imshow(img_np, cmap='gray')
                    axes[1].imshow(mask_np, cmap='Greens', alpha=0.5)
                axes[1].set_title('Ground Truth Mask', color='white')
                axes[1].axis('off')

                # Predicted Mask (overlayed)
                if slice_idx is not None:
                    axes[2].imshow(img_np[slice_idx], cmap='gray')
                    axes[2].imshow(pred_np[slice_idx], cmap='Reds', alpha=0.5)
                else:
                    axes[2].imshow(img_np, cmap='gray')
                    axes[2].imshow(pred_np, cmap='Reds', alpha=0.5)
                axes[2].set_title('Predicted Mask', color='white')
                axes[2].axis('off')

                plt.tight_layout()
                save_path = os.path.join(output_dir, "prediction_visualization.png")
                plt.savefig(save_path, facecolor='black')
                print(f"✅ Visualization saved to: {save_path}")
                plt.close()

        if num_samples_tested > 0:
            avg_test_dice = total_test_dice / num_samples_tested
            avg_test_iou = total_test_iou / num_samples_tested
            print(f"\n--- Test Set Metrics ---")
            print(f"Average Test Dice Score: {avg_test_dice:.4f}")
            print(f"Average Test IoU Score: {avg_test_iou:.4f}")
        else:
            print("No samples processed in the test set.")


# main.py content (now the main execution block)
def main():
    """Main function to run the entire pipeline."""
    # --- Setup ---
    print(f"Using device: {DEVICE}")

    # Create output directory in Google Drive
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory ensured: {OUTPUT_DIR}")

    # Get data file paths ---
    train_files, val_files, test_files = get_data_splits(DATA_DIR)

    # Create Datasets and DataLoaders ---
    train_dataset = SimpleNpyDataset(train_files)
    val_dataset = SimpleNpyDataset(val_files)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    #  Initialize Model, Loss, and Optimizer ---
    print("\n--- Initializing Model ---")
    model = LightUNet3D(in_channels=1, out_channels=NUM_CLASSES).to(DEVICE)
    print(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    #  Train the model ---
    model_save_path = os.path.join(OUTPUT_DIR, MODEL_SAVE_NAME)
    train_model(
        model, train_loader, val_loader, optimizer, criterion,
        EPOCHS, DEVICE, model_save_path
    )

    # Run prediction and visualize results on the test set ---
    if test_files:
        test_dataset = SimpleNpyDataset(test_files)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        predict_and_visualize(model, test_loader, DEVICE, model_save_path, OUTPUT_DIR)
    else:
        print("\n--- Skipping visualization as no files were allocated for the test set. ---")

    print("\n[COMPLETE] Script finished.")

if __name__ == "__main__":
    main()
