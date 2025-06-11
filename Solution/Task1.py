import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
import random

# - PATH SETUP
# Dynamically find the base path from the current script location
script_dir = os.path.abspath(os.path.dirname(__file__))
while os.path.basename(script_dir) != "Machine-Learning":
    parent = os.path.dirname(script_dir)
    if parent == script_dir:
        raise FileNotFoundError("Could not locate 'Machine-Learning' directory in path tree.")
    script_dir = parent

# Define dataset and save directories
base_dir = os.path.join(script_dir, "triple_mnist")
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")
test_dir = os.path.join(base_dir, "test")

# Create directory to save outputs
save_dir = os.path.join(script_dir, "Task 1")
os.makedirs(save_dir, exist_ok=True)

# - DATA LOADING
# Load grayscale images and corresponding labels from directory structure
def load_data(directory):
    images, labels = [], []
    label_dirs = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]

    if not label_dirs:
        print(f"No subdirectories found in {directory}. Please check the directory path.")
        return np.array([]), np.array([])

    for label in label_dirs:
        path = os.path.join(directory, label)
        image_paths = glob.glob(os.path.join(path, "*.png")) + glob.glob(os.path.join(path, "*.jpg"))
        for img_path in image_paths:
            img = Image.open(img_path)
            if img.mode != 'L':
                img = img.convert('L')
            arr = np.array(img)
            images.append(arr)
            labels.append(label)

    return np.array(images), np.array(labels)

# - LOAD TRAINING DATA
print("\n- LOADING DATA")
print("Loading the Triple-MNIST dataset from training directory")
try:
    train_images, train_labels = load_data(train_dir)
    print("Dataset loaded successfully")

    # - DATASET INFORMATION
    if len(train_images) > 0:
        print("\n- DATASET INFORMATION")
        print(f"Number of training images: {len(train_images)}")
        print(f"Image dimensions: {train_images[0].shape}")
        print(f"Number of unique label combinations: {len(np.unique(train_labels))}")

        # Compute distribution of first digit across labels
        first_digits = [label[0] for label in train_labels]
        unique_digits, counts = np.unique(first_digits, return_counts=True)
        print("\n- FIRST DIGIT DISTRIBUTION")
        for d, c in zip(unique_digits, counts):
            print(f"Digit {d}: {c} images")

        # - RANDOM SAMPLE VISUALIZATION
        print("\n- VISUALIZING RANDOM SAMPLES")
        num_samples = min(10, len(train_images))
        sample_indices = random.sample(range(len(train_images)), num_samples)

        plt.figure(figsize=(15, 6))
        for i, idx in enumerate(sample_indices):
            plt.subplot(2, 5, i + 1)
            plt.imshow(train_images[idx], cmap='gray')
            plt.title(f"Label: {train_labels[idx]}")
            plt.axis('off')
        plt.suptitle("Random Samples from Training Data")
        plt.tight_layout()
        plt.subplots_adjust(top=0.88)
        sample_fig_path = os.path.join(save_dir, "random_samples.png")
        plt.savefig(sample_fig_path)
        plt.close()
        print(f"Saved: {sample_fig_path}")

        # - SELECTED LABEL CLASS VISUALIZATION
        print("\n- VISUALIZING SELECTED CLASS SAMPLES")
        selected_labels = np.random.choice(np.unique(train_labels), 5, replace=False)
        plt.figure(figsize=(15, 5))

        for i, label in enumerate(selected_labels):
            indices = np.where(train_labels == label)[0]
            if len(indices) > 0:
                idx = indices[0]
                plt.subplot(1, 5, i + 1)
                plt.imshow(train_images[idx], cmap='gray')
                plt.title(f"Label: {train_labels[idx]}")
                plt.axis('off')

        plt.suptitle("Selected Samples from Different Classes")
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        selected_fig_path = os.path.join(save_dir, "selected_samples.png")
        plt.savefig(selected_fig_path)
        plt.close()
        print(f"Saved: {selected_fig_path}")

except Exception as e:
    print(f"\nError loading or analyzing dataset: {e}")
    print("Please check the dataset structure and adjust the loading function if needed.")