import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
import random

# Define dataset directories
base_dir = 'C:\\Users\\samue\\OneDrive\\Documents\\Github\\GitHub\\Machine-Learning\\triple_mnist'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')

# Define save directory for plots
save_dir = 'C:\\Users\\samue\\OneDrive\\Documents\\Github\\GitHub\\Machine-Learning\\Task 1'
# Create the directory if it doesn't exist
os.makedirs(save_dir, exist_ok=True)

# Function to load images and labels from the directory structure
def load_data(directory):
    images = []
    labels = []
   
    # Get all subdirectories (label classes)
    label_dirs = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
   
    if not label_dirs:
        print(f"No subdirectories found in {directory}. Please check the directory path.")
        return np.array([]), np.array([])
   
    # Loop through each label directory
    for label in label_dirs:
        label_path = os.path.join(directory, label)
        image_paths = glob.glob(os.path.join(label_path, '*.png')) + glob.glob(os.path.join(label_path, '*.jpg'))
       
        # Loop through each image in this label directory
        for img_path in image_paths:
            # Load image
            img = Image.open(img_path)
            # If not already grayscale
            if img.mode != 'L':
                img = img.convert('L')
            img_array = np.array(img)
           
            images.append(img_array)
            # The label is the directory name (three digits)
            labels.append(label)
   
    return np.array(images), np.array(labels)

# Load training data
print("Loading the Triple-MNIST dataset...")
try:
    train_images, train_labels = load_data(train_dir)
   
    if len(train_images) > 0:
        # Print dataset information
        print(f"\n- DATASET INFORMATION")
        print(f"Number of training images: {len(train_images)}")
        print(f"Image dimensions: {train_images[0].shape}")
        print(f"Number of unique label combinations: {len(np.unique(train_labels))}")
       
        # Distribution of first digits
        first_digits = [label[0] for label in train_labels]
        unique_first_digits, counts = np.unique(first_digits, return_counts=True)
        print("\n- DISTRIBUTION OF FIRST DIGITS")
        for digit, count in zip(unique_first_digits, counts):
            print(f"Digit {digit}: {count} images")
        
        print("")

        # Visualize random samples
        plt.figure(figsize=(15, 10))
        num_samples = min(10, len(train_images))
        indices = random.sample(range(len(train_images)), num_samples)
       
        for i, idx in enumerate(indices):
            plt.subplot(2, 5, i+1)
            plt.imshow(train_images[idx], cmap='gray')
            plt.title(f"Label: {train_labels[idx]}")
            plt.axis('off')
        
        plt.suptitle("Random Samples from Training Data")
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        
        # Save the figure instead of displaying it
        random_samples_filename = os.path.join(save_dir, "Random_Samples_from_Training_Data.png")
        plt.savefig(random_samples_filename)
        # Close the figure window
        plt.close()
        print(f"Saved: {random_samples_filename}")
       
        # Also visualize one example from each of a few selected classes
        selected_labels = np.random.choice(np.unique(train_labels), 5, replace=False)
        plt.figure(figsize=(15, 5))
       
        for i, label in enumerate(selected_labels):
            # Find indices of images with this label
            indices = np.where(train_labels == label)[0]
            if len(indices) > 0:
                # Take the first one
                idx = indices[0]
                plt.subplot(1, 5, i+1)
                plt.imshow(train_images[idx], cmap='gray')
                plt.title(f"Label: {train_labels[idx]}")
                plt.axis('off')
       
        plt.suptitle("Selected Samples from Different Classes")
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        
        # Save the figure instead of displaying it
        selected_samples_filename = os.path.join(save_dir, "Selected_Samples_from_Different_Classes.png")
        plt.savefig(selected_samples_filename)
        # Close the figure window
        plt.close()
        print(f"Saved: {selected_samples_filename}")
   
except Exception as e:
    print(f"Error analyzing dataset: {e}")
    print("Please adjust the data loading function to match your dataset's structure.")