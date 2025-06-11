import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import seaborn as sns
import time
from tqdm import tqdm

# Define dataset directories
base_dir = 'C:\\Users\\samue\\OneDrive\\Documents\\Github\\GitHub\\Machine-Learning\\triple_mnist'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')

# Define output directory for saving images
output_dir = 'C:\\Users\\samue\\OneDrive\\Documents\\Github\\GitHub\\Machine-Learning\\triple_mnist'

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created output directory: {output_dir}")
else:
    print(f"Output directory exists: {output_dir}")

# Function to save a figure to the output directory
def save_figure(filename):
    """Save the current figure to the output directory"""
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath)
    print(f"Saved figure to {filepath}")
    return filepath

# Function to extract the individual digits from a three-digit label
def extract_digits(label):
    """Convert a label like '123' to a tuple of ints (1, 2, 3)"""
    return tuple(int(digit) for digit in label)

# Function to split an image into three equal parts horizontally
def split_image(img_array):
    """Split an 84x84 image into three 28x28 sections horizontally"""
    # Assume the image is 84x84 pixels (it may need to be adjusted if dimensions differ)
    height, width = img_array.shape
    part_width = width // 3
    
    # Split into three equal parts
    part1 = img_array[:, :part_width]
    part2 = img_array[:, part_width:2*part_width]
    part3 = img_array[:, 2*part_width:]
    
    return part1, part2, part3

# Function to visualize the image splitting
def visualize_split_image(img_array, parts, title="Original and Split Images"):
    """Visualize original image and its three parts"""
    plt.figure(figsize=(15, 4))
    
    # Show original image
    plt.subplot(1, 4, 1)
    plt.imshow(img_array, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')
    
    # Show three parts
    for i, part in enumerate(parts):
        plt.subplot(1, 4, i+2)
        plt.imshow(part, cmap='gray')
        plt.title(f"Part {i+1}")
        plt.axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    filename = f"split_image_example_{title.replace(' ', '_')}.png"
    save_figure(filename)
    plt.show()

# Function to load and preprocess images from a directory, splitting each image into three parts
def load_split_images(directory, max_samples=None):
    """Load and split images, returning three sets of images along with their labels"""
    images_part1 = []
    images_part2 = []
    images_part3 = []
    # Store three-digit labels
    labels = []
    
    # Get all subdirectories (label classes)
    print(f"Scanning directory: {directory}")
    label_dirs = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    
    if not label_dirs:
        print(f"No subdirectories found in {directory}. Please check the directory path.")
        return [], [], [], []
    
    print(f"Found {len(label_dirs)} label directories")
    
    # Loop through each label directory
    sample_count = 0
    pbar = tqdm(sorted(label_dirs), desc="Loading classes")
    for label in pbar:
        label_path = os.path.join(directory, label)
        image_paths = glob.glob(os.path.join(label_path, '*.png')) + glob.glob(os.path.join(label_path, '*.jpg'))
        
        # Extract the individual digits from the label
        try:
            d1, d2, d3 = extract_digits(label)
        except ValueError:
            print(f"Warning: Label {label} does not contain exactly three digits. Skipping.")
            continue
            
        # Update progress bar description
        pbar.set_description(f"Loading class {label} ({len(image_paths)} images)")
        
        # Loop through each image in this label directory
        for img_path in sorted(image_paths):
            # Load image
            try:
                img = Image.open(img_path)
                # If not already grayscale
                if img.mode != 'L':
                    img = img.convert('L')
                img_array = np.array(img)
                
                # Check image dimensions
                if img_array.shape != (84, 84):
                    print(f"Warning: Image {img_path} has unexpected dimensions {img_array.shape}. Expected (84, 84). Skipping.")
                    continue
                
                # Normalize pixel values to [0, 1]
                img_array = img_array / 255.0
                
                # Split image into three parts
                part1, part2, part3 = split_image(img_array)
                
                # Save the parts and labels
                images_part1.append(part1)
                images_part2.append(part2)
                images_part3.append(part3)
                labels.append((d1, d2, d3))
                
                # Visualize the first few splits for inspection
                if sample_count < 3:
                    visualize_split_image(img_array, [part1, part2, part3], 
                                          title=f"Label_{label}_{sample_count}")
                
                sample_count += 1
                if max_samples is not None and sample_count >= max_samples:
                    pbar.set_description(f"Reached max samples: {max_samples}")
                    break
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")
                continue
        
        if max_samples is not None and sample_count >= max_samples:
            break
    
    print(f"Successfully loaded {len(labels)} images, each split into three parts")
    return (np.array(images_part1), np.array(images_part2), np.array(images_part3), 
            np.array(labels))

# Load and split the datasets
print("\n- LOADING AND SPLITTING DATASETS")
max_train = 30000
max_val = 8000
max_test = 8000

X_train_p1, X_train_p2, X_train_p3, y_train = load_split_images(train_dir, max_samples=max_train)
X_val_p1, X_val_p2, X_val_p3, y_val = load_split_images(val_dir, max_samples=max_val)
X_test_p1, X_test_p2, X_test_p3, y_test = load_split_images(test_dir, max_samples=max_test)

# Extract individual digit labels
y_train_d1 = np.array([label[0] for label in y_train])
y_train_d2 = np.array([label[1] for label in y_train])
y_train_d3 = np.array([label[2] for label in y_train])

y_val_d1 = np.array([label[0] for label in y_val])
y_val_d2 = np.array([label[1] for label in y_val])
y_val_d3 = np.array([label[2] for label in y_val])

y_test_d1 = np.array([label[0] for label in y_test])
y_test_d2 = np.array([label[1] for label in y_test])
y_test_d3 = np.array([label[2] for label in y_test])

# Print dataset information
print("\n- DATASET INFORMATION")
print(f"Training set: {len(y_train)} images")
print(f"Validation set: {len(y_val)} images")
print(f"Test set: {len(y_test)} images")
print(f"Image part dimensions: {X_train_p1[0].shape}")

# Visualize a few samples of split images
plt.figure(figsize=(15, 6))
for i in range(3):
    if i < len(X_train_p1):
        # First digit
        plt.subplot(3, 3, i*3+1)
        plt.imshow(X_train_p1[i], cmap='gray')
        plt.title(f"Sample {i+1}, Digit 1: {y_train_d1[i]}")
        plt.axis('off')
        
        # Second digit
        plt.subplot(3, 3, i*3+2)
        plt.imshow(X_train_p2[i], cmap='gray')
        plt.title(f"Sample {i+1}, Digit 2: {y_train_d2[i]}")
        plt.axis('off')
        
        # Third digit
        plt.subplot(3, 3, i*3+3)
        plt.imshow(X_train_p3[i], cmap='gray')
        plt.title(f"Sample {i+1}, Digit 3: {y_train_d3[i]}")
        plt.axis('off')

plt.tight_layout()
save_figure('split_samples_visualization.png')
plt.show()

# Add channel dimension for CNN
X_train_p1 = X_train_p1.reshape(-1, X_train_p1.shape[1], X_train_p1.shape[2], 1)
X_train_p2 = X_train_p2.reshape(-1, X_train_p2.shape[1], X_train_p2.shape[2], 1)
X_train_p3 = X_train_p3.reshape(-1, X_train_p3.shape[1], X_train_p3.shape[2], 1)

X_val_p1 = X_val_p1.reshape(-1, X_val_p1.shape[1], X_val_p1.shape[2], 1)
X_val_p2 = X_val_p2.reshape(-1, X_val_p2.shape[1], X_val_p2.shape[2], 1)
X_val_p3 = X_val_p3.reshape(-1, X_val_p3.shape[1], X_val_p3.shape[2], 1)

X_test_p1 = X_test_p1.reshape(-1, X_test_p1.shape[1], X_test_p1.shape[2], 1)
X_test_p2 = X_test_p2.reshape(-1, X_test_p2.shape[1], X_test_p2.shape[2], 1)
X_test_p3 = X_test_p3.reshape(-1, X_test_p3.shape[1], X_test_p3.shape[2], 1)

# Define CNN model for digit recognition
def create_digit_cnn(input_shape):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        # 10 classes for digits 0-9
        layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Train three separate CNN models, one for each digit position
print("\n- TRAINING DIGIT RECOGNITION MODELS")

digit_names = ["First", "Second", "Third"]
digit_models = []
digit_histories = []

# Train and evaluate model for each digit position
for i, (X_train_part, y_train_digit, X_val_part, y_val_digit, X_test_part, y_test_digit) in enumerate([
    (X_train_p1, y_train_d1, X_val_p1, y_val_d1, X_test_p1, y_test_d1),
    (X_train_p2, y_train_d2, X_val_p2, y_val_d2, X_test_p2, y_test_d2),
    (X_train_p3, y_train_d3, X_val_p3, y_val_d3, X_test_p3, y_test_d3)
]):
    print(f"Training model for {digit_names[i]} digit")
    
    # Create and train model
    input_shape = X_train_part.shape[1:]
    model = create_digit_cnn(input_shape)
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=3,
            restore_best_weights=True
        )
    ]
    
    history = model.fit(
        X_train_part, y_train_digit,
        epochs=10,
        batch_size=64,
        validation_data=(X_val_part, y_val_digit),
        callbacks=callbacks,
        verbose=1
    )
    
    digit_models.append(model)
    digit_histories.append(history)
    
    # Evaluate on test set
    test_loss, test_accuracy = model.evaluate(X_test_part, y_test_digit, verbose=0)
    print(f"Test accuracy for {digit_names[i]} digit: {test_accuracy:.4f}")
    
    # Plot learning curves
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title(f'Accuracy - {digit_names[i]} Digit')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title(f'Loss - {digit_names[i]} Digit')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    save_figure(f'split_cnn_learning_curves_digit{i+1}.png')
    plt.show()

# Evaluate the combined model on the test set
print("\n- EVALUATING COMBINED MODEL")

# Get predictions for each digit
test_preds = [model.predict(x) for model, x in zip(digit_models, [X_test_p1, X_test_p2, X_test_p3])]
test_pred_classes = [np.argmax(pred, axis=1) for pred in test_preds]

# Measure individual digit accuracies
digit_accuracies = [
    accuracy_score(y_test_d1, test_pred_classes[0]),
    accuracy_score(y_test_d2, test_pred_classes[1]),
    accuracy_score(y_test_d3, test_pred_classes[2])
]

print("Individual digit accuracies:")
for i, acc in enumerate(digit_accuracies):
    print(f"{digit_names[i]} digit accuracy: {acc:.4f}")

# Measure overall sequence accuracy
sequence_correct = np.sum([
    (test_pred_classes[0][i] == y_test_d1[i]) & 
    (test_pred_classes[1][i] == y_test_d2[i]) & 
    (test_pred_classes[2][i] == y_test_d3[i]) 
    for i in range(len(y_test))
])
sequence_accuracy = sequence_correct / len(y_test)
print(f"Overall sequence accuracy: {sequence_accuracy:.4f}")

# Calculate F1 scores
f1_scores = [
    f1_score(y_test_d1, test_pred_classes[0], average='macro'),
    f1_score(y_test_d2, test_pred_classes[1], average='macro'),
    f1_score(y_test_d3, test_pred_classes[2], average='macro')
]

print("F1 Scores (macro):")
for i, f1 in enumerate(f1_scores):
    print(f"{digit_names[i]} digit F1 score: {f1:.4f}")

# Compare with Task 2 results
print("\n- COMPARISON WITH TASK 2")
# From Task 2 results
task2_cnn_digit_accuracies = [0.993, 0.101, 0.087]
task2_cnn_sequence_accuracy = 0.012

improvement_digit = [(new - old) * 100 for new, old in zip(digit_accuracies, task2_cnn_digit_accuracies)]
improvement_sequence = (sequence_accuracy - task2_cnn_sequence_accuracy) * 100

print("Improvement over Task 2 CNN model:")
for i, imp in enumerate(improvement_digit):
    print(f"{digit_names[i]} digit accuracy improvement: {imp:.2f}%")
print(f"Sequence accuracy improvement: {improvement_sequence:.2f}%")

# Visualize results comparison
plt.figure(figsize=(12, 5))

# Plot digit accuracies comparison
plt.subplot(1, 2, 1)
x = np.arange(3)
width = 0.35
plt.bar(x - width/2, task2_cnn_digit_accuracies, width, label='Task 2 CNN')
plt.bar(x + width/2, digit_accuracies, width, label='Task 3 Split CNN')
plt.xticks(x, digit_names)
plt.ylabel('Accuracy')
plt.title('Digit Recognition Accuracy Comparison')
plt.legend()
plt.ylim(0, 1)

# Add value labels
for i, v in enumerate(task2_cnn_digit_accuracies):
    plt.text(i - width/2, v + 0.02, f'{v:.3f}', ha='center')
for i, v in enumerate(digit_accuracies):
    plt.text(i + width/2, v + 0.02, f'{v:.3f}', ha='center')

# Plot sequence accuracy comparison
plt.subplot(1, 2, 2)
sequence_accuracies = [task2_cnn_sequence_accuracy, sequence_accuracy]
plt.bar(["Task 2 CNN", "Task 3 Split CNN"], sequence_accuracies)
plt.ylabel('Accuracy')
plt.title('Sequence Recognition Accuracy Comparison')
# Add some headroom for labels
plt.ylim(0, max(sequence_accuracies) * 1.2)

# Add value labels
for i, v in enumerate(sequence_accuracies):
    plt.text(i, v + 0.01, f'{v:.3f}', ha='center')

plt.tight_layout()
save_figure('task2_vs_task3_comparison.png')
plt.show()

# Generate confusion matrices for each digit
plt.figure(figsize=(15, 5))
for i, (y_true, y_pred) in enumerate(zip([y_test_d1, y_test_d2, y_test_d3], test_pred_classes)):
    plt.subplot(1, 3, i+1)
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {digit_names[i]} Digit')
    plt.xlabel('Predicted')
    plt.ylabel('True')

plt.tight_layout()
save_figure('task3_confusion_matrices.png')
plt.show()

# Calculate confusion matrices percentage (normalized)
plt.figure(figsize=(15, 5))
for i, (y_true, y_pred) in enumerate(zip([y_test_d1, y_test_d2, y_test_d3], test_pred_classes)):
    plt.subplot(1, 3, i+1)
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues')
    plt.title(f'Normalized Confusion Matrix - {digit_names[i]} Digit')
    plt.xlabel('Predicted')
    plt.ylabel('True')

plt.tight_layout()
save_figure('task3_normalized_confusion_matrices.png')
plt.show()

# Save a summary of the task 3 results
with open(os.path.join(output_dir, 'task3_results_summary.txt'), 'w') as f:
    f.write("- TASK 3 RESULTS SUMMARY\n")
    
    f.write("Individual Digit Accuracies:\n")
    for i, acc in enumerate(digit_accuracies):
        f.write(f"{digit_names[i]} digit: {acc:.4f}\n")
    
    f.write(f"\nOverall Sequence Accuracy: {sequence_accuracy:.4f}\n\n")
    
    f.write("F1 Scores (macro):\n")
    for i, f1 in enumerate(f1_scores):
        f.write(f"{digit_names[i]} digit: {f1:.4f}\n")
    
    f.write("\nImprovement over Task 2 CNN model:\n")
    for i, imp in enumerate(improvement_digit):
        f.write(f"{digit_names[i]} digit accuracy improvement: {imp:.2f}%\n")
    f.write(f"Sequence accuracy improvement: {improvement_sequence:.2f}%\n")

print(f"\nResults summary saved to {os.path.join(output_dir, 'task3_results_summary.txt')}")