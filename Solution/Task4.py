import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
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

# Function to save a figure to the output directory and close it
def save_figure(filename):
    """Save the current figure to the output directory and close it"""
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath)
    print(f"Saved figure to {filepath}")
    plt.close()  # Close the figure
    return filepath

# Function to extract the individual digits from a three-digit label
def extract_digits(label):
    """Convert a label like '123' to a tuple of ints (1, 2, 3)"""
    return tuple(int(digit) for digit in label)

# Function to split an image into three equal parts horizontally
def split_image(img_array):
    """Split an 84x84 image into three 28x28 sections horizontally"""
    height, width = img_array.shape
    part_width = width // 3
    
    # Split into three equal parts
    part1 = img_array[:, :part_width]
    part2 = img_array[:, part_width:2*part_width]
    part3 = img_array[:, 2*part_width:]
    
    return part1, part2, part3

# Function to load and preprocess images from a directory, splitting each image into three parts
def load_split_images(directory, max_samples=None):
    """Load and split images, returning three sets of images along with their labels"""
    images_part1 = []
    images_part2 = []
    images_part3 = []
    labels = []  # Store three-digit labels
    
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
                if img.mode != 'L':  # If not already grayscale
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

# Define the Task 3 model for baseline comparison
def create_baseline_cnn(input_shape):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

print("\n- ANALYZING BASELINE MODELS")

# Train the baseline models briefly to see learning curves
# Only for analysis purposes - we'll later compare with enhanced models
digit_names = ["First", "Second", "Third"]
baseline_histories = []

# Using a much smaller subset of data and epochs for this baseline analysis
analysis_epochs = 5
subset_size = 5000
subset_val_size = 1000

print("Running baseline models with small subset of data to analyze learning curves...")

# Plot example of images for each digit position
plt.figure(figsize=(15, 5))
for i in range(3):
    if i < len(X_train_p1):
        plt.subplot(1, 3, i+1)
        digit_image = X_train_p1[i].reshape(X_train_p1[i].shape[0], X_train_p1[i].shape[1])
        plt.imshow(digit_image, cmap='gray')
        plt.title(f"First Digit Sample {i+1}")
        plt.axis('off')
save_figure('first_digit_samples.png')

plt.figure(figsize=(15, 5))
for i in range(3):
    if i < len(X_train_p2):
        plt.subplot(1, 3, i+1)
        digit_image = X_train_p2[i].reshape(X_train_p2[i].shape[0], X_train_p2[i].shape[1])
        plt.imshow(digit_image, cmap='gray')
        plt.title(f"Second Digit Sample {i+1}")
        plt.axis('off')
save_figure('second_digit_samples.png')

plt.figure(figsize=(15, 5))
for i in range(3):
    if i < len(X_train_p3):
        plt.subplot(1, 3, i+1)
        digit_image = X_train_p3[i].reshape(X_train_p3[i].shape[0], X_train_p3[i].shape[1])
        plt.imshow(digit_image, cmap='gray')
        plt.title(f"Third Digit Sample {i+1}")
        plt.axis('off')
save_figure('third_digit_samples.png')

# Train baseline models for analysis
for i, (X_train_part, y_train_digit, X_val_part, y_val_digit) in enumerate([
    (X_train_p1[:subset_size], y_train_d1[:subset_size], X_val_p1[:subset_val_size], y_val_d1[:subset_val_size]),
    (X_train_p2[:subset_size], y_train_d2[:subset_size], X_val_p2[:subset_val_size], y_val_d2[:subset_val_size]),
    (X_train_p3[:subset_size], y_train_d3[:subset_size], X_val_p3[:subset_val_size], y_val_d3[:subset_val_size])
]):
    print(f"Training baseline model for {digit_names[i]} digit...")
    
    model = create_baseline_cnn(X_train_part.shape[1:])
    
    history = model.fit(
        X_train_part, y_train_digit,
        epochs=analysis_epochs,
        batch_size=64,
        validation_data=(X_val_part, y_val_digit),
        verbose=1
    )
    
    baseline_histories.append(history)
    
    # Plot and analyze learning curves to check for overfitting/underfitting
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'Baseline Model - {digit_names[i]} Digit Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Baseline Model - {digit_names[i]} Digit Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    save_figure(f'baseline_learning_curves_{digit_names[i]}.png')

# Analyze the baseline models and print findings
print("\n- ANALYSIS OF BASELINE MODELS")
print("Based on the learning curves analysis:")

for i, digit in enumerate(digit_names):
    # Check if validation loss is increasing while training loss decreases
    train_loss = baseline_histories[i].history['loss']
    val_loss = baseline_histories[i].history['val_loss']
    
    if train_loss[-1] < train_loss[0] and val_loss[-1] > val_loss[0]:
        print(f"- {digit} digit model shows signs of overfitting (validation loss increases)")
    elif train_loss[-1] > train_loss[0] * 0.5:  # If final loss is still > 50% of initial loss
        print(f"- {digit} digit model shows signs of underfitting (training loss remains high)")
    else:
        print(f"- {digit} digit model learning appears balanced")

print("\nProposed enhancements based on analysis:")
print("1. Data Augmentation: To increase training data diversity and reduce overfitting")
print("2. Enhanced Model Architecture: Deeper network with more complexity and regularization")

print("\n- ENHANCEMENT 1: DATA AUGMENTATION")

# Create data augmentation generators
def create_data_generator():
    return ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        fill_mode='nearest'
    )

# Visualize the data augmentations
datagen = create_data_generator()

# Generate example augmentations for a single image
def visualize_augmentations(X, y, datagen, num_augmentations=5):
    # Take a sample image
    X_sample = X[0:1]  # Keep batch dimension
    y_sample = y[0]
    
    # Fit the generator
    datagen.fit(X_sample)
    
    # Generate augmented images
    aug_iter = datagen.flow(X_sample, batch_size=1)
    
    # Plot original and augmented images
    plt.figure(figsize=(12, 3))
    
    # Original image
    plt.subplot(1, num_augmentations+1, 1)
    plt.imshow(X_sample[0].reshape(X_sample.shape[1], X_sample.shape[2]), cmap='gray')
    plt.title(f"Original\n(Label: {y_sample})")
    plt.axis('off')
    
    # Augmented images
    for i in range(num_augmentations):
        aug_img = next(aug_iter)[0]
        plt.subplot(1, num_augmentations+1, i+2)
        plt.imshow(aug_img.reshape(X_sample.shape[1], X_sample.shape[2]), cmap='gray')
        plt.title(f"Augmentation {i+1}")
        plt.axis('off')
    
    plt.tight_layout()
    save_figure(f"data_augmentation_examples.png")

# Visualize augmentations for the first digit
visualize_augmentations(X_train_p1, y_train_d1, datagen)

print("\n- ENHANCEMENT 2: ENHANCED CNN ARCHITECTURE")

# Define an enhanced CNN architecture with more complexity and regularization
def create_enhanced_cnn(input_shape):
    model = models.Sequential([
        # First convolution block - increased filters
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second convolution block - increased filters
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Third convolution block
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Fully connected layers
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    
    # Compile with a slower learning rate for stability
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.0005),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Create the enhanced model and display its architecture
input_shape = X_train_p1.shape[1:]
enhanced_model = create_enhanced_cnn(input_shape)
enhanced_model.summary()

# Try to visualize the model architecture, but handle the case where pydot is not installed
try:
    print("Saving enhanced model architecture to file...")
    tf.keras.utils.plot_model(
        enhanced_model, 
        to_file=os.path.join(output_dir, 'enhanced_model_architecture.png'),
        show_shapes=True, 
        show_layer_names=True
    )
except ImportError:
    print("Note: Could not generate model architecture visualization.")
    print("Continuing with the rest of the code...")

print("\n- TRAINING ENHANCED MODELS")

digit_names = ["First", "Second", "Third"]
enhanced_models = []
enhanced_histories = []

# Create data generators for each digit
datagen1 = create_data_generator()
datagen2 = create_data_generator()
datagen3 = create_data_generator()

# Train and evaluate model for each digit position
for i, (X_train_part, y_train_digit, X_val_part, y_val_digit, X_test_part, y_test_digit, datagen) in enumerate([
    (X_train_p1, y_train_d1, X_val_p1, y_val_d1, X_test_p1, y_test_d1, datagen1),
    (X_train_p2, y_train_d2, X_val_p2, y_val_d2, X_test_p2, y_test_d2, datagen2),
    (X_train_p3, y_train_d3, X_val_p3, y_val_d3, X_test_p3, y_test_d3, datagen3)
]):
    print(f"Training enhanced model for {digit_names[i]} digit")
    
    # Create enhanced model
    input_shape = X_train_part.shape[1:]
    model = create_enhanced_cnn(input_shape)
    
    # Add callbacks for early stopping and learning rate scheduling
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=0.00001
        )
    ]
    
    # Fit the generator to the training data
    datagen.fit(X_train_part)
    
    # Train with data augmentation
    history = model.fit(
        datagen.flow(X_train_part, y_train_digit, batch_size=64),
        epochs=15,
        steps_per_epoch=len(X_train_part) // 64,
        validation_data=(X_val_part, y_val_digit),
        callbacks=callbacks,
        verbose=1
    )
    
    enhanced_models.append(model)
    enhanced_histories.append(history)
    
    # Evaluate on test set
    test_loss, test_accuracy = model.evaluate(X_test_part, y_test_digit, verbose=0)
    print(f"Test accuracy for {digit_names[i]} digit: {test_accuracy:.4f}")
    
    # Plot learning curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title(f'Enhanced Model - {digit_names[i]} Digit Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title(f'Enhanced Model - {digit_names[i]} Digit Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    save_figure(f'enhanced_learning_curves_{digit_names[i]}.png')

print("\n- EVALUATING ENHANCED MODELS")

# Get predictions for each digit position
enhanced_test_preds = [model.predict(x, verbose=0) for model, x in zip(enhanced_models, [X_test_p1, X_test_p2, X_test_p3])]
enhanced_test_pred_classes = [np.argmax(pred, axis=1) for pred in enhanced_test_preds]

# Measure individual digit accuracies
enhanced_digit_accuracies = [
    accuracy_score(y_test_d1, enhanced_test_pred_classes[0]),
    accuracy_score(y_test_d2, enhanced_test_pred_classes[1]),
    accuracy_score(y_test_d3, enhanced_test_pred_classes[2])
]

print("Individual digit accuracies:")
for i, acc in enumerate(enhanced_digit_accuracies):
    print(f"{digit_names[i]} digit accuracy: {acc:.4f}")

# Measure overall sequence accuracy
enhanced_sequence_correct = np.sum([
    (enhanced_test_pred_classes[0][i] == y_test_d1[i]) & 
    (enhanced_test_pred_classes[1][i] == y_test_d2[i]) & 
    (enhanced_test_pred_classes[2][i] == y_test_d3[i]) 
    for i in range(len(y_test))
])
enhanced_sequence_accuracy = enhanced_sequence_correct / len(y_test)
print(f"Overall sequence accuracy: {enhanced_sequence_accuracy:.4f}")

# Calculate F1 scores
enhanced_f1_scores = [
    f1_score(y_test_d1, enhanced_test_pred_classes[0], average='macro'),
    f1_score(y_test_d2, enhanced_test_pred_classes[1], average='macro'),
    f1_score(y_test_d3, enhanced_test_pred_classes[2], average='macro')
]

print("F1 Scores (macro):")
for i, f1 in enumerate(enhanced_f1_scores):
    print(f"{digit_names[i]} digit F1 score: {f1:.4f}")

print("\n- COMPARISON WITH TASK 3 BASELINE")

# Define baseline results (based on Task 3)
# These values should be updated with the actual Task 3 results
task3_digit_accuracies = [0.993, 0.101, 0.087]  # First, Second, Third digits
task3_sequence_accuracy = 0.012  # Overall sequence accuracy

# Calculate improvement percentages
improvement_digit = [(new - old) * 100 for new, old in zip(enhanced_digit_accuracies, task3_digit_accuracies)]
improvement_sequence = (enhanced_sequence_accuracy - task3_sequence_accuracy) * 100

print("Improvement over Task 3 baseline:")
for i, imp in enumerate(improvement_digit):
    print(f"{digit_names[i]} digit accuracy: {task3_digit_accuracies[i]:.4f} → {enhanced_digit_accuracies[i]:.4f} ({imp:+.2f}%)")
print(f"Sequence accuracy: {task3_sequence_accuracy:.4f} → {enhanced_sequence_accuracy:.4f} ({improvement_sequence:+.2f}%)")

# Visualize the improvements
plt.figure(figsize=(12, 5))

# 1. Digit accuracy comparison
plt.subplot(1, 2, 1)
x = np.arange(len(digit_names))
width = 0.35

plt.bar(x - width/2, task3_digit_accuracies, width, label='Task 3 Baseline')
plt.bar(x + width/2, enhanced_digit_accuracies, width, label='Task 4 Enhanced')
plt.title('Per-Digit Accuracy Comparison')
plt.xticks(x, digit_names)
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.legend()

# Add value labels on bars
for i, v in enumerate(task3_digit_accuracies):
    plt.text(i - width/2, v + 0.02, f'{v:.3f}', ha='center')
for i, v in enumerate(enhanced_digit_accuracies):
    plt.text(i + width/2, v + 0.02, f'{v:.3f}', ha='center')

# 2. Sequence accuracy comparison
plt.subplot(1, 2, 2)
sequence_accuracies = [task3_sequence_accuracy, enhanced_sequence_accuracy]
plt.bar(['Task 3 Baseline', 'Task 4 Enhanced'], sequence_accuracies)
plt.title('Overall Sequence Accuracy Comparison')
plt.ylabel('Accuracy')
plt.ylim(0, max(max(sequence_accuracies) * 1.2, 0.05))  # Add some headroom for labels

# Add value labels on bars
for i, v in enumerate(sequence_accuracies):
    plt.text(i, v + 0.005, f'{v:.3f}', ha='center')

plt.tight_layout()
save_figure('task3_vs_task4_comparison.png')

# Generate confusion matrices for each digit
plt.figure(figsize=(15, 5))
for i, (y_true, y_pred) in enumerate(zip([y_test_d1, y_test_d2, y_test_d3], enhanced_test_pred_classes)):
    plt.subplot(1, 3, i+1)
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {digit_names[i]} Digit')
    plt.xlabel('Predicted')
    plt.ylabel('True')

plt.tight_layout()
save_figure('enhanced_model_confusion_matrices.png')

# Save a detailed summary of the enhancement results
with open(os.path.join(output_dir, 'task4_results_summary.txt'), 'w') as f:
    f.write("- TASK 4: MODEL ENHANCEMENT RESULTS\n")
    
    f.write("Enhancement Techniques Applied:\n")
    f.write("1. Data Augmentation:\n")
    f.write("   - Rotation range: ±10 degrees\n")
    f.write("   - Width/height shifts: ±10%\n")
    f.write("   - Zoom range: ±10%\n\n")
    
    f.write("2. Enhanced CNN Architecture:\n")
    f.write("   - Increased model complexity (more filters and layers)\n")
    f.write("   - Added batch normalization for training stability\n")
    f.write("   - Increased dropout for better regularization\n")
    f.write("   - Learning rate scheduling for more effective training\n\n")
    
    f.write("Performance Comparison:\n")
    f.write("Individual Digit Accuracies:\n")
    for i, (old, new) in enumerate(zip(task3_digit_accuracies, enhanced_digit_accuracies)):
        f.write(f"- {digit_names[i]} digit: {old:.4f} → {new:.4f} (Change: {improvement_digit[i]:+.2f}%)\n")
    
    f.write(f"\nOverall Sequence Accuracy: {task3_sequence_accuracy:.4f} → {enhanced_sequence_accuracy:.4f} (Change: {improvement_sequence:+.2f}%)\n\n")
    
    f.write("Analysis of Results:\n")
    
    # This part provides an analysis based on the actual results
    # We'll need to adapt this based on the observed outcomes
    if enhanced_sequence_accuracy > task3_sequence_accuracy:
        f.write("The enhancements successfully improved overall model performance.")
    else:
        f.write("The enhancements did not significantly improve overall sequence accuracy.")

print(f"\nResults summary saved to {os.path.join(output_dir, 'task4_results_summary.txt')}")