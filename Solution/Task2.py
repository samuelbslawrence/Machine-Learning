import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import seaborn as sns
import time
from tqdm import tqdm
import re

# Define dataset directories
base_dir = 'C:\\Users\\samue\\OneDrive\\Documents\\Github\\GitHub\\Machine-Learning\\triple_mnist'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')

# Define save directory for plots and outputs
save_dir = os.path.join(base_dir, 'Task 2 Outputs')
os.makedirs(save_dir, exist_ok=True)

# Function to extract the individual digits from a three-digit label
def extract_digits(label):
    """Convert a label like '123' to a tuple of ints (1, 2, 3)"""
    return tuple(int(digit) for digit in label)

# Function to load and preprocess images from a directory
def load_data_with_individual_digits(directory, flatten=True, max_samples=None):
    """Load data with separate labels for each digit position"""
    images = []
    digit1_labels = []
    digit2_labels = []
    digit3_labels = []
    full_labels = []
    
    print(f"Scanning directory: {directory}")
    label_dirs = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    
    if not label_dirs:
        print(f"No subdirectories found in {directory}. Please check the directory path.")
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
    
    print(f"Found {len(label_dirs)} label directories")
    
    sample_count = 0
    pbar = tqdm(sorted(label_dirs), desc="Loading classes")
    for label in pbar:
        label_path = os.path.join(directory, label)
        image_paths = glob.glob(os.path.join(label_path, '*.png')) + glob.glob(os.path.join(label_path, '*.jpg'))
        
        try:
            d1, d2, d3 = extract_digits(label)
        except ValueError:
            print(f"Warning: Label {label} does not contain exactly three digits. Skipping.")
            continue
            
        pbar.set_description(f"Loading class {label} ({len(image_paths)} images)")
        
        for img_path in sorted(image_paths):
            try:
                img = Image.open(img_path)
                if img.mode != 'L':
                    img = img.convert('L')
                img_array = np.array(img)
                
                if img_array.shape != (84, 84):
                    print(f"Warning: Image {img_path} has unexpected dimensions {img_array.shape}. Expected (84, 84). Skipping.")
                    continue
                
                img_array = img_array / 255.0
                
                if flatten:
                    img_array = img_array.flatten()
                    
                images.append(img_array)
                digit1_labels.append(d1)
                digit2_labels.append(d2)
                digit3_labels.append(d3)
                full_labels.append(label)
                
                sample_count += 1
                if max_samples is not None and sample_count >= max_samples:
                    pbar.set_description(f"Reached max samples: {max_samples}")
                    break
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")
                continue
        
        if max_samples is not None and sample_count >= max_samples:
            break
    
    print(f"Successfully loaded {len(images)} images")
    return np.array(images), np.array(digit1_labels), np.array(digit2_labels), np.array(digit3_labels), np.array(full_labels)

# Load the datasets
print("\n- LOADING DATASETS")
print("Loading and preprocessing datasets...")
X_train, y_train_d1, y_train_d2, y_train_d3, y_train_full = load_data_with_individual_digits(train_dir, flatten=True, max_samples=30000)
X_val, y_val_d1, y_val_d2, y_val_d3, y_val_full = load_data_with_individual_digits(val_dir, flatten=True, max_samples=8000)
X_test, y_test_d1, y_test_d2, y_test_d3, y_test_full = load_data_with_individual_digits(test_dir, flatten=True, max_samples=8000)

# Print dataset information
print("\n- DATASET INFORMATION")
print(f"Training set: {X_train.shape[0]} images")
print(f"Validation set: {X_val.shape[0]} images")
print(f"Test set: {X_test.shape[0]} images")
print(f"Image vector size: {X_train.shape[1]}")
print(f"Unique values for first digit: {np.unique(y_train_d1)}")
print(f"Unique values for second digit: {np.unique(y_train_d2)}")
print(f"Unique values for third digit: {np.unique(y_train_d3)}")

# Visualize some examples to verify the data
print("\n- DATA VISUALIZATION")
print("Close image to continue.")
plt.figure(figsize=(15, 6))
for i in range(5):
    if i < len(X_train):
        plt.subplot(1, 5, i+1)
        img = X_train[i].reshape(84, 84)
        plt.imshow(img, cmap='gray')
        plt.title(f"Label: {y_train_d1[i]}{y_train_d2[i]}{y_train_d3[i]}")
        plt.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'sample_images.png'))
plt.show()

# Model 1: Decision Tree with PCA
print("\n- DECISION TREE MODEL")

print("Applying PCA for dimensionality reduction...")
n_components = 100
pca = PCA(n_components=n_components)
X_train_pca = pca.fit_transform(X_train)
X_val_pca = pca.transform(X_val)
X_test_pca = pca.transform(X_test)
print(f"Reduced feature dimension from {X_train.shape[1]} to {X_train_pca.shape[1]}")
print(f"Explained variance ratio: {np.sum(pca.explained_variance_ratio_):.4f}")

dt_models = []
dt_accuracies = []

for digit_position, (y_train_digit, y_val_digit, y_test_digit) in enumerate(
    [(y_train_d1, y_val_d1, y_test_d1), 
     (y_train_d2, y_val_d2, y_test_d2), 
     (y_train_d3, y_val_d3, y_test_d3)]):
    
    print(f"\nTraining Decision Tree for digit position {digit_position+1}...")
    
    dt_model = DecisionTreeClassifier(max_depth=20, random_state=42)
    dt_model.fit(X_train_pca, y_train_digit)
    dt_models.append(dt_model)
    
    y_val_pred = dt_model.predict(X_val_pca)
    val_accuracy = accuracy_score(y_val_digit, y_val_pred)
    print(f"Validation accuracy for digit {digit_position+1}: {val_accuracy:.4f}")
    
    y_test_pred = dt_model.predict(X_test_pca)
    test_accuracy = accuracy_score(y_test_digit, y_test_pred)
    dt_accuracies.append(test_accuracy)
    print(f"Test accuracy for digit {digit_position+1}: {test_accuracy:.4f}")

dt_test_preds = [dt_models[i].predict(X_test_pca) for i in range(3)]
dt_correct_sequences = np.sum([
    (dt_test_preds[0][i] == y_test_d1[i]) & 
    (dt_test_preds[1][i] == y_test_d2[i]) & 
    (dt_test_preds[2][i] == y_test_d3[i]) 
    for i in range(len(X_test_pca))
])
dt_sequence_accuracy = dt_correct_sequences / len(X_test_pca)
print(f"\nDecision Tree overall sequence accuracy: {dt_sequence_accuracy:.4f}")

print("\n- DECISION TREE SUMMARY")
print(f"Per-digit test accuracies: {['{:.4f}'.format(a) for a in dt_accuracies]}")
print(f"Sequence accuracy: {dt_sequence_accuracy:.4f}")

# Model 2: CNN
print("\n- CNN MODEL")

X_train_cnn, y_train_d1_cnn, y_train_d2_cnn, y_train_d3_cnn, _ = load_data_with_individual_digits(
    train_dir, flatten=False, max_samples=30000)
X_val_cnn, y_val_d1_cnn, y_val_d2_cnn, y_val_d3_cnn, _ = load_data_with_individual_digits(
    val_dir, flatten=False, max_samples=8000)
X_test_cnn, y_test_d1_cnn, y_test_d2_cnn, y_test_d3_cnn, _ = load_data_with_individual_digits(
    test_dir, flatten=False, max_samples=8000)

X_train_cnn = X_train_cnn.reshape(-1, 84, 84, 1)
X_val_cnn = X_val_cnn.reshape(-1, 84, 84, 1)
X_test_cnn = X_test_cnn.reshape(-1, 84, 84, 1)

def create_digit_recognition_cnn():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(84, 84, 1)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

cnn_models = []
cnn_accuracies = []
digit_names = ["first", "second", "third"]

for digit_position, (y_train_digit, y_val_digit, y_test_digit) in enumerate([
    (y_train_d1_cnn, y_val_d1_cnn, y_test_d1_cnn),
    (y_train_d2_cnn, y_val_d2_cnn, y_test_d2_cnn),
    (y_train_d3_cnn, y_val_d3_cnn, y_test_d3_cnn)
]):
    print(f"\nTraining CNN for {digit_names[digit_position]} digit...")
    
    model = create_digit_recognition_cnn()
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=3,
            restore_best_weights=True
        )
    ]
    
    history = model.fit(
        X_train_cnn, y_train_digit,
        epochs=10,
        batch_size=64,
        validation_data=(X_val_cnn, y_val_digit),
        callbacks=callbacks,
        verbose=1
    )
    
    cnn_models.append(model)
    
    test_loss, test_accuracy = model.evaluate(X_test_cnn, y_test_digit, verbose=0)
    cnn_accuracies.append(test_accuracy)
    print(f"Test accuracy for {digit_names[digit_position]} digit: {test_accuracy:.4f}")
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title(f'Accuracy - {digit_names[digit_position].capitalize()} Digit')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title(f'Loss - {digit_names[digit_position].capitalize()} Digit')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'cnn_learning_curves_digit{digit_position+1}.png'))
    plt.show()

cnn_test_preds = [np.argmax(cnn_models[i].predict(X_test_cnn), axis=1) for i in range(3)]
cnn_correct_sequences = np.sum([
    (cnn_test_preds[0][i] == y_test_d1_cnn[i]) & 
    (cnn_test_preds[1][i] == y_test_d2_cnn[i]) & 
    (cnn_test_preds[2][i] == y_test_d3_cnn[i]) 
    for i in range(len(X_test_cnn))
])
cnn_sequence_accuracy = cnn_correct_sequences / len(X_test_cnn)
print(f"\nCNN overall sequence accuracy: {cnn_sequence_accuracy:.4f}")

print("\n- CNN SUMMARY")
print(f"Per-digit test accuracies: {['{:.4f}'.format(a) for a in cnn_accuracies]}")
print(f"Sequence accuracy: {cnn_sequence_accuracy:.4f}")

# Model Comparison and Visualization
print("\n- MODEL COMPARISON")

plt.figure(figsize=(10, 6))
x = np.arange(3)
width = 0.35

plt.bar(x - width/2, dt_accuracies, width, label='Decision Tree')
plt.bar(x + width/2, cnn_accuracies, width, label='CNN')
plt.xticks(x, ["First Digit", "Second Digit", "Third Digit"])
plt.ylabel('Accuracy')
plt.title('Per-Digit Accuracy Comparison')
plt.legend()
plt.ylim(0, 1)

for i, v in enumerate(dt_accuracies):
    plt.text(i - width/2, v + 0.02, f'{v:.3f}', ha='center')
for i, v in enumerate(cnn_accuracies):
    plt.text(i + width/2, v + 0.02, f'{v:.3f}', ha='center')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'per_digit_accuracy.png'))
plt.show()

plt.figure(figsize=(8, 6))
sequence_accuracies = [dt_sequence_accuracy, cnn_sequence_accuracy]
plt.bar(["Decision Tree", "CNN"], sequence_accuracies)
plt.ylabel('Accuracy')
plt.title('Overall Sequence Accuracy Comparison')
plt.ylim(0, 1)

for i, v in enumerate(sequence_accuracies):
    plt.text(i, v + 0.02, f'{v:.3f}', ha='center')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'sequence_accuracy.png'))
plt.show()

digit_pos = 0
cm_dt = confusion_matrix(y_test_d1, dt_test_preds[digit_pos])
cm_cnn = confusion_matrix(y_test_d1_cnn, cnn_test_preds[digit_pos])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Blues', ax=ax1)
ax1.set_title('Decision Tree Confusion Matrix - First Digit')
ax1.set_xlabel('Predicted')
ax1.set_ylabel('True')

sns.heatmap(cm_cnn, annot=True, fmt='d', cmap='Blues', ax=ax2)
ax2.set_title('CNN Confusion Matrix - First Digit')
ax2.set_xlabel('Predicted')
ax2.set_ylabel('True')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'confusion_matrices.png'))
plt.show()