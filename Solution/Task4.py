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
from tqdm import tqdm

# PATH SETUP
# Dynamically find the base path from the current script location
script_dir = os.path.abspath(os.path.dirname(__file__))
while os.path.basename(script_dir) != "Machine-Learning":
    parent = os.path.dirname(script_dir)
    if parent == script_dir:
        raise FileNotFoundError(
            "Could not locate 'Machine-Learning' directory in path tree."
        )
    script_dir = parent

# Define dataset and output directories
base_dir = os.path.join(script_dir, "triple_mnist")
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")
test_dir = os.path.join(base_dir, "test")

# Create directory to save outputs
output_dir = os.path.join(script_dir, "Task 4")
os.makedirs(output_dir, exist_ok=True)

# === UTILITY FUNCTIONS ===
def save_figure(filename):
    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()

def extract_digits(label):
    return tuple(int(d) for d in label)

def split_image(img):
    w = img.shape[1] // 3
    return img[:, :w], img[:, w:2*w], img[:, 2*w:]

def load_split_images(directory, max_samples=None):
    print(f"Loading data from: {directory}")
    part1, part2, part3, labels = [], [], [], []
    count = 0
    label_dirs = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    
    for label in sorted(label_dirs):  # Removed tqdm from here
        try:
            d1, d2, d3 = extract_digits(label)
        except:
            continue
        for path in sorted(glob.glob(os.path.join(directory, label, '*.png'))):
            try:
                img = Image.open(path).convert('L')
                arr = np.array(img) / 255.0
                if arr.shape != (84, 84): continue
                p1, p2, p3 = split_image(arr)
                part1.append(p1)
                part2.append(p2)
                part3.append(p3)
                labels.append((d1, d2, d3))
                count += 1
                if max_samples and count >= max_samples:
                    break
            except:
                continue
        if max_samples and count >= max_samples:
            break

    print(f"Loaded {len(labels)} samples from: {directory}")
    return np.array(part1), np.array(part2), np.array(part3), np.array(labels)

# === DATA LOADING ===
print("\n- LOADING DATA")
X_train_p1, X_train_p2, X_train_p3, y_train = load_split_images(train_dir, 30000)
X_val_p1, X_val_p2, X_val_p3, y_val = load_split_images(val_dir, 8000)
X_test_p1, X_test_p2, X_test_p3, y_test = load_split_images(test_dir, 8000)

# === DATASET INFORMATION ===
print("\n- DATASET INFORMATION")
print(f"Training set: {len(y_train)} images")
print(f"Validation set: {len(y_val)} images")
print(f"Test set: {len(y_test)} images")
print(f"Split image size: {X_train_p1[0].shape}")

y_train_digits = [y_train[:, i] for i in range(3)]
y_val_digits = [y_val[:, i] for i in range(3)]
y_test_digits = [y_test[:, i] for i in range(3)]

X_data = [[X_train_p1, X_train_p2, X_train_p3], [X_val_p1, X_val_p2, X_val_p3], [X_test_p1, X_test_p2, X_test_p3]]
for dataset in X_data:
    for i in range(3):
        dataset[i] = dataset[i].reshape(-1, dataset[i].shape[1], dataset[i].shape[2], 1)

# === BASELINE CNN TRAINING ===
def create_baseline_cnn(input_shape):
    model = models.Sequential([
        layers.Conv2D(32, 3, activation='relu', padding='same', input_shape=input_shape),
        layers.MaxPooling2D(2),
        layers.Conv2D(64, 3, activation='relu', padding='same'),
        layers.MaxPooling2D(2),
        layers.Conv2D(64, 3, activation='relu', padding='same'),
        layers.MaxPooling2D(2),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

print("\n- TRAINING BASELINE MODELS")
digit_names = ["First", "Second", "Third"]
baseline_histories = []
for i, (X_train, y_train_digit, X_val, y_val_digit) in enumerate(zip(X_data[0], y_train_digits, X_data[1], y_val_digits)):
    print(f"Training baseline model for {digit_names[i]} digit")
    model = create_baseline_cnn(X_train[:5000].shape[1:])
    history = model.fit(X_train[:5000], y_train_digit[:5000], epochs=5, batch_size=64,
                        validation_data=(X_val[:1000], y_val_digit[:1000]), verbose=1)
    baseline_histories.append(history)

# === ENHANCED CNN TRAINING ===
def create_enhanced_cnn(input_shape):
    model = models.Sequential([
        layers.Conv2D(64, 3, activation='relu', padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Conv2D(64, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        layers.Dropout(0.25),
        layers.Conv2D(128, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        layers.Dropout(0.25),
        layers.Conv2D(256, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer=optimizers.Adam(0.0005), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

print("\n- TRAINING ENHANCED MODELS")
enhanced_models, enhanced_histories = [], []
datagens = [ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.1) for _ in range(3)]

for i, (X_train, y_train_digit, X_val, y_val_digit, X_test, y_test_digit, datagen) in enumerate(
    zip(X_data[0], y_train_digits, X_data[1], y_val_digits, X_data[2], y_test_digits, datagens)):
    
    print(f"Training enhanced model for {digit_names[i]} digit")
    model = create_enhanced_cnn(X_train.shape[1:])
    datagen.fit(X_train)
    
    history = model.fit(
        datagen.flow(X_train, y_train_digit, batch_size=64),
        epochs=15,
        validation_data=(X_val, y_val_digit),
        verbose=1,
        callbacks=[  # Only ReduceLROnPlateau kept
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=2, min_lr=0.00001
            )
        ]
    )

    print(f"Completed full 15 epochs for {digit_names[i]} digit.\n")
    enhanced_models.append(model)
    enhanced_histories.append(history)


print("- EVALUATING ENHANCED MODELS")
preds = [np.argmax(model.predict(X_data[2][i], verbose=0), axis=1) for i, model in enumerate(enhanced_models)]
accs = [accuracy_score(y_test_digits[i], preds[i]) for i in range(3)]
seq_acc = np.mean([(a==b and b==c) for a,b,c in zip(*preds)])

# - LEARNING CURVES AND EVALUATION VISUALS
# Save learning curves for enhanced models
for i, history in enumerate(enhanced_histories):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Val')
    plt.title(f'Enhanced - {digit_names[i]} Digit Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Val')
    plt.title(f'Enhanced - {digit_names[i]} Digit Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    save_figure(f'enhanced_learning_curve_digit{i+1}.png')

# Accuracy comparison chart
baseline_accs = [0.993, 0.101, 0.087]  # Task 3 baseline values
x = np.arange(3)
width = 0.35

plt.figure(figsize=(10, 5))
plt.bar(x - width/2, baseline_accs, width, label='Task 3 Baseline')
plt.bar(x + width/2, accs, width, label='Task 4 Enhanced')
plt.xticks(x, digit_names)
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.title('Per-Digit Accuracy Comparison')
plt.legend()
for i in range(3):
    plt.text(i - width/2, baseline_accs[i] + 0.01, f"{baseline_accs[i]:.3f}", ha='center')
    plt.text(i + width/2, accs[i] + 0.01, f"{accs[i]:.3f}", ha='center')
save_figure('per_digit_accuracy_comparison.png')

# Sequence accuracy comparison
plt.figure(figsize=(6, 5))
plt.bar(['Task 3', 'Task 4'], [0.012, seq_acc], color=['blue', 'orange'])
plt.title('Sequence Accuracy Comparison')
plt.ylabel('Accuracy')
plt.ylim(0, max(0.05, seq_acc * 1.2))
plt.text(0, 0.012 + 0.002, '0.012', ha='center')
plt.text(1, seq_acc + 0.002, f"{seq_acc:.3f}", ha='center')
save_figure('sequence_accuracy_comparison.png')

# Confusion matrices
plt.figure(figsize=(15, 5))
for i, (y_true, y_pred) in enumerate(zip(y_test_digits, preds)):
    plt.subplot(1, 3, i+1)
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {digit_names[i]} Digit')
    plt.xlabel('Predicted')
    plt.ylabel('True')
plt.tight_layout()
save_figure('confusion_matrices_enhanced.png')

print("\n- FINAL RESULTS")
for i, acc in enumerate(accs):
    print(f"{digit_names[i]} Digit Accuracy: {acc:.4f}")
print(f"Overall Sequence Accuracy: {seq_acc:.4f}")