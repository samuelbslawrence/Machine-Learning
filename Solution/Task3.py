import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import seaborn as sns
from tqdm import tqdm

# - PATH SETUP
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
save_dir = os.path.join(script_dir, "Task 3")
os.makedirs(save_dir, exist_ok=True)

# - UTILITY FUNCTIONS
# Save figure to Task 3 directory with standard print confirmation
def save_figure(filename):
    path = os.path.join(save_dir, filename)
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")

# Extract three digits from a string label like '123' → (1, 2, 3)
def extract_digits(label):
    return tuple(int(d) for d in label)

# Split a single 84x84 image into three 28x84 vertical slices (one for each digit)
def split_image(img):
    return img[:, :28], img[:, 28:56], img[:, 56:]

# - DATA LOADING
# Load images from disk and split each one into three parts, storing individual digit labels
def load_split_images(directory, max_samples=None):
    print(f"Loading data from: {directory}")
    part1, part2, part3, labels = [], [], [], []
    count = 0

    label_dirs = sorted(
        [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    )
    for label in label_dirs:
        try:
            d1, d2, d3 = extract_digits(label)
        except:
            continue
        img_dir = os.path.join(directory, label)
        for img_path in sorted(glob.glob(os.path.join(img_dir, "*.png"))):
            img = Image.open(img_path).convert("L")
            arr = np.array(img) / 255.0
            if arr.shape != (84, 84):
                continue
            p1, p2, p3 = split_image(arr)
            part1.append(p1)
            part2.append(p2)
            part3.append(p3)
            labels.append((d1, d2, d3))
            count += 1
            if max_samples and count >= max_samples:
                break
        if max_samples and count >= max_samples:
            break

    print(f"Loaded {len(labels)} samples from: {directory}")
    return np.array(part1), np.array(part2), np.array(part3), np.array(labels)

# - LOADING DATASETS
print("\n- LOADING AND SPLITTING DATASETS")
X_train_p1, X_train_p2, X_train_p3, y_train = load_split_images(train_dir, 30000)
X_val_p1, X_val_p2, X_val_p3, y_val = load_split_images(val_dir, 8000)
X_test_p1, X_test_p2, X_test_p3, y_test = load_split_images(test_dir, 8000)

# - DATASET INFORMATION
# Extract digit labels for each digit position separately
y_train_d1, y_train_d2, y_train_d3 = y_train[:, 0], y_train[:, 1], y_train[:, 2]
y_val_d1, y_val_d2, y_val_d3 = y_val[:, 0], y_val[:, 1], y_val[:, 2]
y_test_d1, y_test_d2, y_test_d3 = y_test[:, 0], y_test[:, 1], y_test[:, 2]

print("\n- DATASET INFORMATION")
print(f"Training set: {len(y_train)} images")
print(f"Validation set: {len(y_val)} images")
print(f"Test set: {len(y_test)} images")
print(f"Split image size: {X_train_p1[0].shape}")

# - PREPROCESSING
# Add channel dimension for CNN (shape becomes: [batch, height, width, 1])
for arr in [
    X_train_p1,
    X_train_p2,
    X_train_p3,
    X_val_p1,
    X_val_p2,
    X_val_p3,
    X_test_p1,
    X_test_p2,
    X_test_p3,
]:
    arr.shape = (*arr.shape, 1)

# - CNN MODEL
# Define a CNN suitable for digit recognition on a 28x84 grayscale input
def cnn_model(input_shape):
    model = models.Sequential(
        [
            layers.Conv2D(
                32, 3, activation="relu", padding="same", input_shape=input_shape
            ),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2),
            layers.Conv2D(64, 3, activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2),
            layers.Conv2D(64, 3, activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=optimizers.Adam(0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

# - TRAINING MODELS
print("\n- TRAINING DIGIT RECOGNITION MODELS")
# Train one CNN for each of the three digit positions using split image parts
digit_names = ["Digit 1", "Digit 2", "Digit 3"]
data_parts = [
    (X_train_p1, y_train_d1, X_val_p1, y_val_d1, X_test_p1, y_test_d1),
    (X_train_p2, y_train_d2, X_val_p2, y_val_d2, X_test_p2, y_test_d2),
    (X_train_p3, y_train_d3, X_val_p3, y_val_d3, X_test_p3, y_test_d3),
]
models_trained, histories, predictions, test_accuracies = [], [], [], []

for i, (X_tr, y_tr, X_val, y_val, X_te, y_te) in enumerate(data_parts):
    print(f"Training model for {digit_names[i]}")
    model = cnn_model(X_tr.shape[1:])
    history = model.fit(
        X_tr, y_tr, epochs=10, batch_size=64, validation_data=(X_val, y_val), verbose=1
    )

    acc = model.evaluate(X_te, y_te, verbose=0)[1]
    pred = np.argmax(model.predict(X_te, verbose=0), axis=1)

    print(f"\nTest accuracy: {acc:.4f}")
    test_accuracies.append(acc)
    predictions.append(pred)
    models_trained.append(model)
    histories.append(history)

    # Save learning curves for each digit model
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Train")
    plt.plot(history.history["val_accuracy"], label="Val")
    plt.title(f"Accuracy - {digit_names[i]}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Train")
    plt.plot(history.history["val_loss"], label="Val")
    plt.title(f"Loss - {digit_names[i]}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    save_figure(f"learning_curve_digit{i+1}.png")

# - SEQUENCE ACCURACY
# Check how many predictions match all three digits at once (full label accuracy)
print("\n- EVALUATING COMBINED MODEL")
seq_acc = np.mean(
    [
        (a == y_test_d1[i]) and (b == y_test_d2[i]) and (c == y_test_d3[i])
        for i, (a, b, c) in enumerate(zip(*predictions))
    ]
)
print(f"Overall sequence accuracy: {seq_acc:.4f}")

# - COMPARISON WITH TASK 2
# Compare per-digit and sequence performance with baseline from Task 2
task2_accuracies = [0.993, 0.101, 0.087]
task2_seq = 0.012
improvements = [
    (new - old) * 100 for new, old in zip(test_accuracies, task2_accuracies)
]
seq_improvement = (seq_acc - task2_seq) * 100

print("\n- COMPARISON WITH TASK 2")
for i, imp in enumerate(improvements):
    print(f"{digit_names[i]} improvement: {imp:.2f}%")
print(f"Sequence accuracy improvement: {seq_improvement:.2f}%")

# - VISUALIZE COMPARISON
# Bar charts comparing accuracy and sequence performance side by side
plt.figure(figsize=(12, 5))
x = np.arange(3)
width = 0.35
plt.subplot(1, 2, 1)
plt.bar(x - width / 2, task2_accuracies, width, label="Task 2 CNN")
plt.bar(x + width / 2, test_accuracies, width, label="Task 3 Split CNN")
plt.xticks(x, digit_names)
plt.ylabel("Accuracy")
plt.title("Digit Accuracy Comparison")
plt.legend()
for i in range(3):
    plt.text(
        i - width / 2,
        task2_accuracies[i] + 0.02,
        f"{task2_accuracies[i]:.3f}",
        ha="center",
    )
    plt.text(
        i + width / 2,
        test_accuracies[i] + 0.02,
        f"{test_accuracies[i]:.3f}",
        ha="center",
    )

plt.subplot(1, 2, 2)
plt.bar(["Task 2", "Task 3"], [task2_seq, seq_acc])
plt.ylabel("Sequence Accuracy")
plt.title("Sequence Accuracy Comparison")
for i, v in enumerate([task2_seq, seq_acc]):
    plt.text(i, v + 0.005, f"{v:.3f}", ha="center")

plt.tight_layout()
save_figure("accuracy_comparison.png")

# - CONFUSION MATRICES
# Generate confusion matrices for each digit classifier
plt.figure(figsize=(15, 5))
for i, (y_true, y_pred) in enumerate(
    zip([y_test_d1, y_test_d2, y_test_d3], predictions)
):
    plt.subplot(1, 3, i + 1)
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {digit_names[i]}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
plt.tight_layout()
save_figure("confusion_matrices.png")

# - COMPLETION
print("\nAll outputs and visualizations saved to:", save_dir)