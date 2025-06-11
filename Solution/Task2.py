import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import seaborn as sns

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

# Define dataset and save directories
base_dir = os.path.join(script_dir, "triple_mnist")
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")
test_dir = os.path.join(base_dir, "test")

# Create directory to save outputs
save_dir = os.path.join(script_dir, "Task 2")
os.makedirs(save_dir, exist_ok=True)

# - LOAD TRAINING DATA
# Extract the individual digits from the directory name
print("\n- LOADING DATA")
def extract_digits(label):
    return tuple(map(int, label))

# Load image data and labels from the specified directory
def load_data(dir, flatten=True, max_samples=None):
    print(f"Loading data from: {dir}")
    images, d1, d2, d3, full = [], [], [], [], []
    for label in sorted(os.listdir(dir)):
        path = os.path.join(dir, label)
        if not os.path.isdir(path):
            continue
        try:
            a, b, c = extract_digits(label)
        except:
            print(f"Skipping invalid label: {label}")
            continue
        for img_path in sorted(glob.glob(os.path.join(path, "*.png"))):
            img = Image.open(img_path).convert("L")
            if img.size != (84, 84):
                continue
            arr = np.array(img) / 255.0
            images.append(arr.flatten() if flatten else arr)
            d1.append(a)
            d2.append(b)
            d3.append(c)
            full.append(label)
            if max_samples and len(images) >= max_samples:
                break
        if max_samples and len(images) >= max_samples:
            break
    print(f"Loaded {len(images)} samples from: {dir}")
    return np.array(images), np.array(d1), np.array(d2), np.array(d3), np.array(full)

# - LOADING DATASETS
X_train, y1_train, y2_train, y3_train, _ = load_data(train_dir, True, 30000)
X_val, y1_val, y2_val, y3_val, _ = load_data(val_dir, True, 8000)
X_test, y1_test, y2_test, y3_test, _ = load_data(test_dir, True, 8000)

# Print dataset info
print("\n- DATASET INFORMATION")
print(f"Training set: {X_train.shape[0]} images")
print(f"Validation set: {X_val.shape[0]} images")
print(f"Test set: {X_test.shape[0]} images")
print(f"Image vector size: {X_train.shape[1]}")
print(f"Unique values for first digit: {np.unique(y1_train)}")
print(f"Unique values for second digit: {np.unique(y2_train)}")
print(f"Unique values for third digit: {np.unique(y3_train)}")

# - VISUALIZE SAMPLE IMAGES
# Show 5 example images from the training set
plt.figure(figsize=(15, 4))
for i in range(5):
    img = X_train[i].reshape(84, 84)
    plt.subplot(1, 5, i + 1)
    plt.imshow(img, cmap="gray")
    plt.title(f"Label: {y1_train[i]}{y2_train[i]}{y3_train[i]}")
    plt.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "sample_images.png"))
plt.close()

# - PCA + DECISION TREE
print("\n- DECISION TREE MODEL")

# Apply PCA to reduce feature dimensionality
print("Applying PCA for dimensionality reduction")
pca = PCA(n_components=100).fit(X_train)
X_train_pca, X_val_pca, X_test_pca = (
    pca.transform(X_train),
    pca.transform(X_val),
    pca.transform(X_test),
)
print(f"Reduced feature dimension from {X_train.shape[1]} to {X_train_pca.shape[1]}")
print(f"Explained variance ratio: {np.sum(pca.explained_variance_ratio_):.4f}")

# Train a decision tree for each digit position
def train_dt(X_train, y_train, X_val, y_val, X_test, y_test, digit_position):
    print(f"\nTraining Decision Tree for digit position {digit_position+1}")
    model = DecisionTreeClassifier(max_depth=20, random_state=42).fit(X_train, y_train)
    val_acc = accuracy_score(y_val, model.predict(X_val))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    print(f"Validation accuracy for digit {digit_position+1}: {val_acc:.4f}")
    print(f"Test accuracy for digit {digit_position+1}: {test_acc:.4f}")
    return model, test_acc

dt_models, dt_accuracies = [], []
for i, (yt, yv, yte) in enumerate(
    [
        (y1_train, y1_val, y1_test),
        (y2_train, y2_val, y2_test),
        (y3_train, y3_val, y3_test),
    ]
):
    m, acc = train_dt(X_train_pca, yt, X_val_pca, yv, X_test_pca, yte, i)
    dt_models.append(m)
    dt_accuracies.append(acc)

# Compute sequence accuracy for all digits
dt_preds = [m.predict(X_test_pca) for m in dt_models]
dt_seq_acc = np.mean(
    [
        (a == y1_test[i]) and (b == y2_test[i]) and (c == y3_test[i])
        for i, (a, b, c) in enumerate(zip(*dt_preds))
    ]
)

# Print summary results for Decision Tree
print(f"\nDecision Tree overall sequence accuracy: {dt_seq_acc:.4f}")
print("\n- DECISION TREE SUMMARY")
print(f"Per-digit test accuracies: {[f'{a:.4f}' for a in dt_accuracies]}")
print(f"Sequence accuracy: {dt_seq_acc:.4f}")

# - CNN
print("\n- CNN MODEL")

# Build a CNN model
def cnn_model():
    model = models.Sequential(
        [
            layers.Conv2D(
                32, 3, activation="relu", padding="same", input_shape=(84, 84, 1)
            ),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2),
            layers.Conv2D(64, 3, activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2),
            layers.Conv2D(128, 3, activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=optimizers.Adam(0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

# Reload data for CNN input (unflattened and reshaped)
X_train_cnn, y1_t, y2_t, y3_t, _ = load_data(train_dir, False, 30000)
X_val_cnn, y1_v, y2_v, y3_v, _ = load_data(val_dir, False, 8000)
X_test_cnn, y1_te, y2_te, y3_te, _ = load_data(test_dir, False, 8000)
X_train_cnn, X_val_cnn, X_test_cnn = [
    x.reshape(-1, 84, 84, 1) for x in [X_train_cnn, X_val_cnn, X_test_cnn]
]

# Train a CNN per digit and store results
cnn_accuracies, cnn_preds, histories = [], [], []
for i, (yt, yv, yte) in enumerate(
    [(y1_t, y1_v, y1_te), (y2_t, y2_v, y2_te), (y3_t, y3_v, y3_te)]
):
    print(f"\n- TRAINING CNN FOR DIGIT POSITION {i+1}")
    model = cnn_model()
    history = model.fit(
        X_train_cnn,
        yt,
        epochs=10,
        batch_size=64,
        validation_data=(X_val_cnn, yv),
        verbose=1,
    )
    acc = model.evaluate(X_test_cnn, yte, verbose=0)[1]
    print(f"Test accuracy for digit {i+1}: {acc:.4f}")
    pred = np.argmax(model.predict(X_test_cnn, verbose=0), axis=1)
    cnn_accuracies.append(acc)
    cnn_preds.append(pred)
    histories.append(history)

# Compute sequence accuracy for CNN
cnn_seq_acc = np.mean(
    [
        (a == y1_te[i]) and (b == y2_te[i]) and (c == y3_te[i])
        for i, (a, b, c) in enumerate(zip(*cnn_preds))
    ]
)

# Print summary results for CNN
print(f"\nCNN overall sequence accuracy: {cnn_seq_acc:.4f}")
print("\n- CNN SUMMARY")
print(f"Per-digit test accuracies: {[f'{a:.4f}' for a in cnn_accuracies]}")
print(f"Sequence accuracy: {cnn_seq_acc:.4f}")

# - LEARNING CURVES
# Save learning curves for each digit model
for i, history in enumerate(histories):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Train")
    plt.plot(history.history["val_accuracy"], label="Val")
    plt.title(f"Accuracy - Digit {i+1}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Train")
    plt.plot(history.history["val_loss"], label="Val")
    plt.title(f"Loss - Digit {i+1}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"cnn_learning_curves_digit{i+1}.png"))
    plt.close()

# - BAR CHARTS AND CONFUSION MATRICES
# Save bar chart comparing per-digit model accuracy
def save_bar_chart(x_labels, dt_vals, cnn_vals, title, ylabel, filename):
    x = np.arange(len(x_labels))
    width = 0.35
    plt.figure(figsize=(10, 6))
    plt.bar(x - width / 2, dt_vals, width, label="Decision Tree")
    plt.bar(x + width / 2, cnn_vals, width, label="CNN")
    plt.xticks(x, x_labels)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.ylim(0, 1)
    for i, v in enumerate(dt_vals):
        plt.text(i - width / 2, v + 0.02, f"{v:.3f}", ha="center")
    for i, v in enumerate(cnn_vals):
        plt.text(i + width / 2, v + 0.02, f"{v:.3f}", ha="center")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()

# Save sequence accuracy comparison bar
def save_sequence_accuracy_bar():
    plt.figure(figsize=(6, 5))
    accs = [dt_seq_acc, cnn_seq_acc]
    models = ["Decision Tree", "CNN"]
    plt.bar(models, accs, color=["blue", "orange"])
    plt.ylabel("Accuracy")
    plt.title("Sequence Accuracy Comparison")
    for i, v in enumerate(accs):
        plt.text(i, v + 0.002, f"{v:.4f}", ha="center")
    plt.ylim(0.0, 0.1)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "sequence_accuracy.png"))
    plt.close()

# Save confusion matrices for digit 1
def save_confusion_matrices():
    cm_dt = confusion_matrix(y1_test, dt_preds[0])
    cm_cnn = confusion_matrix(y1_te, cnn_preds[0])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    sns.heatmap(cm_dt, annot=True, fmt="d", cmap="Blues", ax=ax1)
    ax1.set_title("DT Confusion Matrix - Digit 1")
    sns.heatmap(cm_cnn, annot=True, fmt="d", cmap="Blues", ax=ax2)
    ax2.set_title("CNN Confusion Matrix - Digit 1")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrices.png"))
    plt.close()

# Generate final outputs
save_bar_chart(
    ["Digit 1", "Digit 2", "Digit 3"],
    dt_accuracies,
    cnn_accuracies,
    "Per-Digit Accuracy Comparison",
    "Accuracy",
    "per_digit_accuracy.png",
)
save_sequence_accuracy_bar()
save_confusion_matrices()

# - COMPLETION
# Notify user of save location
print("\nAll plots and visual outputs saved to:", save_dir)