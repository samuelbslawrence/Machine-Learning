import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix, classification_report
import glob
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

# Define dataset and output directories
base_dir = os.path.join(script_dir, "triple_mnist")
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")
test_dir = os.path.join(base_dir, "test")

# Create output directory for Task 5
output_dir = os.path.join(script_dir, "Task 5")
os.makedirs(output_dir, exist_ok=True)

# Model save path
model_path = os.path.join(output_dir, "Task5_Model.h5")

plt.style.use("default")

class DigitRecognizer:
    def __init__(self):
        self.model = None
        self.history = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.label_binarizer = LabelBinarizer()
        self.best_accuracy = 0

    def load_and_preprocess_data(self):
        print("\n- LOADING DATA")

        if not os.path.exists(train_dir):
            raise FileNotFoundError(f"Training directory not found: {train_dir}")

        images = []
        labels = []

        folders = [
            f
            for f in os.listdir(train_dir)
            if os.path.isdir(os.path.join(train_dir, f))
        ]

        if not folders:
            raise ValueError(f"No folders found in training directory: {train_dir}")

        print(f"Found {len(folders)} folders to process...")

        for folder_idx, folder in enumerate(folders):
            folder_path = os.path.join(train_dir, folder)
            image_files = glob.glob(os.path.join(folder_path, "*.png"))

            print(
                f"\rProcessing folder {folder_idx+1}/{len(folders)}: {folder} ({len(image_files)} images)",
                end="",
            )

            for img_file in image_files[:60]:
                filename = os.path.basename(img_file)
                digits_str = filename.split("_")[1].split(".")[0]

                img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue

                # Split image into individual digits
                height, width = img.shape
                digit_width = width // 3

                for i, digit_label in enumerate(digits_str):
                    x_start = max(0, i * digit_width - 2)
                    x_end = min(width, (i + 1) * digit_width + 2)
                    digit_img = img[:, x_start:x_end]

                    digit_img = self.preprocess_digit(digit_img)

                    images.append(digit_img)
                    labels.append(int(digit_label))

        print(f"\n\nPreprocessing complete!")

        # Convert to numpy arrays
        self.X = np.array(images)
        self.y = np.array(labels)

        # Reshape for CNN
        self.X = self.X.reshape(-1, 28, 28, 1)

        # One-hot encode labels
        self.y_encoded = self.label_binarizer.fit_transform(self.y)

        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y_encoded, test_size=0.2, random_state=42, stratify=self.y
        )

        print(f"Dataset Statistics:")
        print(f"  - Total digit images: {len(self.X)}")
        print(f"  - Training samples: {len(self.X_train)}")
        print(f"  - Test samples: {len(self.X_test)}")
        print(f"  - Number of classes: 10 (digits 0-9)")
        print(f"  - Image shape: 28x28 pixels")

        self.save_data_distribution()

    def preprocess_digit(self, digit_img):
        digit_img = cv2.resize(digit_img, (28, 28))
        digit_img = cv2.GaussianBlur(digit_img, (3, 3), 0)
        digit_img = cv2.adaptiveThreshold(
            digit_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        if np.mean(digit_img) > 127:
            digit_img = 255 - digit_img

        digit_img = digit_img.astype("float32") / 255.0
        return digit_img

    def save_data_distribution(self):
        train_digits = np.argmax(self.y_train, axis=1)
        test_digits = np.argmax(self.y_test, axis=1)

        # Training distribution
        plt.figure(figsize=(8, 5))
        plt.hist(train_digits, bins=10, color="skyblue", edgecolor="black", alpha=0.7)
        plt.title("Training Data Distribution", fontsize=14)
        plt.xlabel("Digit")
        plt.ylabel("Count")
        plt.xticks(range(10))
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "train_distribution.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

        # Test distribution
        plt.figure(figsize=(8, 5))
        plt.hist(test_digits, bins=10, color="lightcoral", edgecolor="black", alpha=0.7)
        plt.title("Test Data Distribution", fontsize=14)
        plt.xlabel("Digit")
        plt.ylabel("Count")
        plt.xticks(range(10))
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "test_distribution.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

    def create_model(self):
        print("\n- BUILDING NEURAL NETWORK MODEL")

        self.model = keras.Sequential(
            [
                layers.Input(shape=(28, 28, 1)),
                # First Conv Block
                layers.Conv2D(32, (3, 3), padding="same"),
                layers.BatchNormalization(),
                layers.Activation("relu"),
                layers.Conv2D(32, (3, 3), padding="same"),
                layers.BatchNormalization(),
                layers.Activation("relu"),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),
                # Second Conv Block
                layers.Conv2D(64, (3, 3), padding="same"),
                layers.BatchNormalization(),
                layers.Activation("relu"),
                layers.Conv2D(64, (3, 3), padding="same"),
                layers.BatchNormalization(),
                layers.Activation("relu"),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),
                # Third Conv Block
                layers.Conv2D(128, (3, 3), padding="same"),
                layers.BatchNormalization(),
                layers.Activation("relu"),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),
                # Dense layers
                layers.Flatten(),
                layers.Dense(256),
                layers.BatchNormalization(),
                layers.Activation("relu"),
                layers.Dropout(0.5),
                layers.Dense(128),
                layers.BatchNormalization(),
                layers.Activation("relu"),
                layers.Dropout(0.5),
                layers.Dense(10, activation="softmax"),
            ]
        )

        initial_learning_rate = 0.001
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate, decay_steps=1000, decay_rate=0.9, staircase=True
        )

        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
            loss="categorical_crossentropy",
            metrics=[
                "accuracy",
                keras.metrics.Precision(name="precision"),
                keras.metrics.Recall(name="recall"),
            ],
        )

        print("\nModel Architecture:")
        self.model.summary()

    def save_data_augmentation_examples(self):
        datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            fill_mode="nearest",
        )

        plt.figure(figsize=(12, 6))
        sample_idx = np.random.randint(0, len(self.X_train))
        original = self.X_train[sample_idx]

        # Original image
        plt.subplot(2, 3, 1)
        plt.imshow(original.reshape(28, 28), cmap="gray")
        plt.title("Original")
        plt.axis("off")

        # Augmented versions
        for i in range(5):
            plt.subplot(2, 3, i + 2)
            augmented = datagen.random_transform(original)
            plt.imshow(augmented.reshape(28, 28), cmap="gray")
            plt.title(f"Augmented {i+1}")
            plt.axis("off")

        plt.suptitle("Data Augmentation Examples", fontsize=16)
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "data_augmentation_examples.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

    def train_model(self):
        print("\n- STARTING TRAINING")

        # Create data augmentation
        datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            fill_mode="nearest",
        )

        # Save augmentation examples
        self.save_data_augmentation_examples()

        # Custom callback for saving plots
        class PlotCallback(keras.callbacks.Callback):
            def __init__(self, recognizer):
                self.recognizer = recognizer
                self.losses = []
                self.val_losses = []
                self.accs = []
                self.val_accs = []
                self.lrs = []

            def on_epoch_end(self, epoch, logs=None):
                self.losses.append(logs["loss"])
                self.val_losses.append(logs["val_loss"])
                self.accs.append(logs["accuracy"])
                self.val_accs.append(logs["val_accuracy"])
                self.lrs.append(float(self.model.optimizer.learning_rate))

                # Save plots every 5 epochs
                if (epoch + 1) % 5 == 0:
                    self.save_training_plots(epoch + 1)
                    self.save_live_predictions(epoch + 1)

                # Update best accuracy
                if logs["val_accuracy"] > self.recognizer.best_accuracy:
                    self.recognizer.best_accuracy = logs["val_accuracy"]

            def save_training_plots(self, epoch):
                # Loss plot
                plt.figure(figsize=(10, 5))
                plt.subplot(1, 2, 1)
                plt.plot(self.losses, "b-", label="Training Loss")
                plt.plot(self.val_losses, "r-", label="Validation Loss")
                plt.title(f"Training Progress - Loss (Epoch {epoch})")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.legend()
                plt.grid(True, alpha=0.3)

                # Accuracy plot
                plt.subplot(1, 2, 2)
                plt.plot(self.accs, "g-", label="Training Accuracy")
                plt.plot(self.val_accs, "orange", label="Validation Accuracy")
                plt.title(f"Training Progress - Accuracy (Epoch {epoch})")
                plt.xlabel("Epoch")
                plt.ylabel("Accuracy")
                plt.legend()
                plt.grid(True, alpha=0.3)

                plt.tight_layout()
                plt.savefig(
                    os.path.join(output_dir, f"training_progress_epoch_{epoch}.png"),
                    dpi=150,
                    bbox_inches="tight",
                )
                plt.close()

                # Learning rate plot
                plt.figure(figsize=(8, 5))
                plt.plot(self.lrs, "y-", linewidth=2)
                plt.title(f"Learning Rate Schedule (Epoch {epoch})")
                plt.xlabel("Epoch")
                plt.ylabel("Learning Rate")
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(
                    os.path.join(output_dir, f"learning_rate_epoch_{epoch}.png"),
                    dpi=150,
                    bbox_inches="tight",
                )
                plt.close()

            def save_live_predictions(self, epoch):
                # Get random test samples and make predictions
                indices = np.random.choice(len(self.recognizer.X_test), 16)

                plt.figure(figsize=(12, 8))
                for i, idx in enumerate(indices):
                    plt.subplot(4, 4, i + 1)
                    img = self.recognizer.X_test[idx].reshape(28, 28)
                    plt.imshow(img, cmap="gray")

                    pred = self.model.predict(
                        self.recognizer.X_test[idx : idx + 1], verbose=0
                    )
                    pred_label = np.argmax(pred)
                    true_label = np.argmax(self.recognizer.y_test[idx])
                    confidence = pred[0][pred_label]

                    color = "green" if pred_label == true_label else "red"
                    plt.title(
                        f"P:{pred_label} ({confidence:.2f})\nT:{true_label}",
                        fontsize=10,
                        color=color,
                    )
                    plt.axis("off")

                plt.suptitle(f"Live Predictions - Epoch {epoch}", fontsize=16)
                plt.tight_layout()
                plt.savefig(
                    os.path.join(output_dir, f"live_predictions_epoch_{epoch}.png"),
                    dpi=150,
                    bbox_inches="tight",
                )
                plt.close()

        plot_callback = PlotCallback(self)

        # Other callbacks
        early_stopping = keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=10, restore_best_weights=True, verbose=1
        )

        checkpoint = keras.callbacks.ModelCheckpoint(
            model_path, monitor="val_accuracy", save_best_only=True, verbose=1
        )

        # Train the model
        self.history = self.model.fit(
            datagen.flow(self.X_train, self.y_train, batch_size=32),
            validation_data=(self.X_test, self.y_test),
            epochs=50,
            callbacks=[plot_callback, early_stopping, checkpoint],
            verbose=1,
        )

    def analyze_model(self):
        print("\n- COMPREHENSIVE MODEL ANALYSIS")

        # Load best model
        self.model = keras.models.load_model(model_path)

        # Evaluate on test set
        test_results = self.model.evaluate(self.X_test, self.y_test, verbose=0)

        print(f"\nTest Results:")
        print(f"  - Loss: {test_results[0]:.4f}")
        print(f"  - Accuracy: {test_results[1]:.4f}")
        if len(test_results) > 2:
            print(f"  - Precision: {test_results[2]:.4f}")
        if len(test_results) > 3:
            print(f"  - Recall: {test_results[3]:.4f}")

        # Get predictions
        predictions = self.model.predict(self.X_test, verbose=0)
        predicted_labels = np.argmax(predictions, axis=1)
        true_labels = np.argmax(self.y_test, axis=1)

        # Classification report
        print("\nDetailed Classification Report:")
        print(
            classification_report(
                true_labels, predicted_labels, target_names=[str(i) for i in range(10)]
            )
        )

        # Save individual analysis plots
        self.save_confusion_matrix(true_labels, predicted_labels)
        self.save_class_accuracy(true_labels, predicted_labels)
        self.save_confidence_distribution(predictions, predicted_labels, true_labels)
        self.save_confused_pairs(true_labels, predicted_labels)
        self.save_sample_predictions(predicted_labels, true_labels, predictions)
        self.save_hardest_examples(predicted_labels, true_labels, predictions)

    def save_confusion_matrix(self, true_labels, predicted_labels):
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(true_labels, predicted_labels)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", square=True)
        plt.title("Confusion Matrix", fontsize=16)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "confusion_matrix.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

    def save_class_accuracy(self, true_labels, predicted_labels):
        plt.figure(figsize=(10, 6))
        class_accuracy = []
        for i in range(10):
            mask = true_labels == i
            if np.sum(mask) > 0:
                acc = np.mean(predicted_labels[mask] == i)
                class_accuracy.append(acc)
            else:
                class_accuracy.append(0)

        bars = plt.bar(range(10), class_accuracy, color="skyblue", edgecolor="black")
        plt.title("Per-Class Accuracy", fontsize=16)
        plt.xlabel("Digit")
        plt.ylabel("Accuracy")
        plt.ylim(0, 1.1)
        plt.xticks(range(10))
        plt.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, acc in zip(bars, class_accuracy):
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{acc:.3f}",
                ha="center",
                va="bottom",
            )

        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "class_accuracy.png"), dpi=150, bbox_inches="tight"
        )
        plt.close()

    def save_confidence_distribution(self, predictions, predicted_labels, true_labels):
        plt.figure(figsize=(10, 6))
        correct_confidences = []
        incorrect_confidences = []

        for i in range(len(predictions)):
            confidence = np.max(predictions[i])
            if predicted_labels[i] == true_labels[i]:
                correct_confidences.append(confidence)
            else:
                incorrect_confidences.append(confidence)

        plt.hist(
            correct_confidences,
            bins=20,
            alpha=0.7,
            label="Correct",
            color="green",
            edgecolor="black",
        )
        plt.hist(
            incorrect_confidences,
            bins=20,
            alpha=0.7,
            label="Incorrect",
            color="red",
            edgecolor="black",
        )
        plt.title("Prediction Confidence Distribution", fontsize=16)
        plt.xlabel("Confidence")
        plt.ylabel("Count")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "confidence_distribution.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

    def save_confused_pairs(self, true_labels, predicted_labels):
        cm = confusion_matrix(true_labels, predicted_labels)
        confused_pairs = []
        for i in range(10):
            for j in range(10):
                if i != j:
                    confusion_count = cm[i, j]
                    if confusion_count > 0:
                        confused_pairs.append((i, j, confusion_count))

        confused_pairs.sort(key=lambda x: x[2], reverse=True)
        top_confused = confused_pairs[:10]

        if top_confused:
            plt.figure(figsize=(10, 6))
            labels = [f"{p[0]}→{p[1]}" for p in top_confused]
            counts = [p[2] for p in top_confused]

            plt.barh(range(len(labels)), counts, color="coral", edgecolor="black")
            plt.yticks(range(len(labels)), labels)
            plt.xlabel("Misclassification Count")
            plt.title("Top 10 Most Confused Digit Pairs", fontsize=16)
            plt.grid(True, alpha=0.3, axis="x")
            plt.tight_layout()
            plt.savefig(
                os.path.join(output_dir, "confused_pairs.png"),
                dpi=150,
                bbox_inches="tight",
            )
            plt.close()

    def save_sample_predictions(self, predicted_labels, true_labels, predictions):
        plt.figure(figsize=(12, 8))
        sample_indices = np.random.choice(len(self.X_test), 16, replace=False)

        for i, idx in enumerate(sample_indices):
            plt.subplot(4, 4, i + 1)
            plt.imshow(self.X_test[idx].reshape(28, 28), cmap="gray")

            pred_label = predicted_labels[idx]
            true_label = true_labels[idx]
            confidence = predictions[idx][pred_label]

            color = "green" if pred_label == true_label else "red"
            plt.title(f"{pred_label}({confidence:.2f})", fontsize=10, color=color)
            plt.axis("off")

        plt.suptitle("Sample Predictions", fontsize=16)
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "sample_predictions.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

    def save_hardest_examples(self, predicted_labels, true_labels, predictions):
        incorrect_indices = np.where(predicted_labels != true_labels)[0]
        if len(incorrect_indices) > 0:
            confidences = [np.max(predictions[i]) for i in incorrect_indices]
            sorted_indices = incorrect_indices[np.argsort(confidences)][:16]

            plt.figure(figsize=(12, 8))
            for i, idx in enumerate(sorted_indices):
                plt.subplot(4, 4, i + 1)
                plt.imshow(self.X_test[idx].reshape(28, 28), cmap="gray")

                pred_label = predicted_labels[idx]
                true_label = true_labels[idx]
                confidence = predictions[idx][pred_label]

                plt.title(
                    f"P:{pred_label}({confidence:.2f})\nT:{true_label}",
                    fontsize=9,
                    color="red",
                )
                plt.axis("off")

            plt.suptitle("Hardest Examples (Lowest Confidence Errors)", fontsize=16)
            plt.tight_layout()
            plt.savefig(
                os.path.join(output_dir, "hardest_examples.png"),
                dpi=150,
                bbox_inches="tight",
            )
            plt.close()

# - MAIN EXECUTION
def main():
    recognizer = DigitRecognizer()

    try:
        # Load and preprocess data
        recognizer.load_and_preprocess_data()

        # Create model
        recognizer.create_model()

        # Train model
        recognizer.train_model()

        # Analyze model
        recognizer.analyze_model()

        print("\n- TRAINING COMPLETE")
        print(f"All outputs saved to: {output_dir}")
        print(f"Model saved as: {model_path}")
        print(f"Best accuracy achieved: {recognizer.best_accuracy:.4f}")

    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        import traceback

        traceback.print_exc()

if __name__ == "__main__":
    main()