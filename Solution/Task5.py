import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ======= PATH SETUP =======
base_dir = "C:\\Users\\samue\\OneDrive\Documents\\Github\\GitHub\\Machine-Learning\\triple_mnist"
train_dir = os.path.join(base_dir, "train")
output_dir = "C:\\Users\\samue\\OneDrive\Documents\\Github\\GitHub\\Machine-Learning\\Task 5"
model_path = os.path.join(output_dir, "task5_model.h5")
os.makedirs(output_dir, exist_ok=True)

# ======= DATA LOADING =======
print("Loading and preprocessing data...")
images, labels = [], []
folders = [f for f in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, f))]

for folder in folders:
    folder_path = os.path.join(train_dir, folder)
    for img_path in glob.glob(os.path.join(folder_path, "*.png")):
        digits = os.path.basename(img_path).split('_')[1].split('.')[0]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        h, w = img.shape
        for i in range(3):
            digit_img = img[:, i * w // 3:(i + 1) * w // 3]
            digit_img = cv2.resize(digit_img, (28, 28))
            digit_img = cv2.GaussianBlur(digit_img, (3, 3), 0)
            digit_img = cv2.adaptiveThreshold(digit_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                              cv2.THRESH_BINARY, 11, 2)
            if np.mean(digit_img) > 127:
                digit_img = 255 - digit_img
            digit_img = digit_img.astype('float32') / 255.0
            digit_img = np.expand_dims(digit_img, axis=-1)
            images.append(digit_img)
            labels.append(int(digits[i]))

X = np.array(images)
y = np.array(labels)
print(f"Total digits: {len(X)}")

# ======= ENCODE & SPLIT =======
lb = LabelBinarizer()
y_encoded = lb.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, stratify=y, random_state=42)

# ======= AUGMENTATION =======
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)
datagen.fit(X_train)

# ======= MODEL DEFINITION =======
initial_lr = 0.001
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=initial_lr,
    decay_steps=1000,
    decay_rate=0.9,
    staircase=True
)

model = keras.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, (3, 3), padding='same'),
    layers.BatchNormalization(), layers.Activation('relu'),
    layers.Conv2D(32, (3, 3), padding='same'),
    layers.BatchNormalization(), layers.Activation('relu'),
    layers.MaxPooling2D((2, 2)), layers.Dropout(0.25),

    layers.Conv2D(64, (3, 3), padding='same'),
    layers.BatchNormalization(), layers.Activation('relu'),
    layers.Conv2D(64, (3, 3), padding='same'),
    layers.BatchNormalization(), layers.Activation('relu'),
    layers.MaxPooling2D((2, 2)), layers.Dropout(0.25),

    layers.Flatten(),
    layers.Dense(128), layers.BatchNormalization(), layers.Activation('relu'), layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ======= TRAINING =======
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=64),
    validation_data=(X_test, y_test),
    epochs=15,
    verbose=1
)

model.save(model_path)
print(f"Saved model to: {model_path}")

# ======= TRAINING PLOTS =======
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.title('Accuracy')
plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.title('Loss')
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'training_plots.png'))
plt.close()

# ======= EVALUATION =======
predictions = model.predict(X_test)
y_pred = np.argmax(predictions, axis=1)
y_true = np.argmax(y_test, axis=1)

print("\nClassification Report:")
print(classification_report(y_true, y_pred))

# ======= CONFUSION MATRIX =======
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
plt.title('Confusion Matrix')
plt.xlabel('Predicted'); plt.ylabel('True')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
plt.close()

# ======= DATA DISTRIBUTION =======
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(np.argmax(y_train, axis=1), bins=np.arange(11)-0.5, color='skyblue', edgecolor='white')
plt.title('Training Data Distribution')
plt.xlabel('Digit'); plt.ylabel('Count')
plt.xticks(range(10)); plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.hist(np.argmax(y_test, axis=1), bins=np.arange(11)-0.5, color='salmon', edgecolor='white')
plt.title('Test Data Distribution')
plt.xlabel('Digit'); plt.ylabel('Count')
plt.xticks(range(10)); plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'data_distribution.png'))
plt.close()

# ======= TOP CONFUSED PAIRS =======
confused_pairs = [(i, j, cm[i, j]) for i in range(10) for j in range(10) if i != j and cm[i, j] > 0]
confused_pairs.sort(key=lambda x: x[2], reverse=True)
top_confused = confused_pairs[:10]

if top_confused:
    labels = [f'{p[0]}→{p[1]}' for p in top_confused]
    counts = [p[2] for p in top_confused]
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=counts, y=labels, palette='Reds_r')
    plt.title('Top 10 Most Confused Digit Pairs')
    plt.xlabel('Misclassification Count')
    plt.ylabel('Digit Pair')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_confused_pairs.png'))
    plt.close()

# ======= HARDEST EXAMPLES =======
incorrect_indices = np.where(y_pred != y_true)[0]
if len(incorrect_indices) > 0:
    confidences = np.max(predictions, axis=1)
    hardest_indices = incorrect_indices[np.argsort(confidences[incorrect_indices])[:8]]
    
    plt.figure(figsize=(10, 5))
    for i, idx in enumerate(hardest_indices):
        plt.subplot(2, 4, i + 1)
        plt.imshow(X_test[idx].reshape(28, 28), cmap='gray')
        plt.title(f'P:{y_pred[idx]} T:{y_true[idx]}\nConf:{confidences[idx]:.2f}', fontsize=8)
        plt.axis('off')
    
    plt.suptitle('Hardest Misclassifications (Lowest Confidence)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'hardest_examples.png'))
    plt.close()