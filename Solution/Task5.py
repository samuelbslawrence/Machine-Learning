import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.utils import set_random_seed, Progbar
import seaborn as sns
import random
import warnings
import time

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
set_random_seed(42)
np.random.seed(42)
random.seed(42)

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
output_dir = os.path.join(script_dir, "Task 5")
os.makedirs(output_dir, exist_ok=True)

# - UTILITY FUNCTIONS
# Define a function to save matplotlib figures neatly to disk
def save_figure(filename):
    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")

# Test save function
plt.figure(figsize=(6, 4))
plt.plot([1, 2, 3, 4], [1, 4, 2, 3])
plt.title("Task 5 Started - Save Function Working")
save_figure("00_save_test.png")

# - DATA LOADING FUNCTIONS
def load_images_for_gan(directory, max_samples=5000):
    """Load images in format suitable for GAN training (84x84)"""
    print(f"Loading GAN training data from: {directory}")
    images, labels = [], []
    count = 0

    label_dirs = sorted([d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))])
    
    # Calculate total number of potential images for progress bar
    total_potential = min(50 * 50, max_samples) if max_samples else 50 * 50  # 50 labels * 50 images max
    progbar = Progbar(total_potential, stateful_metrics=['loaded'])
    
    for label_idx, label in enumerate(label_dirs[:50]):
        try:
            d1, d2, d3 = tuple(int(d) for d in label)
        except:
            continue
        
        img_dir = os.path.join(directory, label)
        for img_idx, img_path in enumerate(glob.glob(os.path.join(img_dir, "*.png"))[:50]):
            img = Image.open(img_path).convert("L")
            arr = np.array(img) / 255.0
            if arr.shape != (84, 84):
                continue
            images.append(arr)
            labels.append(label)
            count += 1
            
            # Update progress bar
            progbar.update(count, values=[('loaded', count)])
            
            if max_samples and count >= max_samples:
                break
        if max_samples and count >= max_samples:
            break

    print(f"\nLoaded {len(images)} images for GAN training")
    return np.array(images), labels

def load_images_for_classifier(directory, max_samples=None):
    """Load images for classifier with multi-output for each digit position"""
    print(f"Loading classifier data from: {directory}")
    images = []
    # Three sets of labels (one for each digit position)
    labels_pos1 = []
    labels_pos2 = []
    labels_pos3 = []
    label_strings = []
    count = 0
    
    label_dirs = sorted([d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))])
    
    # Estimate total images for progress bar
    total_estimate = len(label_dirs) * 10  # Rough estimate
    if max_samples:
        total_estimate = min(total_estimate, max_samples)
    
    progbar = Progbar(total_estimate, stateful_metrics=['loaded'])
    
    for label in label_dirs:
        try:
            d1, d2, d3 = tuple(int(d) for d in label)
            # Skip if any digit is not 0-9
            if not all(0 <= d <= 9 for d in [d1, d2, d3]):
                continue
        except:
            continue
        
        img_dir = os.path.join(directory, label)
        for img_path in glob.glob(os.path.join(img_dir, "*.png")):
            img = Image.open(img_path).convert("L")
            arr = np.array(img) / 255.0
            if arr.shape != (84, 84):
                continue
            images.append(arr)
            labels_pos1.append(d1)
            labels_pos2.append(d2)
            labels_pos3.append(d3)
            label_strings.append(label)
            count += 1
            
            # Update progress bar
            if count <= total_estimate:
                progbar.update(count, values=[('loaded', count)])
            
            if max_samples and count >= max_samples:
                break
        if max_samples and count >= max_samples:
            break
    
    images = np.array(images)
    
    # Convert to one-hot encoding (10 classes per position)
    num_classes = 10  # 0-9 for each position
    y_pos1 = tf.keras.utils.to_categorical(labels_pos1, num_classes)
    y_pos2 = tf.keras.utils.to_categorical(labels_pos2, num_classes)
    y_pos3 = tf.keras.utils.to_categorical(labels_pos3, num_classes)
    
    print(f"\nLoaded {len(images)} images for classifier")
    print(f"Number of classes per position: {num_classes}")
    
    return images, [y_pos1, y_pos2, y_pos3], label_strings

# - LOADING DATASETS
print("\n- LOADING DATASETS")
X_train_gan, train_labels_gan = load_images_for_gan(train_dir, max_samples=5000)
X_train_full, y_train_full, train_label_strings = load_images_for_classifier(train_dir, max_samples=10000)
X_test, y_test, test_label_strings = load_images_for_classifier(test_dir, max_samples=2000)

# - DATASET ANALYSIS
# Count individual digits (0-9) across all labels
print("\n- DATASET INFORMATION")
print(f"Training set for GAN: {len(X_train_gan)} images")
print(f"Training set for classifier: {len(X_train_full)} images")
print(f"Test set: {len(X_test)} images")
print(f"Image size: {X_train_gan[0].shape}")

digit_counts = {str(i): 0 for i in range(10)}
total_digits = 0

for label in train_label_strings:
    for digit in label:
        digit_counts[digit] += 1
        total_digits += 1

# Display individual digit counts
print("\n- DIGIT DISTRIBUTION")
for digit, count in sorted(digit_counts.items()):
    print(f"{digit}: {count} ({count/total_digits*100:.1f}%)")

# Visualize dataset distribution
plt.figure(figsize=(14, 6))

# Plot most common labels
plt.subplot(1, 2, 1)
unique_labels, label_counts = np.unique(train_label_strings, return_counts=True)
label_distribution = dict(zip(unique_labels, label_counts))
sorted_distribution = dict(sorted(label_distribution.items(), key=lambda x: x[1], reverse=True))
top_labels = list(sorted_distribution.keys())[:15]
top_counts = [sorted_distribution[label] for label in top_labels]
plt.bar(range(len(top_labels)), top_counts, tick_label=top_labels)
plt.title('Top 15 Most Common Digit Combinations')
plt.xlabel('Digit Combination')
plt.ylabel('Count')
plt.xticks(rotation=45)

# Analyze digit frequency
plt.subplot(1, 2, 2)
digits = sorted(digit_counts.keys())
freqs = [digit_counts[d] for d in digits]

plt.bar(digits, freqs)
plt.title('Individual Digit Frequency')
plt.xlabel('Digit')
plt.ylabel('Count')
plt.tight_layout()
save_figure("dataset_distribution.png")

# Reshape and normalize for GAN training
X_train_gan = X_train_gan.reshape(-1, 84, 84, 1)
X_train_gan = (X_train_gan - 0.5) * 2  # Normalize to [-1, 1] for tanh output

# Reshape for classifier (keeping [0,1] normalization)
X_train_full = X_train_full.reshape(-1, 84, 84, 1)
X_test = X_test.reshape(-1, 84, 84, 1)

# Save sample real images
plt.figure(figsize=(12, 8))
for i in range(16):
    plt.subplot(4, 4, i + 1)
    img = X_train_gan[i, :, :, 0]
    plt.imshow(img, cmap='gray')
    plt.title(f'Real: {train_labels_gan[i]}')
    plt.axis('off')
plt.suptitle('Sample Real Images from Dataset')
plt.tight_layout()
save_figure("real_samples.png")

# - GAN MODEL ARCHITECTURE
def build_generator(latent_dim=100):
    """Build the generator network for creating synthetic images"""
    model = models.Sequential([
        layers.Input(shape=(latent_dim,)),
        layers.Dense(21 * 21 * 128, use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),
        layers.Reshape((21, 21, 128)),
        
        layers.Conv2DTranspose(64, 4, strides=2, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),
        
        layers.Conv2DTranspose(32, 4, strides=2, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),
        
        layers.Conv2DTranspose(1, 4, strides=1, padding='same', use_bias=False, activation='tanh')
    ])
    return model

def build_discriminator():
    """Build the discriminator network for distinguishing real from fake images"""
    model = models.Sequential([
        layers.Input(shape=(84, 84, 1)),
        layers.Conv2D(32, 4, strides=2, padding='same'),
        layers.LeakyReLU(0.2),
        layers.Dropout(0.3),
        
        layers.Conv2D(64, 4, strides=2, padding='same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),
        layers.Dropout(0.3),
        
        layers.Conv2D(128, 4, strides=2, padding='same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),
        layers.Dropout(0.3),
        
        layers.Flatten(),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

class DCGAN:
    """Deep Convolutional GAN for generating synthetic triple-digit MNIST images"""
    def __init__(self, latent_dim=100):
        self.latent_dim = latent_dim
        self.generator = build_generator(latent_dim)
        self.discriminator = build_discriminator()
        
        self.gen_optimizer = optimizers.Adam(0.0002, beta_1=0.5)
        self.disc_optimizer = optimizers.Adam(0.0002, beta_1=0.5)
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        
        print("DCGAN initialized successfully")
        print(f"Generator parameters: {self.generator.count_params():,}")
        print(f"Discriminator parameters: {self.discriminator.count_params():,}")
    
    def discriminator_loss(self, real_output, fake_output):
        """Calculate discriminator loss"""
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        return real_loss + fake_loss
    
    def generator_loss(self, fake_output):
        """Calculate generator loss"""
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)
    
    @tf.function
    def train_step(self, images, batch_size):
        """Perform one training step for both generator and discriminator"""
        noise = tf.random.normal([batch_size, self.latent_dim])
        
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)
            
            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)
            
            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)
        
        gen_gradients = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        disc_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        
        self.gen_optimizer.apply_gradients(zip(gen_gradients, self.generator.trainable_variables))
        self.disc_optimizer.apply_gradients(zip(disc_gradients, self.discriminator.trainable_variables))
        
        return gen_loss, disc_loss
    
    def generate_images(self, num_images=16):
        """Generate synthetic images using the trained generator"""
        noise = tf.random.normal([num_images, self.latent_dim])
        generated_images = self.generator(noise, training=False)
        return generated_images.numpy()

# - GAN TRAINING
print("\n- INITIALIZING GAN")
gan = DCGAN(latent_dim=100)

# Save initial random generated images
initial_images = gan.generate_images(16)
plt.figure(figsize=(10, 8))
for i in range(16):
    plt.subplot(4, 4, i + 1)
    img = (initial_images[i, :, :, 0] + 1) / 2
    plt.imshow(img, cmap='gray')
    plt.title(f'Pre-train {i+1}')
    plt.axis('off')
plt.suptitle('Generated Images BEFORE Training (Random Noise)')
plt.tight_layout()
save_figure("pre_training_samples.png")

# Option to load a pre-trained model
load_pretrained = True
pretrained_model_path = os.path.join(output_dir, "Task5_Model.h5")

if load_pretrained and os.path.exists(pretrained_model_path):
    print("Loading pre-trained GAN model")
    try:
        gan.generator = tf.keras.models.load_model(pretrained_model_path)
        print(f"Successfully loaded generator from: {pretrained_model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        load_pretrained = False
else:
    load_pretrained = False
    
# Training parameters
EPOCHS = 100 if not load_pretrained else 0
BATCH_SIZE = 32
BUFFER_SIZE = len(X_train_gan)

gen_losses, disc_losses = [], []
sample_images_history = []

if not load_pretrained:
    print(f"\n- TRAINING GAN FOR {EPOCHS} EPOCHS")
    train_dataset = tf.data.Dataset.from_tensor_slices(X_train_gan)
    train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        epoch_start = time.time()
        epoch_gen_loss = []
        epoch_disc_loss = []
        
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        
        # Count batches for batch progress bar
        num_batches = len(X_train_gan) // BATCH_SIZE + (1 if len(X_train_gan) % BATCH_SIZE != 0 else 0)
        batch_progbar = Progbar(num_batches, stateful_metrics=['batch_gen_loss', 'batch_disc_loss'])
        
        batch_count = 0
        for batch in train_dataset:
            gen_loss, disc_loss = gan.train_step(batch, min(BATCH_SIZE, len(batch)))
            epoch_gen_loss.append(gen_loss.numpy())
            epoch_disc_loss.append(disc_loss.numpy())
            
            batch_count += 1
            batch_progbar.update(batch_count, values=[
                ('batch_gen_loss', gen_loss.numpy()),
                ('batch_disc_loss', disc_loss.numpy())
            ])
        
        avg_gen_loss = np.mean(epoch_gen_loss)
        avg_disc_loss = np.mean(epoch_disc_loss)
        gen_losses.append(avg_gen_loss)
        disc_losses.append(avg_disc_loss)
        
        # Save images every 10 epochs
        if (epoch + 1) % 10 == 0:
            sample_images = gan.generate_images(16)
            sample_images_history.append(sample_images.copy())
            
            plt.figure(figsize=(10, 8))
            for i in range(16):
                plt.subplot(4, 4, i + 1)
                img = (sample_images[i, :, :, 0] + 1) / 2
                plt.imshow(img, cmap='gray')
                plt.axis('off')
            plt.suptitle(f'Generated Images - Epoch {epoch + 1}')
            plt.tight_layout()
            save_figure(f'generated_epoch_{epoch + 1:03d}.png')
    
    total_time = time.time() - start_time
    print(f"\nGAN training completed in {total_time/60:.1f} minutes")
    
    # Save trained model
    model_path = os.path.join(output_dir, "gan_generator_model.h5")
    gan.generator.save(model_path)
    print(f"Generator model saved to: {model_path}")
    
    # Save training curves
    if gen_losses:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(gen_losses, label='Generator Loss', alpha=0.8)
        plt.plot(disc_losses, label='Discriminator Loss', alpha=0.8)
        plt.title('GAN Training Losses')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(gen_losses, label='Generator Loss', color='blue', alpha=0.8)
        plt.title('Generator Loss Detail')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        save_figure('training_curves.png')

# - SYNTHETIC IMAGE GENERATION
print("\n- GENERATING SYNTHETIC IMAGES")
num_synthetic = 5000
synthetic_images = []

batch_size_gen = 50
num_batches = num_synthetic // batch_size_gen

# Create progress bar for synthetic image generation
generation_progbar = Progbar(num_batches, stateful_metrics=['generated'])

for i in range(num_batches):
    batch_images = gan.generate_images(batch_size_gen)
    synthetic_images.extend(batch_images)
    generation_progbar.update(i + 1, values=[('generated', (i + 1) * batch_size_gen)])

synthetic_images = np.array(synthetic_images)
print(f"\nGenerated {len(synthetic_images)} synthetic images")

# Convert synthetic images from [-1, 1] to [0, 1] for classifier training
synthetic_images_normalized = (synthetic_images + 1) / 2

# Visualize final quality
plt.figure(figsize=(15, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    img = (synthetic_images[i, :, :, 0] + 1) / 2
    plt.imshow(img, cmap='gray')
    plt.title(f'Synthetic {i+1}')
    plt.axis('off')
plt.suptitle('Final Generated Synthetic Images', fontsize=16)
plt.tight_layout()
save_figure('final_synthetic_quality.png')

# - IMAGE QUALITY ANALYSIS
synthetic_mean = np.mean(synthetic_images)
synthetic_std = np.std(synthetic_images)
real_mean = np.mean(X_train_gan)
real_std = np.std(X_train_gan)

print(f"Synthetic images - Mean: {synthetic_mean:.4f}, Std: {synthetic_std:.4f}")
print(f"Real images - Mean: {real_mean:.4f}, Std: {real_std:.4f}")
print(f"Mean difference: {abs(synthetic_mean - real_mean):.4f}")
print(f"Std difference: {abs(synthetic_std - real_std):.4f}")

# Save distribution comparison
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(X_train_gan.flatten(), bins=50, alpha=0.7, label='Real Images', density=True, color='blue')
plt.hist(synthetic_images.flatten(), bins=50, alpha=0.7, label='Synthetic Images', density=True, color='red')
plt.title('Pixel Intensity Distribution Comparison')
plt.xlabel('Pixel Intensity')
plt.ylabel('Density')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
real_means = np.mean(X_train_gan, axis=(1,2,3))
synthetic_means = np.mean(synthetic_images, axis=(1,2,3))
plt.hist(real_means, bins=30, alpha=0.7, label='Real Images', density=True, color='blue')
plt.hist(synthetic_means, bins=30, alpha=0.7, label='Synthetic Images', density=True, color='red')
plt.title('Image-wise Mean Intensity Distribution')
plt.xlabel('Mean Intensity')
plt.ylabel('Density')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
save_figure('distribution_comparison.png')

# - CLASSIFIER MODEL ARCHITECTURE
def build_classifier_model(input_shape=(84, 84, 1), num_classes=10):
    """Build multi-output classifier for three-digit recognition"""
    # Shared feature extraction layers
    input_layer = layers.Input(shape=input_shape)
    
    # Feature extraction
    x = layers.Conv2D(32, (3, 3), activation='relu')(input_layer)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    x = layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    
    # Separate output heads for each digit position
    output1 = layers.Dense(num_classes, activation='softmax', name='digit1')(x)
    output2 = layers.Dense(num_classes, activation='softmax', name='digit2')(x)
    output3 = layers.Dense(num_classes, activation='softmax', name='digit3')(x)
    
    model = models.Model(inputs=input_layer, outputs=[output1, output2, output3])
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss=['categorical_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy'],
        metrics=['accuracy', 'accuracy', 'accuracy']  # One metrics entry per output
    )
    
    return model

# - BASELINE CLASSIFIER TRAINING
print("\n- TRAINING BASELINE CLASSIFIER (ORIGINAL DATA ONLY)")
classifier_baseline = build_classifier_model()

# Define callbacks
early_stopping = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=3,
    min_lr=0.00001
)

# Split training data into train and validation sets
train_size = int(0.8 * len(X_train_full))
X_train, X_val = X_train_full[:train_size], X_train_full[train_size:]
y_train = [y_train_full[0][:train_size], y_train_full[1][:train_size], y_train_full[2][:train_size]]
y_val = [y_train_full[0][train_size:], y_train_full[1][train_size:], y_train_full[2][train_size:]]

# Train the baseline model
print("Training baseline classifier for 20 epochs...")
baseline_history = classifier_baseline.fit(
    X_train, y_train,
    epochs=20,
    batch_size=64,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping, reduce_lr],
    verbose=0  # Fixed: Changed from verbose=1 to avoid duplicate progress bars
)
print("Baseline classifier training completed!")

# Save the baseline model
baseline_model_path = os.path.join(output_dir, "classifier_baseline.h5")
classifier_baseline.save(baseline_model_path)
print(f"Baseline classifier saved to: {baseline_model_path}")

# Evaluate baseline model on test set
baseline_scores = classifier_baseline.evaluate(X_test, y_test, verbose=0)  # Fixed: Changed from verbose=1
print(f"Baseline Test Loss: {baseline_scores[0]:.4f}")
print(f"Baseline Test Accuracy - Digit 1: {baseline_scores[4]:.4f}")
print(f"Baseline Test Accuracy - Digit 2: {baseline_scores[5]:.4f}")
print(f"Baseline Test Accuracy - Digit 3: {baseline_scores[6]:.4f}")
baseline_avg_accuracy = (baseline_scores[4] + baseline_scores[5] + baseline_scores[6]) / 3
print(f"Baseline Average Accuracy: {baseline_avg_accuracy:.4f}")

# - SYNTHETIC DATA LABELING AND AUGMENTATION
print("\n- LABELING SYNTHETIC IMAGES")
# Use trained baseline model to generate pseudo-labels for synthetic images

# Create progress bar for prediction
print("Generating predictions for synthetic images...")
batch_size_pred = 100
num_pred_batches = len(synthetic_images_normalized) // batch_size_pred + 1
pred_progbar = Progbar(num_pred_batches, stateful_metrics=['processed'])

synthetic_pred = [[], [], []]
for i in range(num_pred_batches):
    start_idx = i * batch_size_pred
    end_idx = min((i + 1) * batch_size_pred, len(synthetic_images_normalized))
    if start_idx >= len(synthetic_images_normalized):
        break
    
    batch_pred = classifier_baseline.predict(synthetic_images_normalized[start_idx:end_idx], verbose=0)
    synthetic_pred[0].append(batch_pred[0])
    synthetic_pred[1].append(batch_pred[1])
    synthetic_pred[2].append(batch_pred[2])
    
    pred_progbar.update(i + 1, values=[('processed', end_idx)])

# Concatenate all predictions
synthetic_pred = [np.concatenate(pred, axis=0) for pred in synthetic_pred]

# Filter by confidence threshold
confidence_threshold = 0.7  
high_confidence_indices = (np.max(synthetic_pred[0], axis=1) > confidence_threshold) & \
                         (np.max(synthetic_pred[1], axis=1) > confidence_threshold) & \
                         (np.max(synthetic_pred[2], axis=1) > confidence_threshold)

# Make sure we have at least some synthetic data
min_synthetic = 500
if np.sum(high_confidence_indices) < min_synthetic:
    # Sort by confidence and take top min_synthetic
    confidence_scores = np.mean([
        np.max(synthetic_pred[0], axis=1),
        np.max(synthetic_pred[1], axis=1),
        np.max(synthetic_pred[2], axis=1)
    ], axis=0)
    
    top_indices = np.argsort(confidence_scores)[-min_synthetic:]
    high_confidence_mask = np.zeros_like(high_confidence_indices)
    high_confidence_mask[top_indices] = True
    high_confidence_indices = high_confidence_mask

synthetic_filtered = synthetic_images_normalized[high_confidence_indices]
synthetic_labels_filtered = [
    synthetic_pred[0][high_confidence_indices],
    synthetic_pred[1][high_confidence_indices],
    synthetic_pred[2][high_confidence_indices]
]

print(f"Kept {len(synthetic_filtered)} synthetic images with confidence > {confidence_threshold}")

# Combine original and synthetic data
X_augmented = np.concatenate([X_train, synthetic_filtered], axis=0)
y_augmented = [
    np.concatenate([y_train[0], synthetic_labels_filtered[0]], axis=0),
    np.concatenate([y_train[1], synthetic_labels_filtered[1]], axis=0),
    np.concatenate([y_train[2], synthetic_labels_filtered[2]], axis=0)
]

# Shuffle the augmented dataset
indices = np.arange(len(X_augmented))
np.random.shuffle(indices)
X_augmented = X_augmented[indices]
y_augmented = [y[indices] for y in y_augmented]

print(f"Augmented dataset size: {len(X_augmented)} images")
print(f"Original dataset size: {len(X_train)} images")
print(f"Added {len(X_augmented) - len(X_train)} synthetic images")

# - AUGMENTED CLASSIFIER TRAINING
print("\n- TRAINING AUGMENTED CLASSIFIER (ORIGINAL + SYNTHETIC DATA)")
classifier_augmented = build_classifier_model()

# Train the augmented model
print("Training augmented classifier for 20 epochs...")
augmented_history = classifier_augmented.fit(
    X_augmented, y_augmented,
    epochs=20,
    batch_size=64,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping, reduce_lr],
    verbose=0  # Fixed: Changed from verbose=1 to avoid duplicate progress bars
)
print("Augmented classifier training completed!")

# Save the augmented model
augmented_model_path = os.path.join(output_dir, "classifier_augmented.h5")
classifier_augmented.save(augmented_model_path)
print(f"Augmented classifier saved to: {augmented_model_path}")

# Evaluate augmented model on test set
augmented_scores = classifier_augmented.evaluate(X_test, y_test, verbose=0)  # Fixed: Changed from verbose=1
print(f"Augmented Test Loss: {augmented_scores[0]:.4f}")
print(f"Augmented Test Accuracy - Digit 1: {augmented_scores[4]:.4f}")
print(f"Augmented Test Accuracy - Digit 2: {augmented_scores[5]:.4f}")
print(f"Augmented Test Accuracy - Digit 3: {augmented_scores[6]:.4f}")
augmented_avg_accuracy = (augmented_scores[4] + augmented_scores[5] + augmented_scores[6]) / 3
print(f"Augmented Average Accuracy: {augmented_avg_accuracy:.4f}")

# - PERFORMANCE COMPARISON AND VISUALIZATION
# Calculate performance improvement
if baseline_avg_accuracy > 0:
    accuracy_improvement = augmented_avg_accuracy - baseline_avg_accuracy
    relative_improvement = (accuracy_improvement / baseline_avg_accuracy) * 100
else:
    accuracy_improvement = augmented_avg_accuracy
    relative_improvement = 100.0

# Plot training histories - accuracy
plt.figure(figsize=(15, 10))

# First digit accuracy
plt.subplot(2, 2, 1)
plt.plot(baseline_history.history['digit1_accuracy'], label='Baseline Training')
plt.plot(baseline_history.history['val_digit1_accuracy'], label='Baseline Validation')
plt.plot(augmented_history.history['digit1_accuracy'], label='Augmented Training')
plt.plot(augmented_history.history['val_digit1_accuracy'], label='Augmented Validation')
plt.title('First Digit Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

# Second digit accuracy
plt.subplot(2, 2, 2)
plt.plot(baseline_history.history['digit2_accuracy'], label='Baseline Training')
plt.plot(baseline_history.history['val_digit2_accuracy'], label='Baseline Validation')
plt.plot(augmented_history.history['digit2_accuracy'], label='Augmented Training')
plt.plot(augmented_history.history['val_digit2_accuracy'], label='Augmented Validation')
plt.title('Second Digit Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

# Third digit accuracy
plt.subplot(2, 2, 3)
plt.plot(baseline_history.history['digit3_accuracy'], label='Baseline Training')
plt.plot(baseline_history.history['val_digit3_accuracy'], label='Baseline Validation')
plt.plot(augmented_history.history['digit3_accuracy'], label='Augmented Training')
plt.plot(augmented_history.history['val_digit3_accuracy'], label='Augmented Validation')
plt.title('Third Digit Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

# Loss
plt.subplot(2, 2, 4)
plt.plot(baseline_history.history['loss'], label='Baseline Training')
plt.plot(baseline_history.history['val_loss'], label='Baseline Validation')
plt.plot(augmented_history.history['loss'], label='Augmented Training')
plt.plot(augmented_history.history['val_loss'], label='Augmented Validation')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
save_figure('classifier_comparison.png')

# Bar chart comparing test accuracies
plt.figure(figsize=(12, 8))

# Per-digit accuracy comparison
plt.subplot(2, 1, 1)
x = np.arange(3)
width = 0.35
plt.bar(x - width/2, [baseline_scores[4], baseline_scores[5], baseline_scores[6]], width, label='Baseline Model')
plt.bar(x + width/2, [augmented_scores[4], augmented_scores[5], augmented_scores[6]], width, label='Augmented Model')
plt.xlabel('Digit Position')
plt.ylabel('Accuracy')
plt.title('Test Accuracy by Digit Position')
plt.xticks(x, ['Digit 1', 'Digit 2', 'Digit 3'])
plt.legend()
plt.grid(True, alpha=0.3)

# Average accuracy comparison
plt.subplot(2, 1, 2)
plt.bar(['Baseline Model', 'Augmented Model'], 
        [baseline_avg_accuracy, augmented_avg_accuracy],
        color=['blue', 'green'])
plt.title('Average Test Accuracy Comparison')
plt.ylabel('Accuracy')
plt.ylim(max(0, min(baseline_avg_accuracy, augmented_avg_accuracy) * 0.9), 
         min(1.0, max(baseline_avg_accuracy, augmented_avg_accuracy) * 1.1))
for i, v in enumerate([baseline_avg_accuracy, augmented_avg_accuracy]):
    plt.text(i, v + 0.01, f"{v:.4f}", ha='center')
plt.text(0.5, plt.ylim()[0] + (plt.ylim()[1] - plt.ylim()[0]) * 0.9, 
         f"Improvement: {accuracy_improvement:.4f} ({relative_improvement:.2f}%)", 
         ha='center', bbox=dict(facecolor='white', alpha=0.5))
plt.grid(True, alpha=0.3)

plt.tight_layout()
save_figure('accuracy_comparison.png')

# Generate predictions for a few test examples with both models
num_examples = 10
example_indices = np.random.choice(len(X_test), num_examples, replace=False)

baseline_preds = classifier_baseline.predict(X_test[example_indices], verbose=0)
augmented_preds = classifier_augmented.predict(X_test[example_indices], verbose=0)

baseline_pred_digits = [np.argmax(baseline_preds[i], axis=1) for i in range(3)]
augmented_pred_digits = [np.argmax(augmented_preds[i], axis=1) for i in range(3)]
true_digits = [np.argmax(y_test[i][example_indices], axis=1) for i in range(3)]

# Visualization of predictions
plt.figure(figsize=(15, 10))
for i in range(num_examples):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_test[example_indices[i], :, :, 0], cmap='gray')
    
    true_label = f"{true_digits[0][i]}{true_digits[1][i]}{true_digits[2][i]}"
    baseline_pred = f"{baseline_pred_digits[0][i]}{baseline_pred_digits[1][i]}{baseline_pred_digits[2][i]}"
    augmented_pred = f"{augmented_pred_digits[0][i]}{augmented_pred_digits[1][i]}{augmented_pred_digits[2][i]}"
    
    baseline_correct = (true_label == baseline_pred)
    augmented_correct = (true_label == augmented_pred)
    
    title = f"True: {true_label}\n"
    title += f"Base: {baseline_pred} ({'✓' if baseline_correct else '✗'})\n"
    title += f"Aug: {augmented_pred} ({'✓' if augmented_correct else '✗'})"
    
    plt.title(title)
    plt.axis('off')
    
plt.suptitle('Comparison of Model Predictions', fontsize=16)
plt.tight_layout()
save_figure('prediction_examples.png')

# Analyze accuracy improvement for each digit
digit1_improvement = augmented_scores[4] - baseline_scores[4]
digit2_improvement = augmented_scores[5] - baseline_scores[5]
digit3_improvement = augmented_scores[6] - baseline_scores[6]

# Visualize accuracy improvement by digit position
plt.figure(figsize=(10, 6))
improvements = [digit1_improvement, digit2_improvement, digit3_improvement]
plt.bar(['Digit 1', 'Digit 2', 'Digit 3'], improvements, color=['skyblue', 'lightgreen', 'salmon'])
plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
plt.title('Accuracy Improvement by Digit Position')
plt.ylabel('Improvement (Augmented - Baseline)')
plt.grid(True, alpha=0.3)
for i, v in enumerate(improvements):
    color = 'green' if v > 0 else 'red'
    plt.text(i, v + 0.01 if v > 0 else v - 0.02, f"{v:.4f}", ha='center', va='center', color=color)
plt.tight_layout()
save_figure('improvement_by_position.png')

# - FINAL RESULTS SUMMARY
print("\n- TASK 5 SUMMARY")

print(f"Classifier Performance:")
print(f"- Baseline model (original data only):")
print(f"  - Average test accuracy: {baseline_avg_accuracy:.4f}")
print(f"  - Digit 1 accuracy: {baseline_scores[4]:.4f}")
print(f"  - Digit 2 accuracy: {baseline_scores[5]:.4f}")
print(f"  - Digit 3 accuracy: {baseline_scores[6]:.4f}")

print(f"- Augmented model (original + synthetic data):")
print(f"  - Average test accuracy: {augmented_avg_accuracy:.4f}")
print(f"  - Digit 1 accuracy: {augmented_scores[4]:.4f}")
print(f"  - Digit 2 accuracy: {augmented_scores[5]:.4f}")
print(f"  - Digit 3 accuracy: {augmented_scores[6]:.4f}")

print(f"- Absolute accuracy improvement: {accuracy_improvement:.4f}")
print(f"- Relative accuracy improvement: {relative_improvement:.2f}%")

print(f"\nAll outputs and visualizations saved to: {output_dir}")