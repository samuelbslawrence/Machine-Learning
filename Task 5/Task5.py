import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
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
from datetime import datetime

# Set up paths
TRAIN_PATH = r"G:\Programing\Machine Learing\triple_mnist\train"
OUTPUT_PATH = r"G:\Programing\Machine Learing\Return"
MODEL_PATH = os.path.join(OUTPUT_PATH, "augmented_Third_digit_best.h5")

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Enable interactive mode for real-time plotting
plt.ion()

# Set style for beautiful plots
plt.style.use('dark_background')
try:
    sns.set_palette("husl")
except:
    pass  # Skip if seaborn styling fails

class AdvancedDigitRecognizer:
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
        """Load images and extract individual digits with enhanced preprocessing"""
        print("=" * 60)
        print("LOADING AND PREPROCESSING DATA")
        print("=" * 60)
        
        # Check if training directory exists
        if not os.path.exists(TRAIN_PATH):
            raise FileNotFoundError(f"Training directory not found: {TRAIN_PATH}")
        
        images = []
        labels = []
        
        # Get all folders in the train directory
        folders = [f for f in os.listdir(TRAIN_PATH) if os.path.isdir(os.path.join(TRAIN_PATH, f))]
        
        if not folders:
            raise ValueError(f"No folders found in training directory: {TRAIN_PATH}")
        
        print(f"Found {len(folders)} folders to process...")
        
        for folder_idx, folder in enumerate(folders):
            folder_path = os.path.join(TRAIN_PATH, folder)
            image_files = glob.glob(os.path.join(folder_path, "*.png"))
            
            # Progress indicator
            print(f"\rProcessing folder {folder_idx+1}/{len(folders)}: {folder} ({len(image_files)} images)", end='')
            
            for img_file in image_files[:60]:  # Increased from 50 to 60 for more data
                # Extract the digits from filename
                filename = os.path.basename(img_file)
                digits_str = filename.split('_')[1].split('.')[0]
                
                # Load image
                img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                
                # Split image into individual digits (assuming 3 digits)
                height, width = img.shape
                digit_width = width // 3
                
                for i, digit_label in enumerate(digits_str):
                    # Extract individual digit with some padding
                    x_start = max(0, i * digit_width - 2)
                    x_end = min(width, (i + 1) * digit_width + 2)
                    digit_img = img[:, x_start:x_end]
                    
                    # Enhanced preprocessing
                    digit_img = self.preprocess_digit(digit_img)
                    
                    images.append(digit_img)
                    labels.append(int(digit_label))
        
        print(f"\n\nPreprocessing complete!")
        
        # Convert to numpy arrays
        self.X = np.array(images)
        self.y = np.array(labels)
        
        # Reshape for CNN (add channel dimension)
        self.X = self.X.reshape(-1, 28, 28, 1)
        
        # One-hot encode labels
        self.y_encoded = self.label_binarizer.fit_transform(self.y)
        
        # Split data with stratification
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y_encoded, test_size=0.2, random_state=42, stratify=self.y
        )
        
        print(f"Dataset Statistics:")
        print(f"  - Total digit images: {len(self.X)}")
        print(f"  - Training samples: {len(self.X_train)}")
        print(f"  - Test samples: {len(self.X_test)}")
        print(f"  - Number of classes: 10 (digits 0-9)")
        print(f"  - Image shape: 28x28 pixels")
        
        # Show digit distribution
        self.show_data_distribution()
        
    def preprocess_digit(self, digit_img):
        """Enhanced preprocessing for individual digits"""
        # Resize to 28x28
        digit_img = cv2.resize(digit_img, (28, 28))
        
        # Apply Gaussian blur to reduce noise
        digit_img = cv2.GaussianBlur(digit_img, (3, 3), 0)
        
        # Apply adaptive thresholding for better digit extraction
        digit_img = cv2.adaptiveThreshold(
            digit_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Invert if necessary (ensure digits are white on black)
        if np.mean(digit_img) > 127:
            digit_img = 255 - digit_img
        
        # Normalize pixel values
        digit_img = digit_img.astype('float32') / 255.0
        
        return digit_img
    
    def show_data_distribution(self):
        """Show the distribution of digits in the dataset"""
        plt.figure(figsize=(10, 5))
        
        # Get digit counts
        train_digits = np.argmax(self.y_train, axis=1)
        test_digits = np.argmax(self.y_test, axis=1)
        
        plt.subplot(1, 2, 1)
        plt.hist(train_digits, bins=10, color='skyblue', edgecolor='white', alpha=0.7)
        plt.title('Training Data Distribution', fontsize=14)
        plt.xlabel('Digit')
        plt.ylabel('Count')
        plt.xticks(range(10))
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.hist(test_digits, bins=10, color='lightcoral', edgecolor='white', alpha=0.7)
        plt.title('Test Data Distribution', fontsize=14)
        plt.xlabel('Digit')
        plt.ylabel('Count')
        plt.xticks(range(10))
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_PATH, 'data_distribution.png'))
        plt.show()
        
    def create_advanced_model(self):
        """Create an improved CNN model with batch normalization and more layers"""
        print("\nBuilding Advanced Neural Network Model...")
        
        self.model = keras.Sequential([
            # Input layer
            layers.Input(shape=(28, 28, 1)),
            
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Conv2D(32, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Conv2D(64, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third Convolutional Block
            layers.Conv2D(128, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Dense layers
            layers.Flatten(),
            layers.Dense(256),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.5),
            layers.Dense(128),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.5),
            layers.Dense(10, activation='softmax')
        ])
        
        # Build the model explicitly
        self.model.build(input_shape=(None, 28, 28, 1))
        
        # Use Adam optimizer with learning rate scheduling
        initial_learning_rate = 0.001
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate,
            decay_steps=1000,
            decay_rate=0.9,
            staircase=True
        )
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
            loss='categorical_crossentropy',
            metrics=['accuracy', 
                    keras.metrics.Precision(name='precision'), 
                    keras.metrics.Recall(name='recall')]
        )
        
        # Print model summary
        print("\nModel Architecture:")
        self.model.summary()
        
    def create_data_augmentation(self):
        """Create data augmentation for better generalization"""
        self.datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            fill_mode='nearest'
        )
        
    def visualize_enhanced_training(self):
        """Create enhanced visualization of the training process"""
        print("\nStarting Enhanced Visual Training Process...")
        
        # Create data augmentation
        self.create_data_augmentation()
        
        # Set up the figure with more sophisticated layout
        fig = plt.figure(figsize=(20, 12))
        fig.patch.set_facecolor('#0a0a0a')
        fig.suptitle('Advanced Neural Network Training Visualization', 
                     fontsize=24, fontweight='bold', color='white')
        
        # Create subplots with better organization
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # 1. Sample augmented digits
        ax_samples = fig.add_subplot(gs[0, :2])
        ax_samples.set_title('Data Augmentation Examples', fontsize=16, color='white')
        ax_samples.axis('off')
        
        # Show original and augmented samples
        sample_idx = np.random.randint(0, len(self.X_train))
        original = self.X_train[sample_idx].reshape(28, 28)
        
        # Display original
        ax_orig = ax_samples.inset_axes([0.05, 0.5, 0.15, 0.4])
        ax_orig.imshow(original, cmap='plasma')
        ax_orig.set_title('Original', fontsize=10, color='white')
        ax_orig.axis('off')
        
        # Display augmented versions
        augmented_positions = [(0.25, 0.5), (0.45, 0.5), (0.65, 0.5), (0.85, 0.5)]
        for i, (x, y) in enumerate(augmented_positions):
            aug_ax = ax_samples.inset_axes([x, y, 0.15, 0.4])
            augmented = self.datagen.random_transform(self.X_train[sample_idx])
            aug_ax.imshow(augmented.reshape(28, 28), cmap='plasma')
            aug_ax.set_title(f'Aug {i+1}', fontsize=10, color='white')
            aug_ax.axis('off')
        
        # 2. Loss plot with style
        ax_loss = fig.add_subplot(gs[0, 2])
        ax_loss.set_facecolor('#1a1a1a')
        ax_loss.set_title('Training Progress - Loss', fontsize=16, color='white')
        ax_loss.set_xlabel('Epoch', color='white')
        ax_loss.set_ylabel('Loss', color='white')
        ax_loss.grid(True, alpha=0.2, color='white')
        
        # 3. Accuracy plot with style
        ax_acc = fig.add_subplot(gs[0, 3])
        ax_acc.set_facecolor('#1a1a1a')
        ax_acc.set_title('Training Progress - Accuracy', fontsize=16, color='white')
        ax_acc.set_xlabel('Epoch', color='white')
        ax_acc.set_ylabel('Accuracy', color='white')
        ax_acc.grid(True, alpha=0.2, color='white')
        
        # 4. Learning rate plot
        ax_lr = fig.add_subplot(gs[1, 0])
        ax_lr.set_facecolor('#1a1a1a')
        ax_lr.set_title('Learning Rate Schedule', fontsize=16, color='white')
        ax_lr.set_xlabel('Epoch', color='white')
        ax_lr.set_ylabel('Learning Rate', color='white')
        ax_lr.grid(True, alpha=0.2, color='white')
        
        # 5. Confusion matrix
        ax_conf = fig.add_subplot(gs[1, 1:3])
        ax_conf.set_title('Live Confusion Matrix', fontsize=16, color='white')
        
        # 6. Live predictions
        ax_pred = fig.add_subplot(gs[1, 3])
        ax_pred.set_title('Live Predictions', fontsize=16, color='white')
        ax_pred.axis('off')
        
        # 7. Model metrics
        ax_metrics = fig.add_subplot(gs[2, :2])
        ax_metrics.set_facecolor('#1a1a1a')
        ax_metrics.set_title('Performance Metrics', fontsize=16, color='white')
        ax_metrics.axis('off')
        
        # 8. Training progress samples
        ax_features = fig.add_subplot(gs[2, 2:])
        ax_features.set_title('Training Progress Samples', fontsize=16, color='white')
        ax_features.axis('off')
        
        plt.tight_layout()
        
        # Initialize plot lines
        loss_line, = ax_loss.plot([], [], 'b-', linewidth=2, label='Training Loss')
        val_loss_line, = ax_loss.plot([], [], 'r-', linewidth=2, label='Validation Loss')
        ax_loss.legend(loc='upper right')
        
        acc_line, = ax_acc.plot([], [], 'g-', linewidth=2, label='Training Accuracy')
        val_acc_line, = ax_acc.plot([], [], 'orange', linewidth=2, label='Validation Accuracy')
        ax_acc.legend(loc='lower right')
        
        # Custom callback for real-time visualization
        class EnhancedVisualCallback(keras.callbacks.Callback):
            def __init__(self, axes, lines):
                self.axes = axes
                self.lines = lines
                self.losses = []
                self.val_losses = []
                self.accs = []
                self.val_accs = []
                self.lrs = []
                self.best_val_acc = 0
                
            def on_epoch_end(self, epoch, logs=None):
                # Update metrics
                self.losses.append(logs['loss'])
                self.val_losses.append(logs['val_loss'])
                self.accs.append(logs['accuracy'])
                self.val_accs.append(logs['val_accuracy'])
                self.lrs.append(float(self.model.optimizer.learning_rate))
                
                # Update loss plot
                self.lines['loss'].set_data(range(len(self.losses)), self.losses)
                self.lines['val_loss'].set_data(range(len(self.val_losses)), self.val_losses)
                self.axes['loss'].set_xlim(0, max(1, len(self.losses)))
                self.axes['loss'].set_ylim(0, max(max(self.losses), max(self.val_losses)) * 1.1)
                
                # Update accuracy plot
                self.lines['acc'].set_data(range(len(self.accs)), self.accs)
                self.lines['val_acc'].set_data(range(len(self.val_accs)), self.val_accs)
                self.axes['acc'].set_xlim(0, max(1, len(self.accs)))
                self.axes['acc'].set_ylim(min(0.5, min(self.accs) * 0.95), 1.02)
                
                # Update learning rate plot
                self.axes['lr'].clear()
                self.axes['lr'].plot(self.lrs, 'y-', linewidth=2)
                self.axes['lr'].set_title('Learning Rate Schedule', fontsize=16, color='white')
                self.axes['lr'].set_xlabel('Epoch', color='white')
                self.axes['lr'].set_ylabel('Learning Rate', color='white')
                self.axes['lr'].set_facecolor('#1a1a1a')
                self.axes['lr'].grid(True, alpha=0.2, color='white')
                
                # Update confusion matrix every 3 epochs
                if epoch % 3 == 0:
                    self.update_confusion_matrix()
                
                # Show live predictions
                self.show_live_predictions()
                
                # Update metrics display
                self.update_metrics_display(epoch, logs)
                
                # Show training progress samples every 3 epochs
                if epoch % 3 == 0:
                    self.visualize_training_progress(epoch)
                
                # Check for best model
                if logs['val_accuracy'] > self.best_val_acc:
                    self.best_val_acc = logs['val_accuracy']
                    recognizer.best_accuracy = self.best_val_acc
                
                plt.draw()
                plt.pause(0.01)
                
            def update_confusion_matrix(self):
                self.axes['conf'].clear()
                
                # Get predictions on a subset of validation data
                subset_size = min(1000, len(recognizer.X_test))
                indices = np.random.choice(len(recognizer.X_test), subset_size, replace=False)
                predictions = self.model.predict(recognizer.X_test[indices], verbose=0)
                predicted_labels = np.argmax(predictions, axis=1)
                true_labels = np.argmax(recognizer.y_test[indices], axis=1)
                
                # Create confusion matrix
                cm = confusion_matrix(true_labels, predicted_labels)
                
                # Plot with seaborn for better visualization
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                           square=True, cbar_kws={'label': 'Count'},
                           ax=self.axes['conf'])
                self.axes['conf'].set_xlabel('Predicted', color='white')
                self.axes['conf'].set_ylabel('True', color='white')
                self.axes['conf'].set_title('Live Confusion Matrix', fontsize=16, color='white')
                
            def show_live_predictions(self):
                self.axes['pred'].clear()
                self.axes['pred'].set_title('Live Predictions', fontsize=16, color='white')
                self.axes['pred'].axis('off')
                
                # Select 6 random test samples
                indices = np.random.choice(len(recognizer.X_test), 6)
                
                for i, idx in enumerate(indices):
                    row = i // 2
                    col = i % 2
                    
                    # Create inset
                    left = 0.1 + col * 0.45
                    bottom = 0.65 - row * 0.3
                    width = 0.35
                    height = 0.25
                    
                    inset_ax = self.axes['pred'].inset_axes([left, bottom, width, height])
                    
                    img = recognizer.X_test[idx].reshape(28, 28)
                    inset_ax.imshow(img, cmap='plasma')
                    
                    # Make prediction
                    pred = self.model.predict(recognizer.X_test[idx:idx+1], verbose=0)
                    pred_label = np.argmax(pred)
                    true_label = np.argmax(recognizer.y_test[idx])
                    confidence = pred[0][pred_label]
                    
                    color = 'lime' if pred_label == true_label else 'red'
                    inset_ax.set_title(f'P:{pred_label} ({confidence:.2f})\nT:{true_label}', 
                                      fontsize=9, color=color, pad=2)
                    inset_ax.axis('off')
                    
            def update_metrics_display(self, epoch, logs):
                self.axes['metrics'].clear()
                self.axes['metrics'].axis('off')
                self.axes['metrics'].set_facecolor('#1a1a1a')
                
                # Display current metrics
                metrics_text = f"""
                Epoch: {epoch + 1}/30
                
                Training Metrics:
                  Loss: {logs['loss']:.4f}
                  Accuracy: {logs['accuracy']:.4f}
                  Precision: {logs.get('precision', 0):.4f}
                  Recall: {logs.get('recall', 0):.4f}
                
                Validation Metrics:
                  Loss: {logs['val_loss']:.4f}
                  Accuracy: {logs['val_accuracy']:.4f}
                  Precision: {logs.get('val_precision', 0):.4f}
                  Recall: {logs.get('val_recall', 0):.4f}
                
                Best Validation Accuracy: {self.best_val_acc:.4f}
                Current Learning Rate: {float(self.model.optimizer.learning_rate):.6f}
                """
                
                self.axes['metrics'].text(0.1, 0.9, metrics_text, 
                                        transform=self.axes['metrics'].transAxes,
                                        fontsize=12, color='white', 
                                        verticalalignment='top',
                                        fontfamily='monospace')
                
            def visualize_training_progress(self, epoch):
                try:
                    self.axes['features'].clear()
                    self.axes['features'].set_title(f'Training Progress - Epoch {epoch+1}', fontsize=16, color='white')
                    self.axes['features'].axis('off')
                    
                    # Show training progress samples
                    sample_indices = np.random.choice(len(recognizer.X_test), 6)
                    
                    for i, idx in enumerate(sample_indices):
                        row = i // 3
                        col = i % 3
                        
                        ax = self.axes['features'].inset_axes(
                            [0.1 + col*0.3, 0.5 - row*0.4, 0.25, 0.35]
                        )
                        
                        img = recognizer.X_test[idx].reshape(28, 28)
                        ax.imshow(img, cmap='viridis')
                        
                        # Make prediction
                        pred = self.model.predict(recognizer.X_test[idx:idx+1], verbose=0)
                        pred_label = np.argmax(pred)
                        true_label = np.argmax(recognizer.y_test[idx])
                        confidence = pred[0][pred_label]
                        
                        color = 'lime' if pred_label == true_label else 'red'
                        ax.set_title(f'{pred_label} ({confidence:.2f})', fontsize=10, color=color)
                        ax.axis('off')
                        
                    # Add accuracy info
                    acc_text = f'Current Val Acc: {self.val_accs[-1]:.3f}' if self.val_accs else 'Starting...'
                    self.axes['features'].text(0.5, 0.05, acc_text, 
                                             transform=self.axes['features'].transAxes,
                                             ha='center', va='bottom', 
                                             fontsize=12, color='yellow')
                except Exception as e:
                    # Fallback if visualization fails
                    self.axes['features'].text(0.5, 0.5, f'Epoch {epoch+1}', 
                                             transform=self.axes['features'].transAxes,
                                             ha='center', va='center', 
                                             fontsize=20, color='gray')
        
        # Create callback instance
        axes = {
            'loss': ax_loss, 'acc': ax_acc, 'lr': ax_lr, 
            'conf': ax_conf, 'pred': ax_pred, 'metrics': ax_metrics,
            'features': ax_features
        }
        lines = {
            'loss': loss_line, 'val_loss': val_loss_line,
            'acc': acc_line, 'val_acc': val_acc_line
        }
        
        visual_callback = EnhancedVisualCallback(axes, lines)
        
        # Early stopping callback
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        
        # Model checkpoint
        checkpoint = keras.callbacks.ModelCheckpoint(
            MODEL_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
        
        # Train the model with data augmentation
        print("\nTraining started! Watch the beautiful enhanced learning process...")
        print("This may take a few minutes. The visualizations will update in real-time.\n")
        
        self.history = self.model.fit(
            self.datagen.flow(self.X_train, self.y_train, batch_size=32),
            validation_data=(self.X_test, self.y_test),
            epochs=30,
            callbacks=[visual_callback, early_stopping, checkpoint],
            verbose=1
        )
        
        plt.ioff()
        plt.savefig(os.path.join(OUTPUT_PATH, 'training_visualization.png'), dpi=150, bbox_inches='tight')
        plt.show()
        
    def test_and_analyze_model(self):
        """Comprehensive model testing and analysis"""
        print("\n" + "="*60)
        print("COMPREHENSIVE MODEL ANALYSIS")
        print("="*60)
        
        # Load best model
        print("\nLoading best model...")
        self.model = keras.models.load_model(MODEL_PATH)
        
        # Evaluate on test set
        print("\nEvaluating on test set...")
        test_results = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        
        print(f"\nTest Results:")
        print(f"  - Loss: {test_results[0]:.4f}")
        print(f"  - Accuracy: {test_results[1]:.4f}")
        if len(test_results) > 2:
            print(f"  - Precision: {test_results[2]:.4f}")
        if len(test_results) > 3:
            print(f"  - Recall: {test_results[3]:.4f}")
        
        # Get predictions for detailed analysis
        predictions = self.model.predict(self.X_test, verbose=0)
        predicted_labels = np.argmax(predictions, axis=1)
        true_labels = np.argmax(self.y_test, axis=1)
        
        # Classification report
        print("\nDetailed Classification Report:")
        print(classification_report(true_labels, predicted_labels, 
                                  target_names=[str(i) for i in range(10)]))
        
        # Create comprehensive visualization
        fig = plt.figure(figsize=(20, 12))
        fig.patch.set_facecolor('#0a0a0a')
        fig.suptitle('Model Performance Analysis', fontsize=24, fontweight='bold', color='white')
        
        # 1. Final confusion matrix
        ax1 = plt.subplot(2, 3, 1)
        cm = confusion_matrix(true_labels, predicted_labels)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', square=True, ax=ax1)
        ax1.set_title('Final Confusion Matrix', fontsize=16, color='white')
        ax1.set_xlabel('Predicted', color='white')
        ax1.set_ylabel('True', color='white')
        
        # 2. Per-class accuracy
        ax2 = plt.subplot(2, 3, 2)
        ax2.set_facecolor('#1a1a1a')
        class_accuracy = []
        for i in range(10):
            mask = true_labels == i
            if np.sum(mask) > 0:
                acc = np.mean(predicted_labels[mask] == i)
                class_accuracy.append(acc)
            else:
                class_accuracy.append(0)
        
        bars = ax2.bar(range(10), class_accuracy, color='skyblue', edgecolor='white')
        ax2.set_title('Per-Class Accuracy', fontsize=16, color='white')
        ax2.set_xlabel('Digit', color='white')
        ax2.set_ylabel('Accuracy', color='white')
        ax2.set_ylim(0, 1.1)
        ax2.set_xticks(range(10))
        
        # Add value labels on bars
        for bar, acc in zip(bars, class_accuracy):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom', color='white')
        
        # 3. Confidence distribution
        ax3 = plt.subplot(2, 3, 3)
        ax3.set_facecolor('#1a1a1a')
        correct_confidences = []
        incorrect_confidences = []
        
        for i in range(len(predictions)):
            confidence = np.max(predictions[i])
            if predicted_labels[i] == true_labels[i]:
                correct_confidences.append(confidence)
            else:
                incorrect_confidences.append(confidence)
        
        ax3.hist(correct_confidences, bins=20, alpha=0.7, label='Correct', color='green', edgecolor='white')
        ax3.hist(incorrect_confidences, bins=20, alpha=0.7, label='Incorrect', color='red', edgecolor='white')
        ax3.set_title('Prediction Confidence Distribution', fontsize=16, color='white')
        ax3.set_xlabel('Confidence', color='white')
        ax3.set_ylabel('Count', color='white')
        ax3.legend()
        ax3.grid(True, alpha=0.2)
        
        # 4. Most confused pairs
        ax4 = plt.subplot(2, 3, 4)
        ax4.set_facecolor('#1a1a1a')
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
            labels = [f'{p[0]}→{p[1]}' for p in top_confused]
            counts = [p[2] for p in top_confused]
            
            ax4.barh(range(len(labels)), counts, color='coral', edgecolor='white')
            ax4.set_yticks(range(len(labels)))
            ax4.set_yticklabels(labels, color='white')
            ax4.set_xlabel('Misclassification Count', color='white')
            ax4.set_title('Top 10 Most Confused Digit Pairs', fontsize=16, color='white')
            ax4.grid(True, alpha=0.2, axis='x')
        
        # 5. Sample predictions grid
        ax5 = plt.subplot(2, 3, 5)
        ax5.set_title('Sample Predictions', fontsize=16, color='white')
        ax5.axis('off')
        
        # Show 16 random predictions
        sample_indices = np.random.choice(len(self.X_test), 16, replace=False)
        for i, idx in enumerate(sample_indices):
            row = i // 4
            col = i % 4
            
            ax_sample = ax5.inset_axes([col*0.25, 0.75-row*0.25, 0.2, 0.2])
            ax_sample.imshow(self.X_test[idx].reshape(28, 28), cmap='gray')
            
            pred_label = predicted_labels[idx]
            true_label = true_labels[idx]
            confidence = predictions[idx][pred_label]
            
            color = 'lime' if pred_label == true_label else 'red'
            ax_sample.set_title(f'{pred_label}({confidence:.2f})', 
                               fontsize=8, color=color, pad=1)
            ax_sample.axis('off')
        
        # 6. Hardest examples
        ax6 = plt.subplot(2, 3, 6)
        ax6.set_title('Hardest Examples (Lowest Confidence Errors)', fontsize=16, color='white')
        ax6.axis('off')
        
        # Find incorrectly classified examples with lowest confidence
        incorrect_indices = np.where(predicted_labels != true_labels)[0]
        if len(incorrect_indices) > 0:
            confidences = [np.max(predictions[i]) for i in incorrect_indices]
            sorted_indices = incorrect_indices[np.argsort(confidences)][:8]
            
            for i, idx in enumerate(sorted_indices):
                row = i // 4
                col = i % 4
                
                ax_hard = ax6.inset_axes([col*0.25, 0.75-row*0.35, 0.2, 0.3])
                ax_hard.imshow(self.X_test[idx].reshape(28, 28), cmap='gray')
                
                pred_label = predicted_labels[idx]
                true_label = true_labels[idx]
                confidence = predictions[idx][pred_label]
                
                ax_hard.set_title(f'P:{pred_label}({confidence:.2f})\nT:{true_label}', 
                                 fontsize=8, color='red', pad=1)
                ax_hard.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_PATH, 'model_analysis.png'), dpi=150, bbox_inches='tight')
        plt.show()
        
        # Save detailed predictions
        print(f"\nAll visualizations saved to: {OUTPUT_PATH}")
        print(f"Best model accuracy achieved: {self.best_accuracy:.4f}")
        
    def create_interactive_drawing_app(self):
        """Create an interactive drawing application to test the model"""
        print("\nCreating Interactive Drawing Application...")
        
        # Create a new figure for drawing
        fig, (ax_draw, ax_pred) = plt.subplots(1, 2, figsize=(12, 6))
        fig.patch.set_facecolor('#f0f0f0')
        fig.suptitle('Interactive Digit Recognition - Draw a Digit!', fontsize=18, fontweight='bold')
        
        # Drawing canvas
        ax_draw.set_title('Draw Here', fontsize=14)
        ax_draw.set_xlim(0, 1)
        ax_draw.set_ylim(0, 1)
        ax_draw.set_aspect('equal')
        ax_draw.grid(True, alpha=0.3)
        
        # Prediction display
        ax_pred.set_title('Prediction', fontsize=14)
        ax_pred.axis('off')
        
        # Drawing state
        drawing_state = {
            'drawing': False,
            'lines': [],
            'current_line': [],
            'canvas': np.zeros((28, 28))
        }
        
        def on_press(event):
            if event.inaxes == ax_draw:
                drawing_state['drawing'] = True
                drawing_state['current_line'] = [(event.xdata, event.ydata)]
                
        def on_motion(event):
            if drawing_state['drawing'] and event.inaxes == ax_draw:
                drawing_state['current_line'].append((event.xdata, event.ydata))
                
                # Redraw
                ax_draw.clear()
                ax_draw.set_xlim(0, 1)
                ax_draw.set_ylim(0, 1)
                ax_draw.set_aspect('equal')
                ax_draw.grid(True, alpha=0.3)
                ax_draw.set_title('Draw Here', fontsize=14)
                
                # Draw all lines
                for line in drawing_state['lines']:
                    if len(line) > 1:
                        x_coords, y_coords = zip(*line)
                        ax_draw.plot(x_coords, y_coords, 'k-', linewidth=15)
                
                # Draw current line
                if len(drawing_state['current_line']) > 1:
                    x_coords, y_coords = zip(*drawing_state['current_line'])
                    ax_draw.plot(x_coords, y_coords, 'k-', linewidth=15)
                
                plt.draw()
                
        def on_release(event):
            if drawing_state['drawing']:
                drawing_state['drawing'] = False
                if len(drawing_state['current_line']) > 1:
                    drawing_state['lines'].append(drawing_state['current_line'])
                    predict_digit()
                    
        def clear_canvas(event):
            drawing_state['lines'] = []
            drawing_state['current_line'] = []
            drawing_state['canvas'] = np.zeros((28, 28))
            
            ax_draw.clear()
            ax_draw.set_xlim(0, 1)
            ax_draw.set_ylim(0, 1)
            ax_draw.set_aspect('equal')
            ax_draw.grid(True, alpha=0.3)
            ax_draw.set_title('Draw Here', fontsize=14)
            
            ax_pred.clear()
            ax_pred.set_title('Prediction', fontsize=14)
            ax_pred.axis('off')
            
            plt.draw()
            
        def predict_digit():
            # Convert drawing to 28x28 image
            canvas = np.zeros((280, 280), dtype=np.uint8)
            
            for line in drawing_state['lines']:
                if len(line) > 1:
                    for i in range(len(line) - 1):
                        pt1 = (int(line[i][0] * 280), int((1 - line[i][1]) * 280))
                        pt2 = (int(line[i+1][0] * 280), int((1 - line[i+1][1]) * 280))
                        cv2.line(canvas, pt1, pt2, 255, thickness=20)
            
            # Resize to 28x28
            digit_img = cv2.resize(canvas, (28, 28))
            
            # Preprocess
            if np.max(digit_img) > 0:
                # Apply same preprocessing as training data
                digit_img = recognizer.preprocess_digit(digit_img)
                digit_img = digit_img.reshape(1, 28, 28, 1)
                
                # Make prediction
                prediction = recognizer.model.predict(digit_img, verbose=0)
                predicted_digit = np.argmax(prediction)
                confidence = prediction[0][predicted_digit]
                
                # Display results
                ax_pred.clear()
                ax_pred.imshow(digit_img.reshape(28, 28), cmap='gray')
                ax_pred.set_title(f'Prediction: {predicted_digit}\nConfidence: {confidence:.2%}', 
                                 fontsize=16, fontweight='bold')
                ax_pred.axis('off')
                
                # Show probability distribution
                ax_prob = ax_pred.inset_axes([0.6, 0.1, 0.35, 0.8])
                ax_prob.barh(range(10), prediction[0], color='skyblue')
                ax_prob.set_yticks(range(10))
                ax_prob.set_yticklabels([str(i) for i in range(10)])
                ax_prob.set_xlabel('Probability')
                ax_prob.set_xlim(0, 1)
                ax_prob.grid(True, alpha=0.3, axis='x')
                
                # Highlight predicted digit
                ax_prob.barh(predicted_digit, prediction[0][predicted_digit], color='green')
                
                plt.draw()
        
        # Add clear button
        ax_button = plt.axes([0.7, 0.02, 0.1, 0.04])
        btn_clear = Button(ax_button, 'Clear', color='lightgray', hovercolor='gray')
        btn_clear.on_clicked(clear_canvas)
        
        # Connect event handlers
        fig.canvas.mpl_connect('button_press_event', on_press)
        fig.canvas.mpl_connect('motion_notify_event', on_motion)
        fig.canvas.mpl_connect('button_release_event', on_release)
        
        plt.figtext(0.5, 0.02, 'Click and drag to draw a digit. Release to see prediction.', 
                   ha='center', fontsize=12, style='italic')
        
        plt.tight_layout()
        plt.show()

def main():
    """Main function to run the entire enhanced pipeline"""
    print("="*60)
    print("ADVANCED HANDWRITTEN DIGIT RECOGNITION SYSTEM")
    print("="*60)
    print("Version 2.0 - Enhanced with Advanced Features")
    print("="*60)
    
    # Create recognizer instance
    global recognizer
    recognizer = AdvancedDigitRecognizer()
    
    try:
        # Load and preprocess data
        recognizer.load_and_preprocess_data()
        
        # Create advanced model
        recognizer.create_advanced_model()
        
        # Train with enhanced visualization
        recognizer.visualize_enhanced_training()
        
        # Comprehensive testing and analysis
        recognizer.test_and_analyze_model()
        
        # Create interactive drawing application
        recognizer.create_interactive_drawing_app()
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE - SYSTEM READY!")
        print("="*60)
        print(f"All outputs saved to: {OUTPUT_PATH}")
        print(f"Model saved as: {MODEL_PATH}")
        print(f"Best accuracy achieved: {recognizer.best_accuracy:.4f}")
        print("\nYou can now use the interactive drawing app to test your model!")
        print("="*60)
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        print("\nPlease check that:")
        print(f"1. Training directory exists: {TRAIN_PATH}")
        print(f"2. Image files are in the correct format")
        print(f"3. Output directory is writable: {OUTPUT_PATH}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()