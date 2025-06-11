import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import os
from datetime import datetime
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

class ModelAnalyzer:
    def __init__(self, model_path=None):
        self.models = {}
        self.model_path = model_path
        if model_path:
            self.load_model('default', model_path)
    
    def load_model(self, name, path):
        """Load a model for analysis"""
        try:
            model = tf.keras.models.load_model(path)
            self.models[name] = {
                'model': model,
                'path': path,
                'name': name,
                'size': os.path.getsize(path) / (1024 * 1024)  # Size in MB
            }
            print(f"Loaded model '{name}' from {path}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def analyze_architecture(self, model_name='default'):
        """Analyze and visualize model architecture"""
        if model_name not in self.models:
            print(f"Model '{model_name}' not found")
            return
        
        model = self.models[model_name]['model']
        
        print(f"\n{'='*60}")
        print(f"MODEL ARCHITECTURE ANALYSIS: {model_name}")
        print(f"{'='*60}")
        
        # Basic info
        print(f"\nModel: {self.models[model_name]['path']}")
        print(f"File size: {self.models[model_name]['size']:.2f} MB")
        
        # Layer analysis
        total_params = model.count_params()
        trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
        non_trainable_params = total_params - trainable_params
        
        print(f"\nTotal parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Non-trainable parameters: {non_trainable_params:,}")
        
        # Layer-by-layer breakdown
        print("\nLayer-by-layer breakdown:")
        print("-" * 80)
        print(f"{'Layer (type)':<30} {'Output Shape':<20} {'Param #':<15} {'Trainable':<10}")
        print("-" * 80)
        
        for layer in model.layers:
            trainable = "Yes" if layer.trainable else "No"
            params = layer.count_params()
            output_shape = str(layer.output_shape) if hasattr(layer, 'output_shape') else 'Multiple'
            print(f"{layer.name:<30} {output_shape:<20} {params:<15,} {trainable:<10}")
        
        # Memory usage estimation
        memory_mb = (total_params * 4) / (1024 * 1024)  # 4 bytes per float32
        print(f"\nEstimated memory usage: {memory_mb:.2f} MB")
        
        # Complexity metrics
        conv_layers = sum(1 for layer in model.layers if 'conv' in layer.__class__.__name__.lower())
        dense_layers = sum(1 for layer in model.layers if 'dense' in layer.__class__.__name__.lower())
        
        print(f"\nModel complexity:")
        print(f"- Convolutional layers: {conv_layers}")
        print(f"- Dense layers: {dense_layers}")
        print(f"- Total layers: {len(model.layers)}")
        
        return model
    
    def visualize_filters(self, model_name='default', layer_name=None, max_filters=64):
        """Visualize convolutional filters"""
        if model_name not in self.models:
            print(f"Model '{model_name}' not found")
            return
        
        model = self.models[model_name]['model']
        
        # Find conv layers
        conv_layers = [layer for layer in model.layers 
                      if 'conv' in layer.__class__.__name__.lower()]
        
        if not conv_layers:
            print("No convolutional layers found")
            return
        
        # Select layer
        if layer_name:
            selected_layer = next((l for l in conv_layers if l.name == layer_name), None)
            if not selected_layer:
                print(f"Layer '{layer_name}' not found")
                return
        else:
            selected_layer = conv_layers[0]  # First conv layer
        
        weights = selected_layer.get_weights()[0]
        print(f"\nVisualizing filters from layer: {selected_layer.name}")
        print(f"Filter shape: {weights.shape}")
        
        # Normalize filters
        f_min, f_max = weights.min(), weights.max()
        filters = (weights - f_min) / (f_max - f_min)
        
        # Plot filters
        n_filters = min(filters.shape[3], max_filters)
        n_cols = 8
        n_rows = (n_filters + n_cols - 1) // n_cols
        
        fig = plt.figure(figsize=(n_cols * 2, n_rows * 2))
        fig.suptitle(f'Convolutional Filters - {selected_layer.name}', fontsize=16)
        
        for i in range(n_filters):
            ax = plt.subplot(n_rows, n_cols, i + 1)
            # Get the filter
            if len(filters.shape) == 4:
                if filters.shape[2] == 1:
                    # Grayscale
                    filter_img = filters[:, :, 0, i]
                else:
                    # Take first 3 channels for RGB
                    filter_img = filters[:, :, :3, i]
            else:
                filter_img = filters[:, :, i]
            
            plt.imshow(filter_img, cmap='viridis' if len(filter_img.shape) == 2 else None)
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def analyze_weights(self, model_name='default'):
        """Analyze weight distributions"""
        if model_name not in self.models:
            print(f"Model '{model_name}' not found")
            return
        
        model = self.models[model_name]['model']
        
        # Collect all weights
        all_weights = []
        layer_weights = {}
        
        for layer in model.layers:
            weights = layer.get_weights()
            if weights:
                layer_name = layer.name
                layer_weights[layer_name] = []
                for w in weights:
                    flat_weights = w.flatten()
                    all_weights.extend(flat_weights)
                    layer_weights[layer_name].extend(flat_weights)
        
        # Plot overall weight distribution
        fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(3, 2, figure=fig)
        
        # Overall distribution
        ax1 = fig.add_subplot(gs[0, :])
        ax1.hist(all_weights, bins=100, alpha=0.7, color='blue', edgecolor='black')
        ax1.set_title('Overall Weight Distribution', fontsize=14)
        ax1.set_xlabel('Weight Value')
        ax1.set_ylabel('Frequency')
        ax1.grid(True, alpha=0.3)
        
        # Statistics
        stats_text = f"Mean: {np.mean(all_weights):.4f}\n"
        stats_text += f"Std: {np.std(all_weights):.4f}\n"
        stats_text += f"Min: {np.min(all_weights):.4f}\n"
        stats_text += f"Max: {np.max(all_weights):.4f}"
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Layer-wise box plots
        ax2 = fig.add_subplot(gs[1, :])
        layer_names = list(layer_weights.keys())[:10]  # First 10 layers
        layer_data = [layer_weights[name] for name in layer_names]
        
        bp = ax2.boxplot(layer_data, labels=layer_names, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        ax2.set_title('Weight Distribution by Layer', fontsize=14)
        ax2.set_xlabel('Layer')
        ax2.set_ylabel('Weight Value')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Weight magnitude heatmap
        ax3 = fig.add_subplot(gs[2, 0])
        # Get weights from first few layers
        weight_matrix = []
        for i, layer in enumerate(model.layers[:20]):
            weights = layer.get_weights()
            if weights and len(weights[0].shape) >= 2:
                # Take mean absolute weight for each layer
                mean_weights = np.mean(np.abs(weights[0]), axis=tuple(range(len(weights[0].shape)-1)))
                weight_matrix.append(mean_weights[:50])  # First 50 neurons/filters
        
        if weight_matrix:
            # Pad arrays to same length
            max_len = max(len(row) for row in weight_matrix)
            padded_matrix = [np.pad(row, (0, max_len - len(row)), constant_values=0) for row in weight_matrix]
            
            im = ax3.imshow(padded_matrix, cmap='hot', aspect='auto')
            ax3.set_title('Weight Magnitudes Heatmap', fontsize=14)
            ax3.set_xlabel('Neuron/Filter Index')
            ax3.set_ylabel('Layer Index')
            plt.colorbar(im, ax=ax3)
        
        # Sparsity analysis
        ax4 = fig.add_subplot(gs[2, 1])
        sparsity_threshold = 0.01
        layer_sparsity = []
        layer_names_sparse = []
        
        for name, weights in list(layer_weights.items())[:15]:
            if weights:
                sparsity = np.sum(np.abs(weights) < sparsity_threshold) / len(weights)
                layer_sparsity.append(sparsity * 100)
                layer_names_sparse.append(name)
        
        ax4.bar(range(len(layer_sparsity)), layer_sparsity, color='green', alpha=0.7)
        ax4.set_title(f'Weight Sparsity by Layer (threshold={sparsity_threshold})', fontsize=14)
        ax4.set_xlabel('Layer')
        ax4.set_ylabel('Sparsity (%)')
        ax4.set_xticks(range(len(layer_names_sparse)))
        ax4.set_xticklabels(layer_names_sparse, rotation=45, ha='right')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def compare_models(self, model_names=None):
        """Compare multiple models"""
        if not model_names:
            model_names = list(self.models.keys())
        
        if len(model_names) < 2:
            print("Need at least 2 models to compare")
            return
        
        comparison_data = []
        
        for name in model_names:
            if name not in self.models:
                continue
            
            model = self.models[name]['model']
            
            # Collect metrics
            total_params = model.count_params()
            conv_layers = sum(1 for l in model.layers if 'conv' in l.__class__.__name__.lower())
            dense_layers = sum(1 for l in model.layers if 'dense' in l.__class__.__name__.lower())
            
            # Get weights statistics
            all_weights = []
            for layer in model.layers:
                weights = layer.get_weights()
                if weights:
                    all_weights.extend(weights[0].flatten())
            
            comparison_data.append({
                'Model': name,
                'Total Parameters': total_params,
                'Conv Layers': conv_layers,
                'Dense Layers': dense_layers,
                'Total Layers': len(model.layers),
                'File Size (MB)': self.models[name]['size'],
                'Mean Weight': np.mean(all_weights) if all_weights else 0,
                'Std Weight': np.std(all_weights) if all_weights else 0
            })
        
        # Create comparison DataFrame
        df = pd.DataFrame(comparison_data)
        
        # Visualize comparison
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Model Comparison', fontsize=16)
        
        # Parameters comparison
        axes[0, 0].bar(df['Model'], df['Total Parameters'])
        axes[0, 0].set_title('Total Parameters')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Layer comparison
        x = np.arange(len(df))
        width = 0.35
        axes[0, 1].bar(x - width/2, df['Conv Layers'], width, label='Conv')
        axes[0, 1].bar(x + width/2, df['Dense Layers'], width, label='Dense')
        axes[0, 1].set_title('Layer Types')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(df['Model'], rotation=45)
        axes[0, 1].legend()
        
        # File size
        axes[0, 2].bar(df['Model'], df['File Size (MB)'])
        axes[0, 2].set_title('File Size (MB)')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # Weight statistics
        axes[1, 0].bar(df['Model'], df['Mean Weight'])
        axes[1, 0].set_title('Mean Weight Value')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        axes[1, 1].bar(df['Model'], df['Std Weight'])
        axes[1, 1].set_title('Weight Standard Deviation')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # Summary table
        axes[1, 2].axis('tight')
        axes[1, 2].axis('off')
        table_data = df[['Model', 'Total Parameters', 'Total Layers', 'File Size (MB)']].values
        table = axes[1, 2].table(cellText=table_data, colLabels=df.columns[:4], 
                                cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        
        plt.tight_layout()
        plt.show()
        
        # Print detailed comparison
        print("\nDetailed Model Comparison:")
        print(df.to_string(index=False))
    
    def visualize_activations(self, model_name='default', input_image=None, layer_names=None):
        """Visualize intermediate activations"""
        if model_name not in self.models:
            print(f"Model '{model_name}' not found")
            return
        
        model = self.models[model_name]['model']
        
        # Create random input if none provided
        if input_image is None:
            input_shape = model.input_shape[1:]
            input_image = np.random.rand(1, *input_shape)
            print("Using random input image")
        
        # Select layers to visualize
        if layer_names is None:
            # Get first few conv/pooling layers
            layer_names = []
            for layer in model.layers:
                if any(x in layer.__class__.__name__.lower() for x in ['conv', 'pool']):
                    layer_names.append(layer.name)
                if len(layer_names) >= 4:
                    break
        
        if not layer_names:
            print("No suitable layers found for activation visualization")
            return
        
        # Create model that outputs intermediate activations
        layer_outputs = [model.get_layer(name).output for name in layer_names]
        activation_model = tf.keras.Model(inputs=model.input, outputs=layer_outputs)
        
        # Get activations
        activations = activation_model.predict(input_image, verbose=0)
        
        # Plot activations
        images_per_row = 8
        
        for layer_name, layer_activation in zip(layer_names, activations):
            n_features = layer_activation.shape[-1]
            size = layer_activation.shape[1]
            n_cols = min(n_features, images_per_row)
            n_rows = (n_features + n_cols - 1) // n_cols
            
            fig = plt.figure(figsize=(n_cols * 2, n_rows * 2))
            fig.suptitle(f'Activations: {layer_name}', fontsize=14)
            
            for i in range(min(n_features, n_cols * n_rows)):
                ax = plt.subplot(n_rows, n_cols, i + 1)
                ax.imshow(layer_activation[0, :, :, i], cmap='viridis')
                ax.axis('off')
            
            plt.tight_layout()
            plt.show()
    
    def extract_features(self, model_name='default', data=None, layer_name=None):
        """Extract features from a specific layer"""
        if model_name not in self.models:
            print(f"Model '{model_name}' not found")
            return None
        
        model = self.models[model_name]['model']
        
        # Select feature extraction layer
        if layer_name is None:
            # Find last layer before output
            for i in range(len(model.layers) - 1, -1, -1):
                if 'dense' in model.layers[i].__class__.__name__.lower():
                    if i < len(model.layers) - 1:  # Not the output layer
                        layer_name = model.layers[i].name
                        break
        
        if not layer_name:
            print("No suitable feature extraction layer found")
            return None
        
        # Create feature extraction model
        feature_layer = model.get_layer(layer_name)
        feature_model = tf.keras.Model(inputs=model.input, outputs=feature_layer.output)
        
        print(f"Extracting features from layer: {layer_name}")
        
        # Extract features
        if data is not None:
            features = feature_model.predict(data, verbose=0)
            return features
        else:
            print("No data provided for feature extraction")
            return None
    
    def analyze_predictions(self, model_name='default', test_data=None, test_labels=None):
        """Analyze model predictions"""
        if model_name not in self.models:
            print(f"Model '{model_name}' not found")
            return
        
        model = self.models[model_name]['model']
        
        if test_data is None or test_labels is None:
            print("No test data provided")
            return
        
        # Get predictions
        predictions = model.predict(test_data, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)
        
        # Calculate metrics
        accuracy = np.mean(predicted_classes == test_labels)
        
        # Confidence analysis
        confidences = np.max(predictions, axis=1)
        
        # Plot results
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Prediction Analysis - {model_name}', fontsize=16)
        
        # Confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(test_labels, predicted_classes)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
        axes[0, 0].set_title(f'Confusion Matrix (Accuracy: {accuracy:.2%})')
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('True')
        
        # Confidence distribution
        axes[0, 1].hist(confidences, bins=50, alpha=0.7, color='green')
        axes[0, 1].set_title('Prediction Confidence Distribution')
        axes[0, 1].set_xlabel('Confidence')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].axvline(confidences.mean(), color='red', linestyle='--', 
                         label=f'Mean: {confidences.mean():.3f}')
        axes[0, 1].legend()
        
        # Per-class accuracy
        classes = np.unique(test_labels)
        class_acc = []
        for c in classes:
            mask = test_labels == c
            class_acc.append(np.mean(predicted_classes[mask] == c))
        
        axes[1, 0].bar(classes, class_acc)
        axes[1, 0].set_title('Per-Class Accuracy')
        axes[1, 0].set_xlabel('Class')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].set_ylim(0, 1)
        
        # Confidence vs correctness
        correct = predicted_classes == test_labels
        axes[1, 1].scatter(confidences[correct], np.ones(sum(correct)), 
                         alpha=0.5, label='Correct', s=10)
        axes[1, 1].scatter(confidences[~correct], np.zeros(sum(~correct)), 
                         alpha=0.5, label='Incorrect', s=10)
        axes[1, 1].set_title('Confidence vs Correctness')
        axes[1, 1].set_xlabel('Confidence')
        axes[1, 1].set_ylabel('Correct (1) / Incorrect (0)')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.show()

# Example usage functions
def analyze_all_models(model_dir):
    """Analyze all models in a directory"""
    analyzer = ModelAnalyzer()
    
    # Load all .h5 files
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.h5')]
    
    print(f"Found {len(model_files)} models in {model_dir}")
    
    for model_file in model_files:
        model_path = os.path.join(model_dir, model_file)
        model_name = model_file.replace('.h5', '')
        analyzer.load_model(model_name, model_path)
    
    return analyzer

def quick_model_analysis(model_path):
    """Quick analysis of a single model"""
    analyzer = ModelAnalyzer(model_path)
    
    # Architecture analysis
    analyzer.analyze_architecture()
    
    # Weight analysis
    analyzer.analyze_weights()
    
    # Filter visualization (for CNNs)
    analyzer.visualize_filters()
    
    return analyzer

# Advanced analysis example
def advanced_analysis_example():
    """Example of advanced model analysis"""
    
    # Load models
    model_dir = 'C:/Users/spenc/Desktop/New folder (2)/Machine-Learning/Task 5/models'
    analyzer = analyze_all_models(model_dir)
    
    # Compare all models
    print("\n" + "="*60)
    print("COMPARING ALL MODELS")
    print("="*60)
    analyzer.compare_models()
    
    # Analyze specific model in detail
    if 'augmented_First_digit_best' in analyzer.models:
        print("\n" + "="*60)
        print("DETAILED ANALYSIS: augmented_First_digit_best")
        print("="*60)
        
        # Architecture
        analyzer.analyze_architecture('augmented_First_digit_best')
        
        # Weights
        analyzer.analyze_weights('augmented_First_digit_best')
        
        # Filters
        analyzer.visualize_filters('augmented_First_digit_best')
        
        # Activations with random input
        analyzer.visualize_activations('augmented_First_digit_best')

if __name__ == "__main__":
    # Quick analysis of a single model
    model_path = 'C:/Users/spenc/Desktop/New folder (2)/Machine-Learning/Task 5/mnist_classifier.h5'
    
    if os.path.exists(model_path):
        print("Analyzing MNIST classifier...")
        analyzer = quick_model_analysis(model_path)
    else:
        print(f"Model not found at {model_path}")
        print("Analyzing all models in Task 5 directory...")
        advanced_analysis_example()