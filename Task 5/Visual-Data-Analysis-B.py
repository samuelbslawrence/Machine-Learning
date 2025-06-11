import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import os

def visualize_model_architecture(model_path, save_path=None):
    """Create a beautiful visualization of model architecture"""
    model = tf.keras.models.load_model(model_path)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))
    
    # Left side: Layer structure
    ax1.set_title("Model Architecture", fontsize=16, fontweight='bold')
    
    y_position = 0
    layer_positions = {}
    colors = {'Conv2D': '#FF6B6B', 'Dense': '#4ECDC4', 'MaxPooling2D': '#45B7D1', 
              'Dropout': '#96CEB4', 'Flatten': '#FECA57', 'BatchNormalization': '#DDA0DD'}
    
    for i, layer in enumerate(model.layers):
        layer_type = layer.__class__.__name__
        color = colors.get(layer_type, '#95A5A6')
        
        # Draw layer box
        rect = Rectangle((0, y_position), 5, 0.8, facecolor=color, edgecolor='black', linewidth=2)
        ax1.add_patch(rect)
        
        # Add layer info
        params = layer.count_params()
        if hasattr(layer, 'output_shape'):
            shape_str = str(layer.output_shape)
        else:
            shape_str = "Multiple"
        
        ax1.text(0.1, y_position + 0.4, f"{layer.name}", fontsize=10, va='center')
        ax1.text(5.2, y_position + 0.5, f"{shape_str}", fontsize=8, va='center')
        ax1.text(5.2, y_position + 0.2, f"{params:,} params", fontsize=8, va='center', style='italic')
        
        layer_positions[i] = y_position + 0.4
        y_position += 1
    
    # Draw connections
    for i in range(len(model.layers) - 1):
        ax1.plot([2.5, 2.5], [layer_positions[i], layer_positions[i+1]], 'k-', linewidth=2)
    
    ax1.set_xlim(-1, 10)
    ax1.set_ylim(-0.5, y_position)
    ax1.axis('off')
    
    # Add legend
    for layer_type, color in colors.items():
        ax1.add_patch(Rectangle((7, y_position - len(colors) + list(colors.keys()).index(layer_type)), 
                               0.3, 0.3, facecolor=color))
        ax1.text(7.5, y_position - len(colors) + list(colors.keys()).index(layer_type) + 0.15, 
                layer_type, fontsize=8, va='center')
    
    # Right side: Model statistics
    ax2.set_title("Model Statistics", fontsize=16, fontweight='bold')
    
    stats_text = f"""
Total Parameters: {model.count_params():,}
Trainable Parameters: {sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]):,}
Total Layers: {len(model.layers)}
Input Shape: {model.input_shape}
Output Shape: {model.output_shape}
Model Size: {os.path.getsize(model_path) / (1024*1024):.2f} MB
    """
    
    ax2.text(0.1, 0.8, stats_text, fontsize=12, transform=ax2.transAxes, 
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Layer type distribution
    layer_types = {}
    for layer in model.layers:
        layer_type = layer.__class__.__name__
        layer_types[layer_type] = layer_types.get(layer_type, 0) + 1
    
    # Pie chart
    if layer_types:
        wedges, texts, autotexts = ax2.pie(layer_types.values(), labels=layer_types.keys(), 
                                           autopct='%1.1f%%', startangle=90,
                                           colors=[colors.get(k, '#95A5A6') for k in layer_types.keys()])
        ax2.set_position([0.5, 0.1, 0.4, 0.4])
    
    ax2.axis('off')
    
    plt.suptitle(f"Model Analysis: {os.path.basename(model_path)}", fontsize=18, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def visualize_filters_grid(model_path, layer_index=0):
    """Visualize convolutional filters in a grid"""
    model = tf.keras.models.load_model(model_path)
    
    # Find conv layers
    conv_layers = [(i, layer) for i, layer in enumerate(model.layers) 
                   if 'conv' in layer.__class__.__name__.lower()]
    
    if not conv_layers:
        print("No convolutional layers found!")
        return
    
    layer_idx, layer = conv_layers[min(layer_index, len(conv_layers)-1)]
    weights = layer.get_weights()[0]
    
    print(f"Visualizing layer: {layer.name}")
    print(f"Filter shape: {weights.shape}")
    
    # Normalize filters
    f_min, f_max = weights.min(), weights.max()
    filters = (weights - f_min) / (f_max - f_min)
    
    # Create grid
    n_filters = min(64, filters.shape[3])
    n_cols = 8
    n_rows = (n_filters + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 2))
    fig.suptitle(f'Convolutional Filters: {layer.name}\n{os.path.basename(model_path)}', 
                 fontsize=16, fontweight='bold')
    
    axes = axes.flatten() if n_rows > 1 else [axes]
    
    for i in range(n_filters):
        ax = axes[i]
        
        # Get filter
        if filters.shape[2] == 1:
            filter_img = filters[:, :, 0, i]
            cmap = 'viridis'
        else:
            filter_img = filters[:, :, :3, i]
            cmap = None
        
        im = ax.imshow(filter_img, cmap=cmap)
        ax.set_title(f'Filter {i}', fontsize=8)
        ax.axis('off')
        
        # Add colorbar for single channel
        if filters.shape[2] == 1:
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=6)
    
    # Hide unused subplots
    for i in range(n_filters, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def create_weight_heatmap(model_path):
    """Create a heatmap of weight magnitudes across layers"""
    model = tf.keras.models.load_model(model_path)
    
    # Collect weight statistics
    layer_names = []
    weight_stats = []
    
    for layer in model.layers:
        weights = layer.get_weights()
        if weights and len(weights[0].shape) >= 2:
            layer_names.append(layer.name)
            
            # Calculate statistics
            w = weights[0]
            stats = {
                'mean': np.mean(np.abs(w)),
                'std': np.std(w),
                'max': np.max(np.abs(w)),
                'sparsity': np.sum(np.abs(w) < 0.01) / w.size
            }
            weight_stats.append(stats)
    
    if not weight_stats:
        print("No layers with weights found!")
        return
    
    # Create heatmap data
    stat_names = ['mean', 'std', 'max', 'sparsity']
    heatmap_data = np.array([[stats[stat] for stats in weight_stats] for stat in stat_names])
    
    # Normalize each statistic
    for i in range(len(stat_names)):
        if heatmap_data[i].max() > 0:
            heatmap_data[i] = heatmap_data[i] / heatmap_data[i].max()
    
    # Create heatmap
    plt.figure(figsize=(max(12, len(layer_names) * 0.5), 6))
    
    sns.heatmap(heatmap_data, 
                xticklabels=[name[:15] + '...' if len(name) > 15 else name for name in layer_names],
                yticklabels=['Mean |W|', 'Std Dev', 'Max |W|', 'Sparsity'],
                cmap='YlOrRd', 
                annot=True, 
                fmt='.3f',
                cbar_kws={'label': 'Normalized Value'})
    
    plt.title(f'Weight Statistics Heatmap\n{os.path.basename(model_path)}', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Layer', fontsize=12)
    plt.ylabel('Statistic', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def visualize_learning_capacity(model_path):
    """Visualize model's learning capacity and complexity"""
    model = tf.keras.models.load_model(model_path)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Model Learning Capacity Analysis\n{os.path.basename(model_path)}', 
                 fontsize=16, fontweight='bold')
    
    # 1. Parameter distribution by layer type
    layer_params = {}
    for layer in model.layers:
        layer_type = layer.__class__.__name__
        params = layer.count_params()
        if layer_type in layer_params:
            layer_params[layer_type] += params
        else:
            layer_params[layer_type] = params
    
    if layer_params:
        colors = plt.cm.Set3(np.linspace(0, 1, len(layer_params)))
        wedges, texts, autotexts = ax1.pie(layer_params.values(), 
                                           labels=layer_params.keys(),
                                           autopct='%1.1f%%',
                                           colors=colors,
                                           explode=[0.05] * len(layer_params))
        ax1.set_title('Parameter Distribution by Layer Type')
    
    # 2. Layer depth vs parameters
    layer_names = []
    param_counts = []
    layer_types = []
    
    for layer in model.layers:
        if layer.count_params() > 0:
            layer_names.append(layer.name[:10] + '...' if len(layer.name) > 10 else layer.name)
            param_counts.append(layer.count_params())
            layer_types.append(layer.__class__.__name__)
    
    if param_counts:
        bars = ax2.bar(range(len(param_counts)), param_counts)
        
        # Color by layer type
        unique_types = list(set(layer_types))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_types)))
        type_colors = {t: colors[i] for i, t in enumerate(unique_types)}
        
        for bar, ltype in zip(bars, layer_types):
            bar.set_color(type_colors[ltype])
        
        ax2.set_xlabel('Layer Index')
        ax2.set_ylabel('Number of Parameters')
        ax2.set_title('Parameters per Layer')
        ax2.set_yscale('log')
        
        # Add legend
        handles = [plt.Rectangle((0,0),1,1, color=type_colors[t]) for t in unique_types]
        ax2.legend(handles, unique_types, loc='upper right')
    
    # 3. Receptive field evolution (for CNNs)
    receptive_fields = []
    spatial_dims = []
    
    for layer in model.layers:
        if hasattr(layer, 'output_shape') and len(layer.output_shape) == 4:
            spatial_dims.append(layer.output_shape[1])
            
            # Simple receptive field calculation
            if 'conv' in layer.__class__.__name__.lower():
                kernel_size = layer.kernel_size[0] if hasattr(layer, 'kernel_size') else 1
                receptive_fields.append(kernel_size)
    
    if spatial_dims:
        ax3.plot(spatial_dims, 'b-', marker='o', label='Spatial Dimension')
        ax3.set_xlabel('Conv Layer Index')
        ax3.set_ylabel('Spatial Dimension')
        ax3.set_title('Spatial Dimension Reduction')
        ax3.grid(True, alpha=0.3)
    
    # 4. Model complexity metrics
    total_params = model.count_params()
    memory_mb = (total_params * 4) / (1024 * 1024)  # 4 bytes per float32
    
    metrics = {
        'Total Parameters': f'{total_params:,}',
        'Memory (MB)': f'{memory_mb:.2f}',
        'Total Layers': len(model.layers),
        'Conv Layers': sum(1 for l in model.layers if 'conv' in l.__class__.__name__.lower()),
        'Dense Layers': sum(1 for l in model.layers if 'dense' in l.__class__.__name__.lower()),
        'Trainable Params': f'{sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]):,}'
    }
    
    ax4.axis('off')
    y_pos = 0.9
    for key, value in metrics.items():
        ax4.text(0.1, y_pos, f'{key}:', fontsize=12, fontweight='bold')
        ax4.text(0.6, y_pos, str(value), fontsize=12)
        y_pos -= 0.15
    
    ax4.set_title('Model Complexity Metrics')
    ax4.add_patch(Rectangle((0.05, 0.05), 0.9, 0.9, fill=False, edgecolor='black', linewidth=2))
    
    plt.tight_layout()
    plt.show()

def compare_all_models(model_dir):
    """Create a comprehensive comparison of all models in a directory"""
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.h5')]
    
    if not model_files:
        print(f"No .h5 model files found in {model_dir}")
        return
    
    # Collect data for all models
    model_data = []
    
    for model_file in model_files:
        try:
            model_path = os.path.join(model_dir, model_file)
            model = tf.keras.models.load_model(model_path)
            
            # Calculate metrics
            total_params = model.count_params()
            conv_layers = sum(1 for l in model.layers if 'conv' in l.__class__.__name__.lower())
            dense_layers = sum(1 for l in model.layers if 'dense' in l.__class__.__name__.lower())
            
            model_data.append({
                'name': model_file.replace('.h5', ''),
                'params': total_params,
                'layers': len(model.layers),
                'conv': conv_layers,
                'dense': dense_layers,
                'size_mb': os.path.getsize(model_path) / (1024*1024)
            })
            
            print(f"Loaded: {model_file}")
        except Exception as e:
            print(f"Error loading {model_file}: {e}")
    
    if not model_data:
        print("No models could be loaded")
        return
    
    # Create comparison visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Model Comparison: {os.path.basename(model_dir)}', fontsize=18, fontweight='bold')
    
    names = [m['name'][:15] + '...' if len(m['name']) > 15 else m['name'] for m in model_data]
    
    # 1. Total parameters
    params = [m['params'] for m in model_data]
    bars1 = ax1.bar(names, params, color='skyblue', edgecolor='navy')
    ax1.set_ylabel('Total Parameters')
    ax1.set_title('Model Size (Parameters)')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, param in zip(bars1, params):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{param:,}', ha='center', va='bottom', fontsize=8, rotation=90)
    
    # 2. Layer composition
    x = np.arange(len(names))
    width = 0.35
    
    conv_counts = [m['conv'] for m in model_data]
    dense_counts = [m['dense'] for m in model_data]
    
    bars2 = ax2.bar(x - width/2, conv_counts, width, label='Conv Layers', color='orange')
    bars3 = ax2.bar(x + width/2, dense_counts, width, label='Dense Layers', color='green')
    
    ax2.set_ylabel('Number of Layers')
    ax2.set_title('Layer Type Distribution')
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=45)
    ax2.legend()
    
    # 3. File size
    sizes = [m['size_mb'] for m in model_data]
    bars4 = ax3.bar(names, sizes, color='coral', edgecolor='darkred')
    ax3.set_ylabel('File Size (MB)')
    ax3.set_title('Model File Sizes')
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. Scatter plot: Parameters vs Layers
    ax4.scatter([m['params'] for m in model_data], 
                [m['layers'] for m in model_data],
                s=100, alpha=0.6, c=range(len(model_data)), cmap='viridis')
    
    # Add labels
    for i, m in enumerate(model_data):
        ax4.annotate(m['name'][:10], 
                    (m['params'], m['layers']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax4.set_xlabel('Total Parameters')
    ax4.set_ylabel('Total Layers')
    ax4.set_title('Model Complexity Distribution')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary table
    print("\nModel Summary Table:")
    print("-" * 80)
    print(f"{'Model':<30} {'Parameters':>12} {'Layers':>8} {'Conv':>6} {'Dense':>6} {'Size(MB)':>10}")
    print("-" * 80)
    for m in model_data:
        print(f"{m['name']:<30} {m['params']:>12,} {m['layers']:>8} {m['conv']:>6} {m['dense']:>6} {m['size_mb']:>10.2f}")

# Example usage
if __name__ == "__main__":
    # Analyze a single model
    model_path = 'C:/Users/spenc/Desktop/New folder (2)/Machine-Learning/Task 5/mnist_classifier.h5'
    
    if os.path.exists(model_path):
        print("Analyzing MNIST classifier...")
        
        # Architecture visualization
        visualize_model_architecture(model_path)
        
        # Filter visualization
        visualize_filters_grid(model_path)
        
        # Weight heatmap
        create_weight_heatmap(model_path)
        
        # Learning capacity
        visualize_learning_capacity(model_path)
    
    # Compare all models in directory
    model_dir = 'C:/Users/spenc/Desktop/New folder (2)/Machine-Learning/Task 5/models'
    if os.path.exists(model_dir):
        print(f"\nComparing all models in {model_dir}...")
        compare_all_models(model_dir)