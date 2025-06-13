#!/usr/bin/env python3
"""
Block Analysis Script for S3DIS/ScanNet Few-Shot Segmentation

Usage:
    python analyze_block.py path/to/block.npy
    python analyze_block.py path/to/block.npy --dataset scannet
    python analyze_block.py path/to/block.npy --no-plot
"""

import argparse
import numpy as np
import os
import sys


def get_class_names(dataset):
    """Get class names for different datasets."""
    if dataset.lower() == 's3dis':
        return ['ceiling', 'floor', 'wall', 'beam', 'column', 
                'window', 'door', 'chair', 'table', 'bookcase', 
                'sofa', 'board', 'clutter']
    elif dataset.lower() == 'scannet':
        return ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table',
                'door', 'window', 'bookshelf', 'picture', 'counter', 'blinds',
                'desk', 'shelves', 'curtain', 'dresser', 'pillow', 'mirror',
                'floor mat', 'clothes', 'ceiling', 'books', 'refridgerator',
                'television', 'paper', 'towel', 'shower curtain', 'box',
                'whiteboard', 'person', 'night stand', 'toilet', 'sink',
                'lamp', 'bathtub', 'bag', 'otherstructure', 'otherfurniture', 'otherprop']
    else:
        return []


def save_class_plot(xyz, labels, class_names, block_path, dataset):
    """Save a 3D plot of the block colored by classes."""
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-GUI backend for clusters
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        print("WARNING: matplotlib not available, skipping plot generation")
        return False
    
    # Create libs directory if it doesn't exist
    libs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'libs')
    os.makedirs(libs_dir, exist_ok=True)
    
    # Create figure
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot points colored by class labels
    scatter = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], 
                        c=labels, cmap='tab20', s=1, alpha=0.8)
    
    # Customize plot
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'Block: {os.path.basename(block_path)} - Colored by Classes\nDataset: {dataset.upper()}')
    
    # Add colorbar with class names
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, aspect=30)
    cbar.set_label('Class Label')
    
    # Add legend for class names
    unique_labels = np.unique(labels)
    legend_text = []
    for label in unique_labels:
        label_int = int(label)
        if label_int < len(class_names) and label_int >= 0:
            class_name = class_names[label_int]
        elif label_int == 255:
            class_name = "ignore"
        else:
            class_name = f"unknown_{label_int}"
        legend_text.append(f"{label_int}: {class_name}")
    
    # Add legend as text box
    legend_str = '\n'.join(legend_text[:10])  # Limit to first 10 classes
    if len(unique_labels) > 10:
        legend_str += f"\n... and {len(unique_labels)-10} more"
    
    ax.text2D(0.02, 0.98, legend_str, transform=ax.transAxes, 
              verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
              fontsize=8, family='monospace')
    
    # Generate output filename
    block_name = os.path.splitext(os.path.basename(block_path))[0]
    output_filename = f"{block_name}_classes.png"
    output_path = os.path.join(libs_dir, output_filename)
    
    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Class visualization saved to: {output_path}")
    return True


def analyze_block(block_path, dataset='s3dis', save_plot=True):
    """Analyze a block file and show statistics."""
    
    # Check if file exists
    if not os.path.exists(block_path):
        print(f"ERROR: File {block_path} does not exist!")
        return None
    
    try:
        # Load data
        block_data = np.load(block_path)
    except Exception as e:
        print(f"ERROR: Could not load {block_path}: {e}")
        return None
    
    # Extract data components
    if block_data.shape[1] < 7:
        print(f"ERROR: Expected at least 7 columns (XYZ+RGB+Label), got {block_data.shape[1]}")
        return None
        
    xyz = block_data[:, :3]      # 3D coordinates
    rgb = block_data[:, 3:6]     # RGB colors
    labels = block_data[:, 6]    # Class labels
    
    # Get class names
    class_names = get_class_names(dataset)
    
    # Print header
    print("=" * 70)
    print(f"BLOCK ANALYSIS: {os.path.basename(block_path)}")
    print("=" * 70)
    
    # Basic statistics
    print(f"Dataset: {dataset.upper()}")
    print(f"File path: {block_path}")
    print(f"Shape: {block_data.shape}")
    print(f"Total points: {len(xyz):,}")
    print()
    
    # Coordinate statistics
    print("COORDINATE RANGE:")
    x_range = xyz[:, 0].max() - xyz[:, 0].min()
    y_range = xyz[:, 1].max() - xyz[:, 1].min()
    z_range = xyz[:, 2].max() - xyz[:, 2].min()
    
    print(f"  X: {xyz[:, 0].min():7.2f} to {xyz[:, 0].max():7.2f} (range: {x_range:.2f}m)")
    print(f"  Y: {xyz[:, 1].min():7.2f} to {xyz[:, 1].max():7.2f} (range: {y_range:.2f}m)")
    print(f"  Z: {xyz[:, 2].min():7.2f} to {xyz[:, 2].max():7.2f} (range: {z_range:.2f}m)")
    print(f"  Block size: {x_range:.2f}m × {y_range:.2f}m × {z_range:.2f}m")
    print()
    
    # RGB statistics
    print("RGB COLOR RANGE:")
    print(f"  R: {rgb[:, 0].min():3.0f} to {rgb[:, 0].max():3.0f} (avg: {rgb[:, 0].mean():.1f})")
    print(f"  G: {rgb[:, 1].min():3.0f} to {rgb[:, 1].max():3.0f} (avg: {rgb[:, 1].mean():.1f})")
    print(f"  B: {rgb[:, 2].min():3.0f} to {rgb[:, 2].max():3.0f} (avg: {rgb[:, 2].mean():.1f})")
    print()
    
    # Label analysis
    unique_labels, counts = np.unique(labels, return_counts=True)
    print("CLASS DISTRIBUTION:")
    print("-" * 70)
    print(f"{'Label':<6} {'Class Name':<20} {'Points':<12} {'Percentage':<12} {'Bar':<15}")
    print("-" * 70)
    
    total_points = len(labels)
    max_count = max(counts)
    
    for label, count in zip(unique_labels, counts):
        label_int = int(label)
        
        # Get class name
        if label_int < len(class_names) and label_int >= 0:
            class_name = class_names[label_int]
        elif label_int == 255:
            class_name = "ignore"
        else:
            class_name = f"unknown_{label_int}"
        
        percentage = (count / total_points) * 100
        
        # Create simple bar chart with text
        bar_length = int((count / max_count) * 20)
        bar = "█" * bar_length + "░" * (20 - bar_length)
        
        print(f"{label_int:<6} {class_name:<20} {count:<12,} {percentage:<11.1f}% {bar}")
    
    print("-" * 70)
    print(f"Total: {len(unique_labels)} different classes")
    
    # Point density
    volume = x_range * y_range * z_range
    if volume > 0:
        density = total_points / volume
        print(f"Point density: {density:.0f} points/m³")
    
    print()
    
    # Memory usage
    memory_mb = block_data.nbytes / (1024 * 1024)
    print(f"Memory usage: {memory_mb:.2f} MB")
    
    # Few-shot relevance analysis
    print()
    print("FEW-SHOT RELEVANCE:")
    print("-" * 40)
    
    # Check for furniture classes (common few-shot targets)
    furniture_classes = ['chair', 'table', 'sofa', 'bed', 'bookcase', 'desk']
    found_furniture = []
    
    for label, count in zip(unique_labels, counts):
        label_int = int(label)
        if label_int < len(class_names):
            class_name = class_names[label_int]
            if class_name in furniture_classes:
                percentage = (count / total_points) * 100
                found_furniture.append((class_name, count, percentage))
    
    if found_furniture:
        print("Potential few-shot target classes found:")
        for class_name, count, percentage in found_furniture:
            print(f"  • {class_name}: {count:,} points ({percentage:.1f}%)")
    else:
        print("No common few-shot furniture classes found in this block.")
    
    print("=" * 70)
    
    # Generate plot if requested
    if save_plot:
        print("\nGenerating class visualization...")
        success = save_class_plot(xyz, labels, class_names, block_path, dataset)
        if not success:
            print("Plot generation failed.")
    
    return xyz, rgb, labels


def main():
    parser = argparse.ArgumentParser(
        description='Analyze 3D point cloud blocks from S3DIS or ScanNet datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_block.py Area_1_office_1_block_0.npy
  python analyze_block.py scene0000_00_block_5.npy --dataset scannet
  python analyze_block.py my_block.npy --dataset s3dis
  python analyze_block.py my_block.npy --no-plot
        """
    )
    
    parser.add_argument('block_path', 
                       help='Path to the .npy block file to analyze')
    
    parser.add_argument('--dataset', '-d',
                       choices=['s3dis', 'scannet'],
                       default='s3dis',
                       help='Dataset type (default: s3dis)')
    
    parser.add_argument('--no-plot',
                       action='store_true',
                       help='Skip plot generation (text analysis only)')
    
    args = parser.parse_args()
    
    # Analyze the block
    save_plot = not args.no_plot
    result = analyze_block(args.block_path, args.dataset, save_plot)
    
    if result is None:
        sys.exit(1)


if __name__ == "__main__":
    main()