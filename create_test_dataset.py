#!/usr/bin/env python3
"""
Create a minimal test dataset for ESSEN_OUTDOOR to verify the implementation.
This creates synthetic point cloud blocks with random data.
"""

import os
import numpy as np
from pathlib import Path

def create_synthetic_block(num_points=1000, class_id=0, block_name="test_block_0"):
    """Create a synthetic point cloud block."""
    
    # Generate random 3D coordinates (1m x 1m x 3m block)
    x = np.random.uniform(0, 1, num_points)
    y = np.random.uniform(0, 1, num_points) 
    z = np.random.uniform(0, 3, num_points)
    
    # Generate random RGB colors
    r = np.random.randint(0, 256, num_points)
    g = np.random.randint(0, 256, num_points)
    b = np.random.randint(0, 256, num_points)
    
    # Create labels - mostly the target class with some noise
    labels = np.full(num_points, class_id, dtype=np.int32)
    
    # Add some other classes for realism (10% noise)
    noise_indices = np.random.choice(num_points, size=num_points//10, replace=False)
    noise_classes = np.random.choice([c for c in range(10) if c != class_id], 
                                   size=len(noise_indices))
    labels[noise_indices] = noise_classes
    
    # Combine into (N, 7) array: [X, Y, Z, R, G, B, Label]
    block_data = np.column_stack([x, y, z, r, g, b, labels])
    
    print(f"Created block {block_name}: {block_data.shape}, main class: {class_id}, "
          f"points of main class: {np.sum(labels == class_id)}")
    
    return block_data

def create_test_dataset():
    """Create the minimal ESSEN_OUTDOOR test dataset."""
    
    # Create directory structure
    base_dir = Path("test_essen_outdoor")
    data_dir = base_dir / "blocks_bs1_s1" / "data"
    meta_dir = base_dir / "meta"
    
    # Create directories
    data_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    
    # Create class names file
    class_names = [
        "road",
        "sidewalk", 
        "building",
        "vegetation",
        "vehicle",
        "person",
        "bicycle",
        "traffic_sign",
        "pole",
        "fence"
    ]
    
    with open(meta_dir / "essen_outdoor_classnames.txt", "w") as f:
        for name in class_names:
            f.write(f"{name}\n")
    
    print(f"Created class names file with {len(class_names)} classes")
    
    # Create synthetic blocks for each class
    # We need enough blocks per class to satisfy the few-shot requirements
    blocks_per_class = 10  # Minimum for k_shot=5 + some extra
    
    block_count = 0
    for class_id, class_name in enumerate(class_names):
        print(f"\nCreating blocks for class {class_id}: {class_name}")
        
        for block_idx in range(blocks_per_class):
            # Vary the number of points to make it realistic
            num_points = np.random.randint(800, 1500)
            
            block_name = f"scene_{class_id:02d}_{block_idx:03d}_block_0"
            block_data = create_synthetic_block(
                num_points=num_points,
                class_id=class_id,
                block_name=block_name
            )
            
            # Save block
            output_path = data_dir / f"{block_name}.npy"
            np.save(output_path, block_data)
            block_count += 1
    
    print(f"\n=== DATASET CREATION COMPLETE ===")
    print(f"Created {block_count} blocks total")
    print(f"Dataset location: {base_dir.absolute()}")
    print(f"Data files: {data_dir}")
    print(f"Meta files: {meta_dir}")
    
    # Create a simple config snippet
    config_snippet = f"""
# Add this to your config file:
DATA:
  data_name: essen_outdoor
  data_root: {data_dir.absolute()}
  classes: {len(class_names)}
  fea_dim: 6
"""
    
    with open(base_dir / "config_snippet.txt", "w") as f:
        f.write(config_snippet)
    
    print(f"\nConfig snippet saved to: {base_dir / 'config_snippet.txt'}")
    
    # Verify the dataset
    print("\n=== VERIFICATION ===")
    verify_dataset(data_dir, class_names)
    
    return base_dir

def verify_dataset(data_dir, class_names):
    """Verify the created dataset."""
    
    files = list(data_dir.glob("*.npy"))
    print(f"Found {len(files)} .npy files")
    
    # Check class distribution
    class_counts = {i: 0 for i in range(len(class_names))}
    total_points = 0
    
    for file_path in files[:5]:  # Check first 5 files
        data = np.load(file_path)
        labels = data[:, 6].astype(int)
        
        print(f"{file_path.name}: shape={data.shape}, "
              f"classes={np.unique(labels)}, "
              f"main_class_points={np.max(np.bincount(labels))}")
        
        for class_id in range(len(class_names)):
            class_counts[class_id] += np.sum(labels == class_id)
        
        total_points += len(labels)
    
    print(f"\nClass distribution (first 5 files):")
    for class_id, count in class_counts.items():
        if count > 0:
            print(f"  {class_id} ({class_names[class_id]}): {count} points")
    
    print(f"Total points sampled: {total_points}")

if __name__ == "__main__":
    print("Creating minimal ESSEN_OUTDOOR test dataset...")
    dataset_path = create_test_dataset()
    
    print(f"\n=== READY TO TEST ===")
    print("You can now test with:")
    print(f"python analyze_block.py {dataset_path}/blocks_bs1_s1/data/scene_00_000_block_0.npy --dataset essen_outdoor")
    print("\nOr test the dataset loading with your few-shot code!")