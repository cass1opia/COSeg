#!/usr/bin/env python3
"""
Test script for MultiPointCloudLoaderFS on compute cluster.
Saves neighborhoods for k epochs (5) to test_vis folder.
"""

import os
import sys
import argparse
import numpy as np
from util import config
from pcnn.multi_point_cloud_loader import get_dataloaders_fs
import torch
from pathlib import Path

# Add project root to path (go up one level from scripts/ to project root)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)




def save_episode_data(episode_data, epoch, episode_idx, save_dir):
    """Save episode data (support/query neighborhoods) to files."""

    (support_ptclouds, support_base_masks, support_test_masks,
     query_ptclouds, query_base_labels, query_test_labels, sampled_classes) = episode_data

    # Create episode directory
    episode_dir = os.path.join(save_dir, f"epoch_{epoch}", f"episode_{episode_idx}")
    os.makedirs(episode_dir, exist_ok=True)

    # Save sampled classes
    np.save(os.path.join(episode_dir, "sampled_classes.npy"), sampled_classes)

    # Save support data
    support_dir = os.path.join(episode_dir, "support")
    os.makedirs(support_dir, exist_ok=True)

    for i, (ptcloud, base_mask, test_mask) in enumerate(zip(support_ptclouds, support_base_masks, support_test_masks)):
        np.save(os.path.join(support_dir, f"ptcloud_{i}.npy"), ptcloud.cpu().numpy())
        np.save(os.path.join(support_dir, f"base_mask_{i}.npy"), base_mask.cpu().numpy())
        np.save(os.path.join(support_dir, f"test_mask_{i}.npy"), test_mask.cpu().numpy())

    # Save query data
    query_dir = os.path.join(episode_dir, "query")
    os.makedirs(query_dir, exist_ok=True)

    for i, (ptcloud, base_label, test_label) in enumerate(zip(query_ptclouds, query_base_labels, query_test_labels)):
        np.save(os.path.join(query_dir, f"ptcloud_{i}.npy"), ptcloud.cpu().numpy())
        np.save(os.path.join(query_dir, f"base_label_{i}.npy"), base_label.cpu().numpy())
        np.save(os.path.join(query_dir, f"test_label_{i}.npy"), test_label.cpu().numpy())

    print(f"Saved episode {episode_idx} for epoch {epoch}")
    print(f"  - Sampled classes: {sampled_classes}")
    print(f"  - Support samples: {len(support_ptclouds)}")
    print(f"  - Query samples: {len(query_ptclouds)}")


def test_multipoint_loader(config_path, num_epochs=5, episodes_per_epoch=3, save_dir="./test_vis"):
    """Test MultiPointCloudLoaderFS and save neighborhood data."""

    print("=" * 60)
    print("Testing MultiPointCloudLoaderFS")
    print("=" * 60)

    # Load config
    args = config.load_cfg_from_cfg_file(config_path)
    print(f"Loaded config from: {config_path}")
    print(f"Data name: {args.data_name}")
    print(f"Data root: {args.data_root}")

    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    try:
        # Create dataloaders
        print("\nCreating dataloaders...")
        train_loader, test_loader = get_dataloaders_fs(
            data_dir=args.data_root,
            cache_dir=getattr(args, 'cache_dir', './cache'),
            class_ids=getattr(args, 'class_ids', list(range(13))),
            cvfold=getattr(args, 'cvfold', 0),
            k=getattr(args, 'k_neighbors', 32),
            neighborhood_type=getattr(args, 'neighborhood_type', 'ball'),
            neighborhood_sampling=getattr(args, 'neighborhood_sampling', 'random_sampling'),
            radius=getattr(args, 'radius', 0.1),
            n_way=args.n_way,
            k_shot=args.k_shot,
            n_queries=args.n_queries,
            num_episode=args.num_episode,
            voxel_size=args.voxel_size,
            voxel_max=args.voxel_max,
            seed=getattr(args, 'manual_seed', None)
        )

        print(f"✓ Train loader created with {len(train_loader)} episodes")
        print(f"✓ Test loader created with {len(test_loader)} episodes")

        # Test train loader
        print(f"\nTesting train loader for {num_epochs} epochs...")
        for epoch in range(num_epochs):
            print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")

            # Iterate through episodes
            for episode_idx, episode_data in enumerate(train_loader):
                if episode_idx >= episodes_per_epoch:
                    break

                print(f"Processing episode {episode_idx + 1}/{episodes_per_epoch}...")

                # Save episode data
                save_episode_data(episode_data, epoch + 1, episode_idx + 1, save_dir)

        print(f"\n✓ Test completed successfully!")
        print(f"✓ Data saved to: {save_dir}")

        # Save summary
        summary = {
            'config_path': config_path,
            'num_epochs_tested': num_epochs,
            'episodes_per_epoch': episodes_per_epoch,
            'train_loader_episodes': len(train_loader),
            'test_loader_episodes': len(test_loader),
            'args': {
                'data_name': args.data_name,
                'data_root': args.data_root,
                'n_way': args.n_way,
                'k_shot': args.k_shot,
                'n_queries': args.n_queries,
                'cvfold': getattr(args, 'cvfold', 0),
            }
        }

        import json
        with open(os.path.join(save_dir, "test_summary.json"), 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"✓ Summary saved to: {save_dir}/test_summary.json")

    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


def main():
    parser = argparse.ArgumentParser(description="Test MultiPointCloudLoaderFS")
    parser.add_argument(
        "--config",
        type=str,
        default="config/essen_outdoor_COSeg_fs.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of epochs to test"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Number of episodes per epoch to save"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./test_vis",
        help="Directory to save test data"
    )

    args = parser.parse_args()

    print("MultiPointCloudLoaderFS Test Script")
    print(f"Config: {args.config}")
    print(f"Epochs: {args.epochs}")
    print(f"Episodes per epoch: {args.episodes}")
    print(f"Save directory: {args.save_dir}")

    success = test_multipoint_loader(
        config_path=args.config,
        num_epochs=args.epochs,
        episodes_per_epoch=args.episodes,
        save_dir=args.save_dir
    )

    if success:
        print("\n🎉 Test completed successfully!")
    else:
        print("\n💥 Test failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
