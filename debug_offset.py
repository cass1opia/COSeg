#!/usr/bin/env python3

"""
Debug script to understand the offset processing issue in stratified_transformer.py
"""

import torch

def analyze_offset_processing():
    """
    Analyze the offset processing in TransitionDown and BasicLayer
    to understand why offset might have only 3 elements instead of batch size
    """
    
    print("ANALYZING OFFSET PROCESSING IN STRATIFIED TRANSFORMER")
    print("=" * 60)
    
    # Simulate different scenarios
    scenarios = [
        {"name": "Small batch", "batch_size": 2, "points_per_batch": [1000, 1500]},
        {"name": "Medium batch", "batch_size": 4, "points_per_batch": [1000, 1200, 800, 1500]},
        {"name": "Large batch", "batch_size": 8, "points_per_batch": [1000, 1200, 800, 1500, 900, 1100, 1300, 1400]},
    ]
    
    for scenario in scenarios:
        print(f"\nScenario: {scenario['name']}")
        print(f"Batch size: {scenario['batch_size']}")
        print(f"Points per batch: {scenario['points_per_batch']}")
        
        # Create cumulative offset tensor (how it's typically used)
        cumulative_offset = torch.zeros(scenario['batch_size'], dtype=torch.int32)
        cumulative_offset[0] = scenario['points_per_batch'][0]
        for i in range(1, scenario['batch_size']):
            cumulative_offset[i] = cumulative_offset[i-1] + scenario['points_per_batch'][i]
        
        print(f"Original offset: {cumulative_offset}")
        
        # Simulate TransitionDown processing
        ratio = 0.25
        print(f"Downsampling ratio: {ratio}")
        
        # This is how TransitionDown calculates n_offset
        n_offset = [int(cumulative_offset[0].item() * ratio) + 1]
        count = int(cumulative_offset[0].item() * ratio) + 1
        
        for i in range(1, cumulative_offset.shape[0]):
            count += int((cumulative_offset[i].item() - cumulative_offset[i-1].item()) * ratio) + 1
            n_offset.append(count)
        
        n_offset_tensor = torch.cuda.IntTensor(n_offset) if torch.cuda.is_available() else torch.IntTensor(n_offset)
        print(f"New offset after downsampling: {n_offset_tensor}")
        
        # Simulate BasicLayer processing
        downsample_scale = 8
        print(f"Downsample scale: {downsample_scale}")
        
        # This is how BasicLayer calculates new_offset
        new_offset = [cumulative_offset[0].item() // downsample_scale + 1]
        count = cumulative_offset[0].item() // downsample_scale + 1
        
        for i in range(1, cumulative_offset.shape[0]):
            count += (cumulative_offset[i].item() - cumulative_offset[i-1].item()) // downsample_scale + 1
            new_offset.append(count)
        
        new_offset_tensor = torch.cuda.IntTensor(new_offset) if torch.cuda.is_available() else torch.IntTensor(new_offset)
        print(f"New offset after BasicLayer processing: {new_offset_tensor}")
        
        # Check if there's a pattern that leads to 3 elements
        print(f"Original offset length: {len(cumulative_offset)}")
        print(f"TransitionDown offset length: {len(n_offset_tensor)}")
        print(f"BasicLayer offset length: {len(new_offset_tensor)}")
        
        print("-" * 40)
    
    print("\nPOTENTIAL ISSUE ANALYSIS:")
    print("=" * 60)
    print("The offset tensor should maintain the same length as the batch size")
    print("throughout the processing pipeline. If it's getting reduced to 3 elements,")
    print("it suggests that either:")
    print("1. The batch size is being hardcoded to 3 somewhere")
    print("2. There's an issue with how the offset is being passed between layers")
    print("3. The model is only processing the first 3 layers and cutting off the rest")
    print("4. There's a bug in the offset calculation that's causing it to lose elements")
    print()
    print("Looking at the Stratified class, line 828 shows:")
    print("self.layers = self.layers[:3]")
    print("This limits the processing to only the first 3 layers!")
    print()
    print("This could be the root cause - if the model is designed to only use")
    print("3 layers, then the offset might be getting truncated accordingly.")

if __name__ == "__main__":
    analyze_offset_processing()