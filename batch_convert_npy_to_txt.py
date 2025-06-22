#!/usr/bin/env python3
"""
Script to convert ALL .npy files in a directory to .txt files.
Each .npy file should contain data with shape (N, 7) where columns are [X, Y, Z, R, G, B, Label].
Output .txt files will have the same format: X Y Z R G B Label (space-separated).
"""

import os
import numpy as np
import glob
import argparse
from pathlib import Path


def convert_npy_to_txt(npy_file_path, txt_file_path):
    """
    Convert a single .npy file to .txt format.
    
    Args:
        npy_file_path: Path to input .npy file
        txt_file_path: Path to output .txt file
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Load numpy data
        data = np.load(npy_file_path)
        
        # Check if data has expected shape (N, 7)
        if data.ndim != 2:
            print(f"Warning: {npy_file_path} has unexpected dimensions {data.shape}")
            return False
            
        if data.shape[1] != 7:
            print(f"Warning: {npy_file_path} has {data.shape[1]} columns, expected 7 (X Y Z R G B Label)")
            # Continue anyway, might work
        
        # Write to txt file
        with open(txt_file_path, 'w') as f:
            for i in range(data.shape[0]):
                if data.shape[1] == 7:
                    # Standard format: X Y Z R G B Label
                    f.write(f"{data[i, 0]:.6f} {data[i, 1]:.6f} {data[i, 2]:.6f} "
                           f"{int(data[i, 3])} {int(data[i, 4])} {int(data[i, 5])} {int(data[i, 6])}\n")
                else:
                    # Generic format: write all columns as floats
                    line = " ".join([f"{data[i, j]:.6f}" for j in range(data.shape[1])])
                    f.write(line + "\n")
        
        print(f"✓ Converted: {os.path.basename(npy_file_path)} -> {os.path.basename(txt_file_path)}")
        return True
        
    except Exception as e:
        print(f"✗ Error converting {npy_file_path}: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Convert ALL .npy files in a directory to .txt files")
    parser.add_argument(
        "--input_dir", 
        type=str, 
        default="datasets/test",
        help="Directory containing .npy files (default: datasets/test)"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=None,
        help="Output directory for .txt files (default: same as input_dir)"
    )
    parser.add_argument(
        "--overwrite", 
        action="store_true",
        help="Overwrite existing .txt files"
    )
    parser.add_argument(
        "--recursive", "-r",
        action="store_true",
        help="Search for .npy files recursively in subdirectories"
    )
    
    args = parser.parse_args()
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = args.input_dir
    
    # Check if input directory exists
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist!")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all .npy files
    if args.recursive:
        npy_files = glob.glob(os.path.join(args.input_dir, "**/*.npy"), recursive=True)
    else:
        npy_files = glob.glob(os.path.join(args.input_dir, "*.npy"))
    
    if not npy_files:
        print(f"No .npy files found in '{args.input_dir}'")
        return
    
    print(f"Found {len(npy_files)} .npy files in '{args.input_dir}'")
    print(f"Output directory: '{args.output_dir}'")
    print(f"Recursive search: {args.recursive}")
    print("-" * 60)
    
    converted_count = 0
    skipped_count = 0
    error_count = 0
    
    for npy_file in sorted(npy_files):
        # Generate output filename
        if args.recursive:
            # Maintain directory structure for recursive search
            rel_path = os.path.relpath(npy_file, args.input_dir)
            base_name = os.path.splitext(rel_path)[0]
            txt_file = os.path.join(args.output_dir, base_name + ".txt")
            # Create subdirectories if needed
            os.makedirs(os.path.dirname(txt_file), exist_ok=True)
        else:
            base_name = os.path.splitext(os.path.basename(npy_file))[0]
            txt_file = os.path.join(args.output_dir, base_name + ".txt")
        
        # Check if output file already exists
        if os.path.exists(txt_file) and not args.overwrite:
            print(f"⏭ Skipped: {os.path.basename(txt_file)} (already exists, use --overwrite to force)")
            skipped_count += 1
            continue
        
        # Convert file
        if convert_npy_to_txt(npy_file, txt_file):
            converted_count += 1
        else:
            error_count += 1
    
    print("-" * 60)
    print(f"Conversion Summary:")
    print(f"✓ Converted: {converted_count} files")
    print(f"⏭ Skipped: {skipped_count} files")
    print(f"✗ Errors: {error_count} files")


if __name__ == "__main__":
    main()