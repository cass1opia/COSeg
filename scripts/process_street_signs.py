#!/usr/bin/env python3
"""
Script to process point cloud files for street sign labeling.

This script processes point cloud files in CSV/TXT format and performs the following operations:
1. Save original SemClassID values in SpecificClassID column
2. For points where original SemClassID == 15 (StreetSign): Map original SpecificClassID to SemClassID
3. Save processed files with suffix "_street_sign_labeling.txt"

Input format: X,Y,Z,R,G,B,SemClassID,SpecificClassID,Intensity
Output format: Same structure but with swapped SemClassID/SpecificClassID columns

Usage:
    python scripts/process_street_signs.py --input_dir /path/to/input --output_dir /path/to/output
"""

import os
import sys
import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import numpy as np


def process_point_cloud_file(input_file, output_file):
    """
    Process a single point cloud file.
    
    Args:
        input_file (str): Path to input point cloud file
        output_file (str): Path to output processed file
    
    Returns:
        dict: Statistics about the processing
    """
    
    try:
        # Read the point cloud file
        # Expected format: X,Y,Z,R,G,B,SemClassID,SpecificClassID,Intensity
        df = pd.read_csv(input_file, header=None, names=[
            'X', 'Y', 'Z', 'R', 'G', 'B', 'SemClassID', 'SpecificClassID', 'Intensity'
        ])
        
        # Store original counts for statistics
        original_street_signs = len(df[df['SemClassID'] == 15])
        total_points = len(df)
        
        # Create a copy for processing
        processed_df = df.copy()
        
        # Step 1: Save original SemClassID in SpecificClassID column
        processed_df['SpecificClassID'] = df['SemClassID']
        
        # Step 2: For points where original SemClassID == 15 (StreetSign), 
        # map original SpecificClassID to SemClassID
        street_sign_mask = df['SemClassID'] == 15
        if street_sign_mask.any():
            processed_df.loc[street_sign_mask, 'SemClassID'] = df.loc[street_sign_mask, 'SpecificClassID']
        
        # Save the processed file
        processed_df.to_csv(output_file, header=False, index=False, float_format='%.6f')
        
        # Calculate statistics
        unique_original_sem_classes = df['SemClassID'].nunique()
        unique_original_specific_classes = df['SpecificClassID'].nunique()
        unique_processed_classes = processed_df['SemClassID'].nunique()
        
        stats = {
            'total_points': total_points,
            'original_street_signs': original_street_signs,
            'unique_original_sem_classes': unique_original_sem_classes,
            'unique_original_specific_classes': unique_original_specific_classes,
            'unique_processed_classes': unique_processed_classes,
            'success': True,
            'error': None
        }
        
        return stats
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'total_points': 0,
            'original_street_signs': 0,
            'unique_original_sem_classes': 0,
            'unique_original_specific_classes': 0,
            'unique_processed_classes': 0
        }


def process_directory(input_dir, output_dir, file_extensions=None):
    """
    Process all point cloud files in a directory.
    
    Args:
        input_dir (str): Directory containing input point cloud files
        output_dir (str): Directory to save processed files
        file_extensions (list): List of file extensions to process (default: ['.txt', '.csv'])
    
    Returns:
        dict: Summary statistics of the processing
    """
    
    if file_extensions is None:
        file_extensions = ['.txt', '.csv']
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all point cloud files
    point_cloud_files = []
    for ext in file_extensions:
        point_cloud_files.extend(input_path.glob(f'**/*{ext}'))
    
    if not point_cloud_files:
        print(f"No point cloud files found in {input_dir} with extensions {file_extensions}")
        return None
    
    print(f"Found {len(point_cloud_files)} point cloud files to process")
    
    # Process statistics
    total_stats = {
        'files_processed': 0,
        'files_failed': 0,
        'total_points': 0,
        'total_street_signs': 0,
        'failed_files': []
    }
    
    # Process each file
    for input_file in tqdm(point_cloud_files, desc="Processing files"):
        # Generate output filename
        original_name = input_file.stem
        output_filename = f"{original_name}_street_sign_labeling.txt"
        output_file = output_path / output_filename
        
        # Process the file
        stats = process_point_cloud_file(input_file, output_file)
        
        if stats['success']:
            total_stats['files_processed'] += 1
            total_stats['total_points'] += stats['total_points']
            total_stats['total_street_signs'] += stats['original_street_signs']
            
            print(f"✓ {input_file.name}: {stats['total_points']} points, "
                  f"{stats['original_street_signs']} street signs, "
                  f"{stats['unique_processed_classes']} unique classes")
        else:
            total_stats['files_failed'] += 1
            total_stats['failed_files'].append({
                'file': str(input_file),
                'error': stats['error']
            })
            print(f"✗ {input_file.name}: {stats['error']}")
    
    return total_stats


def main():
    parser = argparse.ArgumentParser(
        description="Process point cloud files for street sign labeling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process all files in input directory
    python scripts/process_street_signs.py --input_dir /path/to/input --output_dir /path/to/output
    
    # Process only specific file types
    python scripts/process_street_signs.py --input_dir /path/to/input --output_dir /path/to/output --extensions .txt .csv
    
    # Show verbose output
    python scripts/process_street_signs.py --input_dir /path/to/input --output_dir /path/to/output --verbose
        """
    )
    
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing input point cloud files"
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str,
        required=True,
        help="Directory to save processed files"
    )
    
    parser.add_argument(
        "--extensions",
        nargs='+',
        default=['.txt', '.csv'],
        help="File extensions to process (default: .txt .csv)"
    )
    
    parser.add_argument(
        "--verbose",
        action='store_true',
        help="Show verbose output"
    )
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist")
        sys.exit(1)
    
    if not os.path.isdir(args.input_dir):
        print(f"Error: '{args.input_dir}' is not a directory")
        sys.exit(1)
    
    print("Point Cloud Street Sign Labeling Processor")
    print("=" * 50)
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"File extensions: {args.extensions}")
    print()
    
    # Process the directory
    stats = process_directory(args.input_dir, args.output_dir, args.extensions)
    
    if stats is None:
        sys.exit(1)
    
    # Print summary
    print("\nProcessing Summary:")
    print("=" * 30)
    print(f"Files processed successfully: {stats['files_processed']}")
    print(f"Files failed: {stats['files_failed']}")
    print(f"Total points processed: {stats['total_points']:,}")
    print(f"Total street signs found: {stats['total_street_signs']:,}")
    
    if stats['failed_files']:
        print(f"\nFailed files:")
        for failed in stats['failed_files']:
            print(f"  - {failed['file']}: {failed['error']}")
    
    if args.verbose and stats['files_processed'] > 0:
        print(f"\nOutput files saved to: {args.output_dir}")
        print("All processed files have suffix: '_street_sign_labeling.txt'")
    
    print("\n✓ Processing completed!")


if __name__ == "__main__":
    main()