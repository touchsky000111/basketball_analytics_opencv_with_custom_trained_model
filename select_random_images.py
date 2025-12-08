#!/usr/bin/env python3
"""
Script to randomly select a specified number of images from a source folder
and copy them to a destination folder.
"""

import argparse
import random
import shutil
from pathlib import Path


def select_random_images(source_dir, dest_dir, num_images, seed=None):
    """
    Randomly select images from source directory and copy them to destination directory.
    
    Args:
        source_dir: Path to source directory containing images
        dest_dir: Path to destination directory to save selected images
        num_images: Number of images to randomly select
        seed: Random seed for reproducibility (optional)
    """
    # Set random seed if provided
    if seed is not None:
        random.seed(seed)
    
    # Convert to Path objects
    source_path = Path(source_dir)
    dest_path = Path(dest_dir)
    
    # Validate source directory
    if not source_path.exists():
        print(f"Error: Source directory not found: {source_dir}")
        return
    
    if not source_path.is_dir():
        print(f"Error: Source path is not a directory: {source_dir}")
        return
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(source_path.glob(f'*{ext}'))
        image_files.extend(source_path.glob(f'*{ext.upper()}'))
    
    # Remove duplicates and sort
    image_files = sorted(list(set(image_files)))
    
    if not image_files:
        print(f"Error: No image files found in {source_dir}")
        return
    
    total_images = len(image_files)
    print(f"Found {total_images} images in {source_dir}")
    
    # Check if we have enough images
    if num_images > total_images:
        print(f"Warning: Requested {num_images} images, but only {total_images} available.")
        print(f"Will copy all {total_images} images.")
        num_images = total_images
    
    # Randomly select images
    selected_images = random.sample(image_files, num_images)
    
    # Create destination directory
    dest_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Copying {len(selected_images)} images to {dest_dir}...")
    
    # Copy selected images
    copied_count = 0
    for i, image_file in enumerate(selected_images, 1):
        dest_file = dest_path / image_file.name
        
        # Handle filename conflicts (if same filename exists)
        if dest_file.exists():
            # Add a suffix to avoid overwriting
            stem = image_file.stem
            suffix = image_file.suffix
            counter = 1
            while dest_file.exists():
                dest_file = dest_path / f"{stem}_{counter}{suffix}"
                counter += 1
        
        shutil.copy2(image_file, dest_file)
        copied_count += 1
        
        if copied_count % 100 == 0:
            print(f"  Copied {copied_count}/{len(selected_images)} images...")
    
    print()
    print(f"Successfully copied {copied_count} images to {dest_dir}")
    print(f"  Source: {source_dir}")
    print(f"  Destination: {dest_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Randomly select images from a source folder and copy them to a destination folder',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Select 800 random images from images folder to db_folder
  python select_random_images.py --source images/ --dest db_folder/ --num 800

  # Select 100 images with a specific random seed (for reproducibility)
  python select_random_images.py --source images/ --dest db_folder/ --num 100 --seed 42
        """
    )
    
    parser.add_argument('--source', type=str, required=True,
                       help='Source directory containing images')
    parser.add_argument('--dest', type=str, required=True,
                       help='Destination directory to save selected images')
    parser.add_argument('--num', type=int, required=True,
                       help='Number of images to randomly select')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility (optional)')
    
    args = parser.parse_args()
    
    # Validate number of images
    if args.num < 1:
        print("Error: --num must be >= 1")
        return
    
    # Select and copy images
    select_random_images(
        source_dir=args.source,
        dest_dir=args.dest,
        num_images=args.num,
        seed=args.seed
    )


if __name__ == '__main__':
    main()

