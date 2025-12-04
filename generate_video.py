"""
Generate video from frames in output_images_old directory
"""

import cv2
import os
import argparse
from pathlib import Path
import re


def natural_sort_key(filename):
    """Extract frame number for natural sorting"""
    match = re.search(r'(\d+)', filename)
    return int(match.group(1)) if match else 0


def get_frame_files(directory):
    """Get all frame files sorted by frame number"""
    frame_files = []
    directory_path = Path(directory)
    
    if not directory_path.exists():
        raise ValueError(f"Directory {directory} does not exist")
    
    # Get all jpg files
    for file_path in directory_path.glob("*.jpg"):
        frame_files.append(file_path)
    
    # Sort by frame number
    frame_files.sort(key=lambda x: natural_sort_key(x.name))
    
    return frame_files


def get_frame_size(frame_path):
    """Get frame dimensions from first frame"""
    img = cv2.imread(str(frame_path))
    if img is None:
        raise ValueError(f"Could not read frame: {frame_path}")
    height, width = img.shape[:2]
    return width, height


def generate_video(input_dir, output_path, fps=30.0):
    """
    Generate video from frames in input directory
    
    Args:
        input_dir: Directory containing frame images
        output_path: Output video file path
        fps: Frames per second for output video
    """
    print(f"Reading frames from: {input_dir}")
    frame_files = get_frame_files(input_dir)
    
    if len(frame_files) == 0:
        raise ValueError(f"No frame files found in {input_dir}")
    
    print(f"Found {len(frame_files)} frames")
    
    # Get frame dimensions from first frame
    first_frame = cv2.imread(str(frame_files[0]))
    if first_frame is None:
        raise ValueError(f"Could not read first frame: {frame_files[0]}")
    
    height, width = first_frame.shape[:2]
    print(f"Frame size: {width}x{height}")
    
    # Define codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        raise ValueError(f"Could not open video writer for {output_path}")
    
    print(f"Generating video: {output_path} at {fps} FPS")
    
    # Write frames to video
    for i, frame_path in enumerate(frame_files):
        frame = cv2.imread(str(frame_path))
        if frame is None:
            print(f"Warning: Could not read frame {frame_path}, skipping...")
            continue
        
        # Resize frame if dimensions don't match (shouldn't happen, but just in case)
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height))
        
        out.write(frame)
        
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(frame_files)} frames...")
    
    # Release everything
    out.release()
    print(f"Video saved successfully: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate video from frames')
    parser.add_argument(
        '--input_dir',
        type=str,
        default='output_images_old',
        help='Input directory containing frame images (default: output_images_old)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='output_video.mp4',
        help='Output video file path (default: output_video.mp4)'
    )
    parser.add_argument(
        '--fps',
        type=float,
        default=30.0,
        help='Frames per second for output video (default: 30.0)'
    )
    
    args = parser.parse_args()
    
    try:
        generate_video(args.input_dir, args.output, args.fps)
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())

