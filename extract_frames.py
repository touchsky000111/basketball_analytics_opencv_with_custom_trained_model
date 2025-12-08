#!/usr/bin/env python3
"""
Script to extract frames from a video file and save them as images.
"""

import cv2
import argparse
from pathlib import Path


def extract_frames(video_path, output_dir, frame_interval=1, start_frame=0, end_frame=None, 
                   image_format='jpg', prefix='frame'):
    """
    Extract frames from a video file and save them as images.
    
    Args:
        video_path: Path to input video file
        output_dir: Directory to save extracted frames
        frame_interval: Extract every Nth frame (1 = all frames, 2 = every other frame, etc.)
        start_frame: Frame number to start extraction from (0-indexed)
        end_frame: Frame number to stop extraction at (None = extract to end)
        image_format: Image format ('jpg', 'png', 'bmp', etc.)
        prefix: Prefix for output filenames (default: 'frame')
    """
    # Open video file
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video properties:")
    print(f"  Total frames: {total_frames}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Resolution: {width}x{height}")
    print(f"  Duration: {total_frames/fps:.2f} seconds")
    print()
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Set end_frame if not specified
    if end_frame is None:
        end_frame = total_frames
    
    # Ensure end_frame doesn't exceed total frames
    end_frame = min(end_frame, total_frames)
    
    # Calculate number of digits needed for frame numbering
    max_frame_num = end_frame - 1
    num_digits = len(str(max_frame_num))
    
    # Seek to start frame
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frame_count = 0
    saved_count = 0
    current_frame = start_frame
    
    print(f"Extracting frames {start_frame} to {end_frame-1} (every {frame_interval} frame(s))...")
    print(f"Output directory: {output_path}")
    print()
    
    while current_frame < end_frame:
        ret, frame = cap.read()
        
        if not ret:
            print(f"Warning: Could not read frame {current_frame}")
            break
        
        # Save frame if it matches the interval
        if frame_count % frame_interval == 0:
            # Format frame number with leading zeros
            frame_num_str = str(current_frame).zfill(num_digits)
            filename = f"{prefix}_{frame_num_str}.{image_format}"
            filepath = output_path / filename
            
            cv2.imwrite(str(filepath), frame)
            saved_count += 1
            
            if saved_count % 100 == 0:
                print(f"Saved {saved_count} frames... (current: frame {current_frame})")
        
        frame_count += 1
        current_frame += 1
    
    cap.release()
    
    print()
    print(f"Extraction complete!")
    print(f"  Total frames processed: {frame_count}")
    print(f"  Frames saved: {saved_count}")
    print(f"  Output directory: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Extract frames from a video file and save them as images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract all frames
  python extract_frames.py --video input.mp4 --output images/

  # Extract every 5th frame
  python extract_frames.py --video input.mp4 --output images/ --interval 5

  # Extract frames 1000 to 2000
  python extract_frames.py --video input.mp4 --output images/ --start 1000 --end 2000

  # Extract as PNG files
  python extract_frames.py --video input.mp4 --output images/ --format png
        """
    )
    
    parser.add_argument('--video', type=str, required=True,
                       help='Path to input video file')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory to save extracted frames')
    parser.add_argument('--interval', type=int, default=1,
                       help='Extract every Nth frame (default: 1, extract all frames)')
    parser.add_argument('--start', type=int, default=0,
                       help='Frame number to start extraction from (default: 0)')
    parser.add_argument('--end', type=int, default=None,
                       help='Frame number to stop extraction at (default: end of video)')
    parser.add_argument('--format', type=str, default='jpg',
                       choices=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
                       help='Image format for output files (default: jpg)')
    parser.add_argument('--prefix', type=str, default='frame',
                       help='Prefix for output filenames (default: frame)')
    
    args = parser.parse_args()
    
    # Validate video file exists
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: Video file not found: {args.video}")
        return
    
    # Validate interval
    if args.interval < 1:
        print("Error: --interval must be >= 1")
        return
    
    # Validate start/end
    if args.end is not None and args.start >= args.end:
        print("Error: --start must be < --end")
        return
    
    # Extract frames
    extract_frames(
        video_path=video_path,
        output_dir=args.output,
        frame_interval=args.interval,
        start_frame=args.start,
        end_frame=args.end,
        image_format=args.format,
        prefix=args.prefix
    )


if __name__ == '__main__':
    main()

