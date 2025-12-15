#!/usr/bin/env python3
"""
Unified script to process basketball videos:
1. Extract frames from video
2. Analyze frames with basketball analyzer
3. Generate output video from analyzed frames

This script integrates extract_frames.py, basketball_analyzer.py, and generate_video.py
"""

import cv2
import argparse
from pathlib import Path
import re
import sys
import os

# Import basketball analyzer
from basketball_analyzer import BasketballAnalyzer


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
    
    Returns:
        Tuple (fps, total_frames, saved_count) or None if failed
    """
    print("\n" + "="*60)
    print("STEP 1: EXTRACTING FRAMES FROM VIDEO")
    print("="*60)
    
    # Open video file
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return None
    
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
    
    print(f"\nExtracting frames {start_frame} to {end_frame-1} (every {frame_interval} frame(s))...")
    print(f"Output directory: {output_path}")
    
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
                print(f"  Saved {saved_count} frames... (current: frame {current_frame})")
        
        frame_count += 1
        current_frame += 1
    
    cap.release()
    
    print(f"\nExtraction complete!")
    print(f"  Total frames processed: {frame_count}")
    print(f"  Frames saved: {saved_count}")
    print(f"  Output directory: {output_path}")
    
    return fps, total_frames, saved_count


def natural_sort_key(filename):
    """Extract frame number for natural sorting"""
    match = re.search(r'(\d+)', filename)
    return int(match.group(1)) if match else 0


def analyze_frames(images_dir, output_dir, analyzer, fps=30.0):
    """
    Analyze frames using basketball analyzer.
    
    Args:
        images_dir: Directory containing input frame images
        output_dir: Directory to save analyzed frames
        analyzer: BasketballAnalyzer instance
        fps: Frames per second (for timestamp calculation)
    
    Returns:
        Number of frames processed
    """
    print("\n" + "="*60)
    print("STEP 2: ANALYZING FRAMES")
    print("="*60)
    
    images_dir_path = Path(images_dir)
    if not images_dir_path.exists():
        print(f"Error: Directory not found: {images_dir}")
        return 0
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(images_dir_path.glob(f'*{ext}'))
        image_files.extend(images_dir_path.glob(f'*{ext.upper()}'))
    
    image_files = sorted(image_files, key=lambda x: natural_sort_key(x.name))
    
    if not image_files:
        print(f"No image files found in {images_dir}")
        return 0
    
    print(f"Processing {len(image_files)} images from {images_dir}")
    
    # Create output directory
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    
    # Process each image with frame numbers for ball tracking
    for i, image_file in enumerate(image_files):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(image_files)} images...")
        
        output_path = output_dir_path / f"detected_{image_file.name}"
        
        # Extract frame number from filename (e.g., frame_000844.jpg -> 844)
        frame_match = re.search(r'frame[_\s]*(\d+)', image_file.name, re.IGNORECASE)
        if frame_match:
            frame_number = int(frame_match.group(1))
        else:
            # Fallback to enumerate index if no frame number found in filename
            frame_number = i
            if i == 0:
                print(f"  Warning: Could not extract frame number from filenames, using index")
        
        # Process image with frame number for continuous ball tracking
        analyzer.process_image(str(image_file), str(output_path), frame_number=frame_number)
    
    print(f"\nAnalysis complete!")
    print(f"  Total frames processed: {len(image_files)}")
    print(f"  Output directory: {output_dir_path}")
    print(f"  Team A Goals: {analyzer.team_goals['Team A']}")
    print(f"  Team B Goals: {analyzer.team_goals['Team B']}")
    print(f"  Total Goals: {sum(analyzer.team_goals.values())}")
    
    return len(image_files)


def generate_video(input_dir, output_path, fps=30.0):
    """
    Generate video from analyzed frames.
    
    Args:
        input_dir: Directory containing analyzed frame images
        output_path: Output video file path
        fps: Frames per second for output video
    
    Returns:
        True if successful, False otherwise
    """
    print("\n" + "="*60)
    print("STEP 3: GENERATING OUTPUT VIDEO")
    print("="*60)
    
    input_dir_path = Path(input_dir)
    if not input_dir_path.exists():
        print(f"Error: Directory {input_dir} does not exist")
        return False
    
    # Get all frame files sorted by frame number
    frame_files = []
    for file_path in input_dir_path.glob("*.jpg"):
        frame_files.append(file_path)
    
    if len(frame_files) == 0:
        print(f"Error: No frame files found in {input_dir}")
        return False
    
    # Sort by frame number
    frame_files.sort(key=lambda x: natural_sort_key(x.name))
    
    print(f"Found {len(frame_files)} frames")
    
    # Get frame dimensions from first frame
    first_frame = cv2.imread(str(frame_files[0]))
    if first_frame is None:
        print(f"Error: Could not read first frame: {frame_files[0]}")
        return False
    
    height, width = first_frame.shape[:2]
    print(f"Frame size: {width}x{height}")
    print(f"Output FPS: {fps}")
    
    # Create output directory if it doesn't exist
    output_path_obj = Path(output_path)
    output_dir = output_path_obj.parent
    if output_dir and not output_dir.exists():
        print(f"Creating output directory: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define codec and create VideoWriter
    # Try different codecs if mp4v fails
    codecs_to_try = [
        ('mp4v', cv2.VideoWriter_fourcc(*'mp4v')),
        ('XVID', cv2.VideoWriter_fourcc(*'XVID')),
        ('MJPG', cv2.VideoWriter_fourcc(*'MJPG')),
    ]
    
    out = None
    codec_name = None
    for name, fourcc in codecs_to_try:
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        if out.isOpened():
            codec_name = name
            print(f"Using codec: {codec_name}")
            break
        else:
            out.release()
            out = None
    
    if out is None or not out.isOpened():
        print(f"Error: Could not open video writer for {output_path}")
        print(f"Tried codecs: {[c[0] for c in codecs_to_try]}")
        return False
    
    print(f"\nGenerating video: {output_path}")
    
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
            print(f"  Processed {i + 1}/{len(frame_files)} frames...")
    
    # Release everything
    out.release()
    print(f"\nVideo saved successfully: {output_path}")
    
    return True


def generate_goal_clips(goal_history, analyzed_frames_dir, output_dir, fps, clip_duration_before=10.0, clip_duration_after=5.0):
    """
    Generate individual video clips for each goal event.
    
    Args:
        goal_history: List of goal events with 'frame', 'team', 'time' keys
        analyzed_frames_dir: Directory containing analyzed frame images
        output_dir: Directory to save goal clips
        fps: Frames per second
        clip_duration_before: Seconds before goal to include (default: 10.0)
        clip_duration_after: Seconds after goal to include (default: 5.0)
    
    Returns:
        List of created clip file paths
    """
    if not goal_history:
        print("No goals detected, skipping goal clip generation")
        return []
    
    print("\n" + "="*60)
    print("STEP 4: GENERATING GOAL CLIPS")
    print("="*60)
    
    analyzed_frames_path = Path(analyzed_frames_dir)
    if not analyzed_frames_path.exists():
        print(f"Error: Analyzed frames directory not found: {analyzed_frames_dir}")
        return []
    
    # Get all analyzed frame files sorted by frame number
    frame_files = []
    for file_path in analyzed_frames_path.glob("*.jpg"):
        frame_files.append(file_path)
    
    if len(frame_files) == 0:
        print(f"Error: No frame files found in {analyzed_frames_dir}")
        return []
    
    # Sort by frame number
    frame_files.sort(key=lambda x: natural_sort_key(x.name))
    
    # Create a mapping of frame number to file path
    frame_to_file = {}
    for frame_file in frame_files:
        # Extract frame number from filename (e.g., detected_frame_000844.jpg -> 844)
        frame_match = re.search(r'frame[_\s]*(\d+)', frame_file.name, re.IGNORECASE)
        if frame_match:
            frame_num = int(frame_match.group(1))
            frame_to_file[frame_num] = frame_file
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Track goal counts per team
    team_goal_counts = {'Team A': 0, 'Team B': 0}
    
    # Get frame dimensions from first frame
    first_frame = cv2.imread(str(frame_files[0]))
    if first_frame is None:
        print(f"Error: Could not read first frame")
        return []
    
    height, width = first_frame.shape[:2]
    
    # Define codec
    codecs_to_try = [
        ('mp4v', cv2.VideoWriter_fourcc(*'mp4v')),
        ('XVID', cv2.VideoWriter_fourcc(*'XVID')),
        ('MJPG', cv2.VideoWriter_fourcc(*'MJPG')),
    ]
    
    created_clips = []
    
    print(f"Generating {len(goal_history)} goal clips...")
    
    for goal in goal_history:
        goal_frame = goal['frame']
        goal_team = goal['team']
        
        # Increment goal count for this team
        team_goal_counts[goal_team] += 1
        goal_number = team_goal_counts[goal_team]
        
        # Create clip filename (normalize team name: "Team A" -> "team_a", "Team B" -> "team_b")
        team_name_normalized = goal_team.lower().replace(' ', '_')
        clip_filename = f"{team_name_normalized}_goal_{goal_number}.mp4"
        clip_path = output_path / clip_filename
        
        # Calculate frame range for clip
        frames_before = int(clip_duration_before * fps)
        frames_after = int(clip_duration_after * fps)
        start_frame = max(0, goal_frame - frames_before)
        end_frame = goal_frame + frames_after
        
        # Collect frames for this clip
        clip_frames = []
        for frame_num in range(start_frame, end_frame + 1):
            if frame_num in frame_to_file:
                clip_frames.append((frame_num, frame_to_file[frame_num]))
        
        if not clip_frames:
            print(f"  Warning: No frames found for goal at frame {goal_frame}, skipping...")
            continue
        
        # Sort frames by frame number
        clip_frames.sort(key=lambda x: x[0])
        
        # Create video writer
        out = None
        codec_name = None
        for name, fourcc in codecs_to_try:
            out = cv2.VideoWriter(str(clip_path), fourcc, fps, (width, height))
            if out.isOpened():
                codec_name = name
                break
            else:
                out.release()
                out = None
        
        if out is None or not out.isOpened():
            print(f"  Error: Could not create video writer for {clip_path}")
            continue
        
        # Write frames to clip
        for frame_num, frame_file in clip_frames:
            frame = cv2.imread(str(frame_file))
            if frame is None:
                continue
            
            # Resize if needed
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height))
            
            out.write(frame)
        
        out.release()
        created_clips.append(clip_path)
        print(f"  ✓ Created: {clip_filename} (frames {start_frame}-{end_frame}, goal at frame {goal_frame})")
    
    print(f"\nGoal clips generation complete!")
    print(f"  Total clips created: {len(created_clips)}")
    
    return created_clips


def main():
    parser = argparse.ArgumentParser(
        description='Complete basketball video processing pipeline: extract frames -> analyze -> generate video',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process entire video with default settings
  python process_basketball_video.py --video input.mp4 --output output.mp4

  # Process with custom frame extraction interval
  python process_basketball_video.py --video input.mp4 --output output.mp4 --interval 2

  # Process specific frame range
  python process_basketball_video.py --video input.mp4 --output output.mp4 --start 1000 --end 2000

  # Keep intermediate files
  python process_basketball_video.py --video input.mp4 --output output.mp4 --keep-intermediate
        """
    )
    
    # Input/Output arguments
    parser.add_argument('--video', type=str, required=True,
                       help='Path to input video file')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to output video file (optional, goal clips will be generated automatically)')
    
    # Frame extraction arguments
    parser.add_argument('--interval', type=int, default=1,
                       help='Extract every Nth frame (default: 1, extract all frames)')
    parser.add_argument('--start', type=int, default=0,
                       help='Frame number to start extraction from (default: 0)')
    parser.add_argument('--end', type=int, default=None,
                       help='Frame number to stop extraction at (default: end of video)')
    parser.add_argument('--format', type=str, default='jpg',
                       choices=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
                       help='Image format for extracted frames (default: jpg)')
    parser.add_argument('--prefix', type=str, default='frame',
                       help='Prefix for extracted frame filenames (default: frame)')
    
    # Directory arguments
    parser.add_argument('--images-dir', type=str, default='images',
                       help='Directory to save extracted frames (default: images)')
    parser.add_argument('--output-images-dir', type=str, default='output_images',
                       help='Directory to save analyzed frames (default: output_images)')
    
    # Analysis arguments (from basketball_analyzer.py)
    parser.add_argument('--ball_rim_model', type=str, default=None,
                       help='Path to ball/rim detection model (.pt)')
    parser.add_argument('--shot_model', type=str, default=None,
                       help='Path to shot detection model (.pt)')
    parser.add_argument('--player_model', type=str, default=None,
                       help='Path to player detection model (.pt)')
    parser.add_argument('--confidence', type=float, default=0.3,
                       help='Confidence threshold for detections (default: 0.3)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use: "cuda", "cpu", or None for auto-detect (default: None)')
    parser.add_argument('--use-yolov8n-ball', action='store_true',
                       help='Use yolov8n (COCO class 32) for ball detection instead of custom model')
    parser.add_argument('--use-ball-detect-module', action='store_true', default=True,
                       help='Use BasketballColorDetector from ball_detect.py for ball detection (default: True)')
    parser.add_argument('--ball-model', type=str, default='model/best.pt',
                       help='Path to ball detection model for ball_detect module (default: model/best.pt)')
    
    # Cleanup arguments
    parser.add_argument('--keep-intermediate', action='store_true',
                       help='Keep intermediate frame directories (images/ and output_images/) after processing')
    
    args = parser.parse_args()
    
    # Validate video file exists
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: Video file not found: {args.video}")
        return 1
    
    # Validate interval
    if args.interval < 1:
        print("Error: --interval must be >= 1")
        return 1
    
    # Validate start/end
    if args.end is not None and args.start >= args.end:
        print("Error: --start must be < --end")
        return 1
    
    # Cleanup function to remove intermediate directories
    def cleanup_intermediate_dirs():
        """Remove intermediate directories if cleanup is enabled"""
        if not args.keep_intermediate:
            print("\n" + "="*60)
            print("CLEANUP: Removing intermediate directories")
            print("="*60)
            
            import shutil
            
            if Path(args.images_dir).exists():
                print(f"  Removing {args.images_dir}/")
                try:
                    shutil.rmtree(args.images_dir)
                    print(f"    ✓ Deleted {args.images_dir}/")
                except Exception as e:
                    print(f"    ✗ Error deleting {args.images_dir}/: {e}")
            
            if Path(args.output_images_dir).exists():
                print(f"  Removing {args.output_images_dir}/")
                try:
                    shutil.rmtree(args.output_images_dir)
                    print(f"    ✓ Deleted {args.output_images_dir}/")
                except Exception as e:
                    print(f"    ✗ Error deleting {args.output_images_dir}/: {e}")
            
            print("Cleanup complete!")
    
    try:
        # STEP 1: Extract frames from video
        result = extract_frames(
            video_path=video_path,
            output_dir=args.images_dir,
            frame_interval=args.interval,
            start_frame=args.start,
            end_frame=args.end,
            image_format=args.format,
            prefix=args.prefix
        )
        
        if result is None:
            print("Error: Frame extraction failed")
            cleanup_intermediate_dirs()
            return 1
        
        fps, total_frames, saved_count = result
        
        # STEP 2: Initialize analyzer and analyze frames
        analyzer = BasketballAnalyzer(
            ball_rim_model_path=args.ball_rim_model,
            shot_model_path=args.shot_model,
            player_model_path=args.player_model,
            confidence_threshold=args.confidence,
            device=args.device,
            use_yolov8n_for_ball=args.use_yolov8n_ball,
            use_ball_detect_module=args.use_ball_detect_module,
            ball_model_path=args.ball_model
        )
        
        frames_processed = analyze_frames(
            images_dir=args.images_dir,
            output_dir=args.output_images_dir,
            analyzer=analyzer,
            fps=fps
        )
        
        if frames_processed == 0:
            print("Error: No frames were analyzed")
            cleanup_intermediate_dirs()
            return 1
        
        # STEP 3: Generate output video (optional, only if output path is provided)
        if args.output:
            success = generate_video(
                input_dir=args.output_images_dir,
                output_path=args.output,
                fps=fps
            )
            
            if not success:
                print("Error: Video generation failed")
                cleanup_intermediate_dirs()
                return 1
        
        # STEP 4: Generate goal clips
        # Determine output directory for goal clips (use same directory as main output, or create 'goal_clips' subdirectory)
        if args.output:
            output_path_obj = Path(args.output)
            goal_clips_dir = output_path_obj.parent / "goal_clips"
        else:
            # If no output specified, create goal_clips in current directory
            goal_clips_dir = Path("goal_clips")
        
        goal_clips = generate_goal_clips(
            goal_history=analyzer.goal_history,
            analyzed_frames_dir=args.output_images_dir,
            output_dir=str(goal_clips_dir),
            fps=fps,
            clip_duration_before=10.0,
            clip_duration_after=5.0
        )
        
        # Cleanup intermediate files (delete by default, unless --keep-intermediate is used)
        cleanup_intermediate_dirs()
        
        # Final summary
        print("\n" + "="*60)
        print("PROCESSING COMPLETE!")
        print("="*60)
        print(f"Input video: {args.video}")
        if args.output:
            print(f"Output video: {args.output}")
        print(f"Frames extracted: {saved_count}")
        print(f"Frames analyzed: {frames_processed}")
        print(f"Team A Goals: {analyzer.team_goals['Team A']}")
        print(f"Team B Goals: {analyzer.team_goals['Team B']}")
        print(f"Total Goals: {sum(analyzer.team_goals.values())}")
        print("\nGoal Events:")
        for goal in analyzer.goal_history:
            print(f"  Frame {goal['frame']} ({goal['time']:.2f}s): {goal['team']}")
        if goal_clips:
            print(f"\nGoal Clips Created ({len(goal_clips)}):")
            for clip_path in goal_clips:
                print(f"  - {clip_path.name}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\nProcessing interrupted by user")
        cleanup_intermediate_dirs()
        return 1
    except Exception as e:
        print(f"\nError during processing: {e}")
        import traceback
        traceback.print_exc()
        cleanup_intermediate_dirs()
        return 1


if __name__ == '__main__':
    sys.exit(main())

