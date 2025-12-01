#!/usr/bin/env python3
"""
Ball_RimV5 is the best RIM detect. 

Script to extract video frames to images folder and detect ball and rim using ball_rimV5.pt model
Can also use already extracted images from images folder
"""

import cv2
import os
from pathlib import Path
from ultralytics import YOLO
import argparse
from collections import defaultdict


def load_existing_images(images_dir):
    """
    Load existing images from a directory
    
    Args:
        images_dir: Directory containing images
    
    Returns:
        List of image file paths
    """
    images_dir = Path(images_dir)
    
    if not images_dir.exists():
        return []
    
    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    
    # Get all image files
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(images_dir.glob(f'*{ext}'))
        image_paths.extend(images_dir.glob(f'*{ext.upper()}'))
    
    # Sort by filename for consistent processing
    image_paths = sorted(image_paths)
    
    return image_paths


def extract_frames(video_path, output_dir, frame_interval=1):
    """
    Extract frames from video to images folder
    
    Args:
        video_path: Path to input video file
        output_dir: Directory to save extracted frames
        frame_interval: Extract every Nth frame (1 = all frames)
    
    Returns:
        List of extracted image file paths
    """
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {video_path}")
    print(f"FPS: {fps}, Total frames: {total_frames}")
    print(f"Extracting frames to: {output_dir}")
    print(f"Frame interval: {frame_interval} (extracting every {frame_interval} frame(s))")
    print()
    
    frame_count = 0
    saved_count = 0
    image_paths = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Extract frame based on interval
        if frame_count % frame_interval == 0:
            # Save frame as image
            image_filename = f"frame_{frame_count:06d}.jpg"
            image_path = output_dir / image_filename
            cv2.imwrite(str(image_path), frame)
            image_paths.append(image_path)
            saved_count += 1
            
            if saved_count % 100 == 0:
                print(f"Extracted {saved_count} frames...")
        
        frame_count += 1
    
    cap.release()
    
    print(f"\nExtraction complete! Saved {saved_count} frames to {output_dir}")
    return image_paths


def detect_ball_rim_in_images(image_paths, model_path=None, confidence_threshold=0.01, save_annotated=True, annotated_dir=None):
    """
    Detect ball and rim in all images using best.pt model
    New model best.pt detects: 0=rim_a, 1=basketball, 2=rim_b
    
    Args:
        image_paths: List of image file paths
        model_path: Path to ball_rim model (None = use default best.pt)
        confidence_threshold: Minimum confidence for detections
        save_annotated: Whether to save annotated images with detections
        annotated_dir: Directory to save annotated images (default: output_image)
    
    Returns:
        Dictionary with detection results
    """
    # Load ball_rim model - ONLY use best.pt
    if model_path is None:
        # ONLY use best.pt - no fallbacks
        if Path("model/best.pt").exists():
            model_path = "model/best.pt"
        else:
            raise ValueError("best.pt model not found at model/best.pt. Please ensure the model exists.")
    
    print(f"\nLoading ball_rim model from: {model_path}")
    model = YOLO(model_path)
    print("Model loaded successfully!")
    
    # Check if using best.pt (new model with rim_a, basketball, rim_b)
    is_best_pt = "best.pt" in str(model_path)
    
    # Print model classes
    if hasattr(model, 'names'):
        print(f"Model classes: {model.names}")
    else:
        # Default classes for best.pt: 0=rim_a, 1=basketball, 2=rim_b
        if is_best_pt:
            print("Model classes: 0=rim_a, 1=basketball, 2=rim_b")
        else:
            print("Model classes: 0=ball, 1=rim")
    print()
    
    # Create annotated directory if needed (default: output_image)
    if save_annotated:
        if annotated_dir:
            annotated_dir = Path(annotated_dir)
        else:
            annotated_dir = Path("output_image")
        annotated_dir.mkdir(parents=True, exist_ok=True)
    
    # Process all images
    total_detections = 0
    images_with_detections = 0
    detection_results = []
    ball_count = 0
    rim_count = 0
    
    print(f"Processing {len(image_paths)} images...")
    print(f"Confidence threshold: {confidence_threshold}")
    # Determine which classes to detect based on model
    if is_best_pt:
        print(f"Detecting: rim_a (class 0), basketball (class 1), rim_b (class 2)")
    else:
        print(f"Detecting: ball (class 0) and rim (class 1)")
    print()
    
    for idx, image_path in enumerate(image_paths):
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Warning: Could not read image {image_path}")
            continue
        
        # Run YOLOv8 inference on ball and rim classes
        # New model best.pt: 0=rim_a, 1=basketball, 2=rim_b
        # Old models: 0=ball, 1=rim
        if is_best_pt:
            # Use new model with rim_a, basketball, rim_b
            results = model.predict(
                image, 
                classes=[0, 1, 2],  # 0=rim_a, 1=basketball, 2=rim_b
                conf=confidence_threshold,
                verbose=False
            )
        else:
            # Use old model with ball and rim
            results = model.predict(
                image, 
                classes=[0, 1],  # 0=ball, 1=rim
                conf=confidence_threshold,
                verbose=False
            )
        
        # Count detections
        num_detections = len(results[0].boxes)
        
        # Create annotated image
        annotated_image = image.copy()
        
        if num_detections > 0:
            images_with_detections += 1
            total_detections += num_detections
            
            # Get detection details with class names
            boxes = results[0].boxes
            names = results[0].names  # Class name dictionary: {0: 'ball', 1: 'rim'}
            detections = []
            
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = names[cls_id]
                
                # Count by type
                if cls_id == 1:  # basketball
                    ball_count += 1
                elif cls_id == 0 or cls_id == 2:  # rim_a or rim_b
                    rim_count += 1
                
                detections.append({
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'confidence': conf,
                    'class_id': cls_id,
                    'class_name': cls_name
                })
                
                # Draw rectangle with label and confidence
                # Choose color: red for basketball, green for rim_a, yellow for rim_b
                if cls_id == 1:  # basketball
                    color = (0, 0, 255)  # Red for basketball (BGR format)
                elif cls_id == 0:  # rim_a
                    color = (0, 255, 0)  # Green for rim_a
                else:  # cls_id == 2 (rim_b)
                    color = (0, 255, 255)  # Yellow for rim_b
                
                # Draw bounding box
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
                
                # Prepare label text with class name and confidence
                label = f"{cls_name} {conf:.2f}"
                
                # Get text size for background rectangle
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                
                # Draw background rectangle for text
                cv2.rectangle(
                    annotated_image,
                    (x1, y1 - text_height - 10),
                    (x1 + text_width, y1),
                    color,
                    -1
                )
                
                # Draw text
                cv2.putText(
                    annotated_image,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),  # White text
                    2
                )
            
            detection_results.append({
                'image': str(image_path),
                'frame_number': idx,
                'num_detections': num_detections,
                'detections': detections
            })
        
        # Save annotated image
        if save_annotated and annotated_dir:
            annotated_filename = Path(image_path).name
            annotated_path = annotated_dir / annotated_filename
            cv2.imwrite(str(annotated_path), annotated_image)
        
        # Progress update
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{len(image_paths)} images... "
                  f"(Found detections in {images_with_detections} images, {total_detections} total detections)")
    
    print(f"\nDetection complete!")
    print(f"Total images processed: {len(image_paths)}")
    print(f"Images with detections: {images_with_detections}")
    print(f"Total detections: {total_detections}")
    print(f"  - Balls detected: {ball_count}")
    print(f"  - Rims detected: {rim_count}")
    
    if save_annotated and annotated_dir:
        print(f"\nAnnotated images saved to: {annotated_dir}")
    
    return {
        'total_images': len(image_paths),
        'images_with_detections': images_with_detections,
        'total_detections': total_detections,
        'ball_count': ball_count,
        'rim_count': rim_count,
        'detection_results': detection_results
    }


def main():
    parser = argparse.ArgumentParser(description='Detect ball and rim in images using best.pt model (uses existing images or extracts from video)')
    parser.add_argument('--images-dir', type=str, default='images',
                       help='Directory containing images to process (default: images)')
    parser.add_argument('--video', type=str, default=None,
                       help='Path to input video file (optional, only used if images not found)')
    parser.add_argument('--output', type=str, default='images',
                       help='Directory to save extracted frames if extracting from video (default: images)')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to ball_rim model (default: model/best.pt, falls back to ball_rimV5.pt)')
    parser.add_argument('--confidence', type=float, default=0.1,
                       help='Confidence threshold for detections (default: 0.2)')
    parser.add_argument('--frame-interval', type=int, default=1,
                       help='Extract every Nth frame if extracting from video (default: 1 = all frames)')
    parser.add_argument('--no-annotated', action='store_true',
                       help='Do not save annotated images with detections')
    parser.add_argument('--annotated-dir', type=str, default='output_image',
                       help='Directory to save annotated images (default: output_image)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Ball and Rim Detection in Images using best.pt")
    print("="*60)
    print()
    
    # Step 1: Check for existing images or extract from video
    print("Step 1: Loading images...")
    print("-" * 60)
    
    # Try to load existing images first
    image_paths = load_existing_images(args.images_dir)
    
    if image_paths:
        print(f"Found {len(image_paths)} existing images in '{args.images_dir}' folder")
        print("Using existing images (skipping video extraction)")
    else:
        # No existing images, try to extract from video
        if args.video and Path(args.video).exists():
            print(f"No images found in '{args.images_dir}', extracting from video...")
            image_paths = extract_frames(
                video_path=args.video,
                output_dir=args.output,
                frame_interval=args.frame_interval
            )
        elif args.video:
            print(f"Error: Video file not found: {args.video}")
            print(f"Error: No images found in '{args.images_dir}' folder")
            return
        else:
            print(f"Error: No images found in '{args.images_dir}' folder")
            print("Please provide --video argument to extract frames from video")
            return
    
    if not image_paths:
        print("Error: No images to process!")
        return
    
    # Step 2: Detect ball and rim in all images
    print("\nStep 2: Detecting ball and rim in images...")
    print("-" * 60)
    results = detect_ball_rim_in_images(
        image_paths=image_paths,
        model_path=args.model,
        confidence_threshold=args.confidence,
        save_annotated=not args.no_annotated,
        annotated_dir=args.annotated_dir
    )
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total images processed: {results['total_images']}")
    print(f"Images with detections: {results['images_with_detections']}")
    print(f"Total detections: {results['total_detections']}")
    print(f"  - Balls detected: {results['ball_count']}")
    print(f"  - Rims detected: {results['rim_count']}")
    if results['total_images'] > 0:
        print(f"Detection rate: {results['images_with_detections']/results['total_images']*100:.2f}%")
    print("="*60)


if __name__ == "__main__":
    main()

