"""
Basketball Video Analysis Script
Uses YOLOv8 models to detect ball, players, hoops and count goals for each team
Based on sample_project/basketball-sports-ai implementation
Integrates ball detection from ball_detect.py and rim detection from rim_detect.py
"""

import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import argparse
from pathlib import Path
import os
import torch

# Import ball detection functionality
try:
    from ball_detect import BasketballColorDetector
    BALL_DETECT_AVAILABLE = True
except ImportError:
    BALL_DETECT_AVAILABLE = False
    print("Warning: ball_detect module not available. Using default ball detection.")


class BasketballAnalyzer:
    def __init__(self, ball_rim_model_path=None, shot_model_path=None, player_model_path=None, confidence_threshold=0.3, device=None, use_yolov8n_for_ball=False, use_ball_detect_module=True, ball_model_path="model/best.pt"):
        """
        Initialize the Basketball Analyzer
        
        Args:
            ball_rim_model_path: Path to ball/rim detection model (.pt). 
                                 Classes: 0=ball, 1=rim
            shot_model_path: Path to shot detection model (.pt)
            player_model_path: Path to player detection model (.pt). If None, uses best.pt for players
            confidence_threshold: Minimum confidence for detections
            device: Device to use ('cuda', 'cpu', or None for auto-detect). Default: None (auto-detect)
            use_yolov8n_for_ball: If True, use yolov8n (COCO class 32: sports ball) for ball detection instead of custom model
            use_ball_detect_module: If True, use BasketballColorDetector from ball_detect.py for ball detection
            ball_model_path: Path to ball detection model (best.pt) for use with ball_detect module
        """
        self.confidence_threshold = confidence_threshold
        self.use_yolov8n_for_ball = use_yolov8n_for_ball
        self.use_ball_detect_module = use_ball_detect_module and BALL_DETECT_AVAILABLE
        self.ball_model_path = ball_model_path
        
        # Set device (GPU/CPU)
        if device is None:
            # Auto-detect: use GPU if available, otherwise CPU
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Verify device availability
        if self.device == 'cuda' and not torch.cuda.is_available():
            print("Warning: CUDA requested but not available. Falling back to CPU.")
            self.device = 'cpu'
        
        # Display device information
        print(f"Using device: {self.device}")
        if self.device == 'cuda':
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA Version: {torch.version.cuda}")
            print(f"  Available GPUs: {torch.cuda.device_count()}")
        else:
            print("  Using CPU for inference")
        
        # Load ball/rim detection model (required) - ONLY use best.pt
        # Model best.pt detects: 0=rim_a, 1=basketball, 2=rim_b
        if ball_rim_model_path and Path(ball_rim_model_path).exists():
            print(f"Loading ball/rim model from {ball_rim_model_path}")
            self.ball_rim_model = YOLO(ball_rim_model_path)
            # YOLOv8 automatically uses GPU if available, device is set for explicit control
        else:
            # ONLY use best.pt model - no fallbacks
            best_pt_path = "model/best.pt"
            if Path(best_pt_path).exists():
                print(f"Loading ball/rim model from {best_pt_path}")
                self.ball_rim_model = YOLO(best_pt_path)
                # YOLOv8 automatically uses GPU if available
                # Print model classes for debugging
                if hasattr(self.ball_rim_model, 'names'):
                    print(f"Model classes: {self.ball_rim_model.names}")
            else:
                raise ValueError(f"best.pt model not found at {best_pt_path}. Please ensure model/best.pt exists.")
        
        # Load shot detection model (optional)
        if shot_model_path and Path(shot_model_path).exists():
            print(f"Loading shot detection model from {shot_model_path}")
            self.shot_model = YOLO(shot_model_path)
            # YOLOv8 automatically uses GPU if available
            self.use_shot_detection = True
        else:
            # Try to use from model folder first
            model_folder_paths = [
                "model/shot_detection_v2.pt",
                "model/shot_detection_v1.pt",
                "model/shot_detection.pt"
            ]
            found = False
            for model_path in model_folder_paths:
                if Path(model_path).exists():
                    print(f"Loading shot detection model from {model_path}")
                    self.shot_model = YOLO(model_path)
                    # YOLOv8 automatically uses GPU if available
                    self.use_shot_detection = True
                    found = True
                    break
            
            # Fallback to sample_project
            if not found:
                default_shot = "sample_project/basketball-sports-ai/model_pt/shot_detection_v2.pt"
                if Path(default_shot).exists():
                    print(f"Loading shot detection model from {default_shot}")
                    self.shot_model = YOLO(default_shot)
                    # YOLOv8 automatically uses GPU if available
                    self.use_shot_detection = True
                else:
                    print("Warning: Shot detection model not found. Shot detection disabled.")
                    self.shot_model = None
                    self.use_shot_detection = False
        
        # Load player detection model using best.pt
        if player_model_path and Path(player_model_path).exists():
            print(f"Loading player detection model from {player_model_path}")
            self.player_model = YOLO(player_model_path)
            # YOLOv8 automatically uses GPU if available
            self.use_player_detection = True
        else:
            # Try to use best.pt for player detection
            best_pt_paths = [
                "model/best.pt",
                "new_model/basketball_detection/weights/best.pt",
                "new_model/basketball_detection_best.pt"
            ]
            found = False
            for model_path in best_pt_paths:
                if Path(model_path).exists():
                    print(f"Loading player detection model (best.pt) from {model_path}")
                    self.player_model = YOLO(model_path)
                    # YOLOv8 automatically uses GPU if available
                    self.use_player_detection = True
                    found = True
                    break
            
            # Fallback to other models if best.pt not found
            if not found:
                model_folder_paths = [
                    "model/yolov8m.pt",
                    "model/yolov8m-pose.pt",
                    "model/yolov8n.pt",
                    "model/yolov8n-pose.pt"
                ]
                for model_path in model_folder_paths:
                    if Path(model_path).exists():
                        print(f"Loading player detection model from {model_path}")
                        self.player_model = YOLO(model_path)
                        # YOLOv8 automatically uses GPU if available
                        self.use_player_detection = True
                        found = True
                        break
                
                # Final fallback to default
                if not found:
                    print("Using default YOLOv8 for player detection")
                    self.player_model = YOLO('yolov8n.pt')
                    # YOLOv8 automatically uses GPU if available
                    self.use_player_detection = True
        
        # Load ball detection model - ONLY use best.pt (class 1 = basketball)
        self.ball_model = None
        self.ball_detector = None  # BasketballColorDetector instance
        
        if self.use_ball_detect_module:
            # Use BasketballColorDetector from ball_detect.py with best.pt
            print("Using BasketballColorDetector from ball_detect.py for ball detection")
            try:
                # Force use of best.pt for ball detection
                best_pt_path = "model/best.pt"
                self.ball_detector = BasketballColorDetector(
                    model_path=best_pt_path,
                    use_yolo=True,
                    confidence_threshold=self.confidence_threshold
                )
                if self.ball_detector.use_yolo:
                    print(f"Ball detection using best.pt model: {best_pt_path}")
                else:
                    print("Warning: Ball detection falling back to color detection method")
            except Exception as e:
                print(f"Warning: Could not initialize BasketballColorDetector: {e}")
                print("Falling back to using ball_rim_model for ball detection")
                self.use_ball_detect_module = False
        
        if not self.use_ball_detect_module:
            # Use best.pt model (ball_rim_model) for ball detection (class 1 = basketball)
            # Disable yolov8n_for_ball option - only use best.pt
            if self.use_yolov8n_for_ball:
                print("Warning: --use-yolov8n-ball is ignored. Using best.pt for ball detection.")
            # Always use the ball_rim_model (best.pt) for ball detection
            self.ball_model = self.ball_rim_model
            print("Ball detection using best.pt model (class 1 = basketball)")
        
        # Goal tracking
        self.team_goals = defaultdict(int)  # Track goals per team
        self.goal_history = []  # Store goal events with frame numbers
        self.recent_goal = None  # Store recent goal info for display: {'team': str, 'frame': int, 'display_until': int}
        
        # Rim detection - track rim_a and rim_b as separate objects
        # Model best.pt detects: 0=rim_a, 1=basketball, 2=rim_b
        self.rim_a_location = None  # xywh: [x, y, w, h] center of the box, width, height
        self.rim_a_bounding_box = None  # xyxy: [x1, y1, x2, y2]
        self.rim_a_standard_line = None  # Line across rim_a for goal detection
        self.rim_a_detected = False  # Flag to track if rim_a was detected
        self.rim_a_team = None  # Which team's rim_a: "Team A" (left) or "Team B" (right)
        
        self.rim_b_location = None  # xywh: [x, y, w, h] center of the box, width, height
        self.rim_b_bounding_box = None  # xyxy: [x1, y1, x2, y2]
        self.rim_b_standard_line = None  # Line across rim_b for goal detection
        self.rim_b_detected = False  # Flag to track if rim_b was detected
        self.rim_b_team = None  # Which team's rim_b: "Team A" (left) or "Team B" (right)
        
        self.frame_width = None  # Store frame width to determine rim position
        
        # Ball tracking
        self.ball_location = None  # xywh: [x, y, w, h]
        self.previous_ball_location = None  # xywh: [x, y, w, h]
        self.ball_tracking = []  # List of ball locations relative to rim: [(x_rel, y_rel, frame_number), ...]
        self.ball_tracking_history = []  # History of all ball tracking sequences
        self.ball_tracking_ratios = {}  # Dictionary mapping frame_number to ratio (horizontal_distance / ball_radius)
        self.rim_reference = None  # Current reference rim location for coordinate system: {'x': int, 'y': int} or None
        self.previous_rim_reference = None  # Previous rim reference to track changes
        
        # Simple goal detection tracking (above/below rim on same vertical line)
        self.above_rim_frames = []  # List of frame numbers when ball is above rim (same vertical line)
        self.below_rim_frames = []  # List of frame numbers when ball is down in rim (same vertical line)
        self.same_vertical_frames = []  # List of frame numbers when ball is on same vertical line as rim
        self.within_rim_frames = []  # List of frame numbers when ball is within rim boundaries (between upper and lower lines)
        self.inside_rim_horizontal_frames = []  # List of frame numbers when ball is inside rim horizontally
        self.goal_checking_triggered = False  # Flag to indicate goal checking logic is triggered
        self.goal_checking_triggered_frame = None  # Frame number when goal checking was triggered
        self.goal_checking_ratio_values = []  # List of (frame_number, ratio) tuples for frames between rim lines
        
        # Goal detection state
        self.rim_crossing_detected = False  # Flag if ball crossed rim plane
        self.ball_above_rim = False  # Track if ball was above rim
        self.last_goal_frame = None  # Track last goal frame to avoid duplicates
        self.ball_below_rim = False  # Track if ball was below rim
        self.rim_crossing_frame = None  # Frame when ball crossed rim
        self.rim_crossing_velocity = None  # Vertical velocity when crossing rim
        self.potential_goal_sequence = []  # Track sequence of ball positions around rim
        
        # Shot detection
        self.shot_detected = False
        self.frame_count = 0
        self.frame_to_count = 60  # Frames to track after shot (will be set based on FPS)
        
        # Team identification
        self.team_colors = {}  # Store dominant colors for each team
        
        # Image capture tracking
        self.rim_image_saved = False  # Track if rim image has been saved
        self.last_ball_image_frame = -1  # Track last frame when ball image was saved
        self.ball_image_save_interval = 30  # Save ball image every N frames when detected
        self.rim_image_saved_frame = -1  # Track frame when rim image was saved
        
        # Horizontal distance tracking
        self.ball_rim_horizontal_distance = None  # Current horizontal distance between ball and rim (pixels)
        self.ball_rim_horizontal_distance_rim = None  # Which rim the distance is measured against ('rim_a' or 'rim_b')
        self.ball_rim_horizontal_distance_ratio = None  # Ratio of horizontal distance to ball radius
    
    def intersect(self, line1, line2):
        """
        Check if two line segments intersect
        
        Args:
            line1: List of two tuples [(x1, y1), (x2, y2)]
            line2: List of two tuples [(x3, y3), (x4, y4)]
            
        Returns:
            True if lines intersect, False otherwise
        """
        [(x1, y1), (x2, y2)] = line1
        [(x3, y3), (x4, y4)] = line2

        # Calculate slopes (m) and y-intercepts (b) for the two lines
        m1 = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else float("inf")
        b1 = y1 - m1 * x1 if m1 != float("inf") else None

        m2 = (y4 - y3) / (x4 - x3) if (x4 - x3) != 0 else float("inf")
        b2 = y3 - m2 * x3 if m2 != float("inf") else None

        # Check for parallel lines
        if m1 == m2:
            if b1 == b2:
                return True  # Lines overlap
            else:
                return False  # Lines are parallel but not overlapping
        else:
            # Check for intersection point
            if m1 == float("inf"):
                x_intersect = x1
                y_intersect = m2 * x_intersect + b2
            elif m2 == float("inf"):
                x_intersect = x3
                y_intersect = m1 * x_intersect + b1
            else:
                x_intersect = (b2 - b1) / (m1 - m2)
                y_intersect = m1 * x_intersect + b1

            # Check if the intersection point lies within the line segments
            if (
                min(x1, x2) <= x_intersect <= max(x1, x2)
                and min(y1, y2) <= y_intersect <= max(y1, y2)
                and min(x3, x4) <= x_intersect <= max(x3, x4)
                and min(y3, y4) <= y_intersect <= max(y3, y4)
            ):
                return True
            else:
                return False
    
    def save_detection_image(self, frame, detection_type, frame_number, fps=30):
        """
        Save an image when rim or ball is detected
        
        Args:
            frame: Frame to save
            detection_type: 'rim' or 'ball'
            frame_number: Current frame number
            fps: Frames per second
        """
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        timestamp = frame_number / fps if fps > 0 else 0
        filename = f"{detection_type}_detected_frame_{frame_number:06d}_time_{timestamp:.2f}s.jpg"
        filepath = output_dir / filename
        
        cv2.imwrite(str(filepath), frame)
        print(f"  → Saved {detection_type} detection image: {filename}")
        return filepath
    
    def detect_rim(self, frame, frame_number=None, save_image=True, fps=30):
        """
        Detect rim_a and rim_b in the frame (classes 0=rim_a, 2=rim_b from best.pt model)
        Model best.pt detects: 0=rim_a, 1=basketball, 2=rim_b
        rim_a and rim_b are tracked as separate objects
        
        Args:
            frame: Input frame
            frame_number: Current frame number (for image saving)
            save_image: Whether to save image when rim is detected
            fps: Frames per second (for timestamp calculation)
            
        Returns:
            True if at least one rim (rim_a or rim_b) detected, False otherwise
        """
        # Store frame width if not already stored
        if self.frame_width is None:
            self.frame_width = frame.shape[1]
        
        # Run YOLOv8 inference on the frame, classes=[0, 2] for rim_a and rim_b
        rim_results = self.ball_rim_model.predict(
            frame, 
            classes=[0, 2],  # 0=rim_a, 2=rim_b
            max_det=2,  # Allow up to 2 rims (one of each type)
            conf=self.confidence_threshold, 
            verbose=False
        )

        # Reset detection flags
        self.rim_a_detected = False
        self.rim_b_detected = False
        
        if rim_results[0].boxes.__len__() != 0:
            boxes = rim_results[0].boxes
            names = rim_results[0].names  # Get class names
            
            # Process each detection separately
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = names.get(cls_id, f"rim_{cls_id}")
                
                rim_location_xywh = box.xywh[0].cpu().numpy().astype(int)  # xywh
                
                # Convert to dictionary
                rim_location = {
                    "x": rim_location_xywh[0],
                    "y": rim_location_xywh[1],
                    "w": rim_location_xywh[2],
                    "h": rim_location_xywh[3]
                }
                
                # Create standard line across rim for goal detection
                standard_line = [
                    (rim_location["x"] - rim_location["w"] // 2, rim_location["y"]),
                    (rim_location["x"] + rim_location["w"] // 2, rim_location["y"])
                ]
                
                # Store rim_a or rim_b separately
                # Note: rim_a and rim_b are just different objects, not assigned to teams
                # Teams score on the opponent's basket, so team assignment is determined during goal detection
                if cls_id == 0:  # rim_a
                    self.rim_a_location = rim_location
                    self.rim_a_bounding_box = np.array([x1, y1, x2, y2])
                    self.rim_a_standard_line = standard_line
                    self.rim_a_detected = True
                    # Determine which team's basket this is based on position (for goal tracking)
                    rim_center_x = rim_location["x"]
                    frame_midpoint = self.frame_width / 2
                    self.rim_a_team = "Team A" if rim_center_x < frame_midpoint else "Team B"
                    # Update rim_a as reference axis (update each frame to handle rim movement)
                    new_rim_reference = {"x": rim_location["x"], "y": rim_location["y"]}
                    if self.rim_reference is None:
                        # First time setting rim reference
                        self.rim_reference = new_rim_reference
                        print(f"  Using rim_a as reference axis: ({self.rim_reference['x']}, {self.rim_reference['y']})")
                    else:
                        # Rim location changed - adjust existing ball tracking coordinates
                        rim_offset_x = new_rim_reference["x"] - self.rim_reference["x"]
                        rim_offset_y = new_rim_reference["y"] - self.rim_reference["y"]
                        if abs(rim_offset_x) > 1 or abs(rim_offset_y) > 1:  # Only adjust if rim moved significantly
                            # Adjust all existing ball tracking points by the rim movement offset
                            for i in range(len(self.ball_tracking)):
                                rel_x, rel_y, frame_num = self.ball_tracking[i]
                                # Adjust relative coordinates to account for rim movement
                                self.ball_tracking[i] = (rel_x - rim_offset_x, rel_y - rim_offset_y, frame_num)
                            self.previous_rim_reference = self.rim_reference.copy()
                            self.rim_reference = new_rim_reference
                            print(f"  Rim_a moved: ({self.previous_rim_reference['x']}, {self.previous_rim_reference['y']}) -> ({self.rim_reference['x']}, {self.rim_reference['y']}) | Adjusted {len(self.ball_tracking)} tracking points")
                        else:
                            # Small movement, just update reference
                            self.rim_reference = new_rim_reference
                elif cls_id == 2:  # rim_b
                    self.rim_b_location = rim_location
                    self.rim_b_bounding_box = np.array([x1, y1, x2, y2])
                    self.rim_b_standard_line = standard_line
                    self.rim_b_detected = True
                    # Determine which team's basket this is based on position (for goal tracking)
                    rim_center_x = rim_location["x"]
                    frame_midpoint = self.frame_width / 2
                    self.rim_b_team = "Team A" if rim_center_x < frame_midpoint else "Team B"
                    # Update rim_b as reference axis if rim_a not available (update each frame to handle rim movement)
                    new_rim_reference = {"x": rim_location["x"], "y": rim_location["y"]}
                    if self.rim_reference is None:
                        # First time setting rim reference
                        self.rim_reference = new_rim_reference
                        print(f"  Using rim_b as reference axis: ({self.rim_reference['x']}, {self.rim_reference['y']})")
                    elif not self.rim_a_detected:
                        # Only use rim_b if rim_a is not detected, and adjust tracking if rim moved
                        rim_offset_x = new_rim_reference["x"] - self.rim_reference["x"]
                        rim_offset_y = new_rim_reference["y"] - self.rim_reference["y"]
                        if abs(rim_offset_x) > 1 or abs(rim_offset_y) > 1:  # Only adjust if rim moved significantly
                            # Adjust all existing ball tracking points by the rim movement offset
                            for i in range(len(self.ball_tracking)):
                                rel_x, rel_y, frame_num = self.ball_tracking[i]
                                # Adjust relative coordinates to account for rim movement
                                self.ball_tracking[i] = (rel_x - rim_offset_x, rel_y - rim_offset_y, frame_num)
                            self.previous_rim_reference = self.rim_reference.copy()
                            self.rim_reference = new_rim_reference
                            print(f"  Rim_b moved: ({self.previous_rim_reference['x']}, {self.previous_rim_reference['y']}) -> ({self.rim_reference['x']}, {self.rim_reference['y']}) | Adjusted {len(self.ball_tracking)} tracking points")
                        else:
                            # Small movement, just update reference
                            self.rim_reference = new_rim_reference
            
            # Save image if rim detected and not already saved
            # Note: Labels are only drawn in draw_annotations() to avoid duplicates
            if save_image and frame_number is not None:
                if not self.rim_image_saved or (frame_number - self.rim_image_saved_frame > 300):
                    annotated_frame = frame.copy()
                    
                    # Draw rim_a bounding box only (no label to avoid duplicates)
                    if self.rim_a_detected and self.rim_a_bounding_box is not None:
                        x1, y1, x2, y2 = self.rim_a_bounding_box
                        color = (0, 255, 0)  # Green for rim_a (BGR format)
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw rim_b bounding box only (no label to avoid duplicates)
                    if self.rim_b_detected and self.rim_b_bounding_box is not None:
                        x1, y1, x2, y2 = self.rim_b_bounding_box
                        color = (0, 255, 255)  # Yellow for rim_b (BGR format)
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    
                    self.save_detection_image(annotated_frame, "rim", frame_number, fps)
                    self.rim_image_saved = True
                    self.rim_image_saved_frame = frame_number
            
            return self.rim_a_detected or self.rim_b_detected
        
        return False
    
    def detect_ball(self, frame, frame_number=None, save_image=True, fps=30):
        """
        Detect ball in the frame using ball_detect.py functionality or default method
        
        Args:
            frame: Input frame
            frame_number: Current frame number (for image saving)
            save_image: Whether to save image when ball is detected
            fps: Frames per second (for timestamp calculation)
            
        Returns:
            Ball location (x, y) or None
        """
        # Use BasketballColorDetector if available
        if self.use_ball_detect_module and self.ball_detector is not None:
            detections = self.ball_detector.detect_ball(frame)
            
            if detections:
                # Get the best detection (highest confidence)
                best_detection = max(detections, key=lambda d: d[4])  # d[4] is confidence
                x, y, w, h, confidence = best_detection
                
                # Convert to xywh format (center x, center y, width, height)
                center_x = x + w // 2
                center_y = y + h // 2
                self.ball_location = np.array([center_x, center_y, w, h])
                ball_position = (center_x, center_y)
                
                # Save image if ball detected (with interval to avoid too many images)
                if save_image and frame_number is not None:
                    if frame_number - self.last_ball_image_frame >= self.ball_image_save_interval:
                        # Draw ball annotation on frame before saving
                        annotated_frame = frame.copy()
                        cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                        cv2.putText(annotated_frame, f"basketball {confidence:.2f}", 
                                   (x, y - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
                        self.save_detection_image(annotated_frame, "ball", frame_number, fps)
                        self.last_ball_image_frame = frame_number
                
                return ball_position
            else:
                self.ball_location = None
                return None
        else:
            # Use best.pt model for ball detection (class 1 = basketball)
            # Always use best.pt - no fallback to yolov8n
            ball_results = self.ball_model.predict(frame, classes=[1], max_det=1, conf=self.confidence_threshold, verbose=False)

            if ball_results[0].boxes.__len__() != 0:
                ball_location_xywh = ball_results[0].boxes.xywh[0].cpu().numpy().astype(int)
                self.ball_location = ball_location_xywh
                ball_position = (int(ball_location_xywh[0]), int(ball_location_xywh[1]))
                
                # Save image if ball detected (with interval to avoid too many images)
                if save_image and frame_number is not None:
                    if frame_number - self.last_ball_image_frame >= self.ball_image_save_interval:
                        # Draw ball annotation on frame before saving
                        annotated_frame = frame.copy()
                        cv2.circle(annotated_frame, ball_position, 10, (0, 0, 255), -1)
                        cv2.circle(annotated_frame, ball_position, 15, (0, 0, 255), 2)
                        cv2.putText(annotated_frame, "basketball", 
                                   (ball_position[0] - 50, ball_position[1] - 20),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
                        self.save_detection_image(annotated_frame, "ball", frame_number, fps)
                        self.last_ball_image_frame = frame_number
                
                return ball_position
            else:
                self.ball_location = None
                return None
    
    def detect_shot(self, frame):
        """
        Detect if a shot is being taken
        
        Args:
            frame: Input frame
            
        Returns:
            True if shot detected, False otherwise
        """
        if not self.use_shot_detection:
            return False
        
        shot_results = self.shot_model.predict(frame, max_det=1, conf=self.confidence_threshold, verbose=False)
        
        if shot_results[0].boxes.__len__() != 0:
            return True
        return False
    
    def detect_players(self, frame):
        """
        Detect players in the frame using best.pt model
        
        Args:
            frame: Input frame
            
        Returns:
            List of player bounding boxes
        """
        if not self.use_player_detection:
            return []
        
        # Detect using best.pt model
        # If best.pt is a basketball detection model, it might detect players or other objects
        # We'll detect all classes and filter for persons (class 0) if available
        player_results = self.player_model.predict(frame, conf=self.confidence_threshold, verbose=False)
        
        players = []
        for result in player_results:
            boxes = result.boxes
            names = result.names  # Get class names
            
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = names.get(cls_id, f"class_{cls_id}")
                
                # If model has person class (0), only include persons
                # Otherwise, include all detections (assuming best.pt detects players/people)
                if 0 in names:  # COCO person class exists
                    if cls_id == 0:  # Person class
                        players.append({
                            'bbox': (int(x1), int(y1), int(x2), int(y2)),
                            'confidence': conf,
                            'class_id': cls_id,
                            'class_name': cls_name
                        })
                else:
                    # Assume all detections are players/people if no person class
                    players.append({
                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                        'confidence': conf,
                        'class_id': cls_id,
                        'class_name': cls_name
                    })
        
        return players
    
    def calculate_vertical_velocity(self, y1, y2, frame1, frame2):
        """
        Calculate vertical velocity (pixels per frame)
        Positive velocity = downward movement, Negative = upward
        
        Args:
            y1, y2: Y coordinates
            frame1, frame2: Frame numbers
            
        Returns:
            Vertical velocity (positive = downward)
        """
        if frame2 == frame1:
            return 0
        return (y2 - y1) / (frame2 - frame1)
    
    def calculate_vertical_distance(self, ball_location, rim_location):
        """
        Calculate vertical distance between ball and rim center
        
        Args:
            ball_location: Ball location as numpy array [x, y, w, h] or dict with 'x', 'y'
            rim_location: Rim location as dict with 'x', 'y', 'w', 'h'
            
        Returns:
            Vertical distance in pixels (positive = ball below rim, negative = ball above rim)
            or None if locations are invalid
        """
        if ball_location is None or rim_location is None:
            return None
        
        # Extract ball y position
        if isinstance(ball_location, np.ndarray):
            ball_y = ball_location[1]
        elif isinstance(ball_location, dict):
            ball_y = ball_location.get('y')
        else:
            return None
        
        # Extract rim y position (center of rim)
        rim_y = rim_location.get('y')
        if rim_y is None:
            return None
        
        # Calculate vertical distance (positive = ball below rim, negative = ball above rim)
        vertical_distance = ball_y - rim_y
        
        return vertical_distance
    
    def calculate_horizontal_distance(self, ball_location, rim_location):
        """
        Calculate horizontal distance between ball and rim center
        
        Args:
            ball_location: Ball location as numpy array [x, y, w, h] or dict with 'x', 'y'
            rim_location: Rim location as dict with 'x', 'y', 'w', 'h'
            
        Returns:
            Horizontal distance in pixels (positive = ball to the right of rim, negative = ball to the left of rim)
            or None if locations are invalid
        """
        if ball_location is None or rim_location is None:
            return None
        
        # Extract ball x position
        if isinstance(ball_location, np.ndarray):
            ball_x = ball_location[0]
        elif isinstance(ball_location, dict):
            ball_x = ball_location.get('x')
        else:
            return None
        
        # Extract rim x position (center of rim)
        rim_x = rim_location.get('x')
        if rim_x is None:
            return None
        
        # Calculate horizontal distance (positive = ball to the right of rim, negative = ball to the left of rim)
        horizontal_distance = ball_x - rim_x
        
        return horizontal_distance
    
    def get_ball_radius(self, ball_location):
        """
        Get the radius of the ball from its location
        
        Args:
            ball_location: Ball location as numpy array [x, y, w, h] or dict with 'x', 'y', 'w', 'h'
            
        Returns:
            Ball radius in pixels, or None if location is invalid
        """
        if ball_location is None:
            return None
        
        # Extract ball width and height
        if isinstance(ball_location, np.ndarray):
            if len(ball_location) >= 4:
                ball_w = ball_location[2]
                ball_h = ball_location[3]
            else:
                return None
        elif isinstance(ball_location, dict):
            ball_w = ball_location.get('w')
            ball_h = ball_location.get('h')
            if ball_w is None or ball_h is None:
                return None
        else:
            return None
        
        # Calculate radius as average of width and height divided by 2 (for circular ball)
        ball_radius = (ball_w + ball_h) / 4.0
        
        return ball_radius
    
    def calculate_horizontal_distance_ratio(self, ball_location, rim_location):
        """
        Calculate the ratio of horizontal distance to ball radius
        
        Args:
            ball_location: Ball location as numpy array [x, y, w, h] or dict with 'x', 'y', 'w', 'h'
            rim_location: Rim location as dict with 'x', 'y', 'w', 'h'
            
        Returns:
            Ratio (horizontal_distance / ball_radius), or None if calculation is invalid
        """
        horizontal_distance = self.calculate_horizontal_distance(ball_location, rim_location)
        ball_radius = self.get_ball_radius(ball_location)
        
        if horizontal_distance is None or ball_radius is None or ball_radius == 0:
            return None
        
        ratio = horizontal_distance / ball_radius
        
        return ratio
    
    def check_goal(self, frame_number):
        """
        Goal detection logic:
        Within 30 frames, if ball is above rim and later is down in rim,
        on the same vertical line, that means goal.
        
        Checks against both rim_a and rim_b separately.
        Rim can exist from earlier frame (not necessarily in current frame).
        
        Args:
            frame_number: Current frame number
            
        Returns:
            Tuple (goal_scored: bool, team: str or None)
        """
        # Reset goal checking trigger flag at the start of each frame check
        # (only once per frame, not per rim check)
        self.goal_checking_triggered = False
        self.goal_checking_triggered_frame = None
        self.goal_checking_ratio_values = []  # Clear ratio values
        
        # Rim can exist from earlier frame (not necessarily in current frame)
        # Check if rim was detected at any point (even if not in current frame)
        rim_a_available = self.rim_a_location is not None and self.rim_a_standard_line is not None
        rim_b_available = self.rim_b_location is not None and self.rim_b_standard_line is not None
        
        if not rim_a_available and not rim_b_available:
            return False, None  # No rim detected (even from earlier frames)
        
        # Need at least 2 tracking points (ball above rim and later down in rim)
        if len(self.ball_tracking) < 2:
            return False, None  # Need at least 2 frames with ball tracking
        
        # Check goal against rim_a if available
        if rim_a_available:
            goal, rim_owner_team = self._check_goal_against_rim(
                self.rim_a_location, 
                self.rim_a_standard_line, 
                self.rim_a_team,
                frame_number
            )
            if goal:
                # Teams score on opponent's basket, so invert the team
                scoring_team = "Team B" if rim_owner_team == "Team A" else "Team A"
                return True, scoring_team
        
        # Check goal against rim_b if available
        if rim_b_available:
            goal, rim_owner_team = self._check_goal_against_rim(
                self.rim_b_location, 
                self.rim_b_standard_line, 
                self.rim_b_team,
                frame_number
            )
            if goal:
                # Teams score on opponent's basket, so invert the team
                scoring_team = "Team B" if rim_owner_team == "Team A" else "Team A"
                return True, scoring_team
        
        return False, None
    
    def _check_goal_against_rim(self, rim_location, standard_line, rim_team, frame_number):
        """
        Check if a goal was scored against a specific rim (rim_a or rim_b)
        
        Goal detection logic:
        Within 20 frames, if both "above in rim" and "down in rim" exist 
        (on same vertical line), that means goal.
        
        Args:
            rim_location: Dictionary with rim location {x, y, w, h}
            standard_line: Line across rim for goal detection
            rim_team: Team that owns this rim
            frame_number: Current frame number
            
        Returns:
            Tuple (goal_scored: bool, team: str or None)
        """
        if rim_location is None or standard_line is None or rim_team is None:
            return False, None
        
        # Check if ball is currently detected
        if self.ball_location is None:
            return False, None
        
        rim_x_center = rim_location["x"]
        rim_y = rim_location["y"]
        rim_width = rim_location["w"]
        rim_height = rim_location["h"]
        
        # Convert rim location to relative coordinates for comparison
        if self.rim_reference is not None:
            rim_x_center_rel = rim_x_center - self.rim_reference["x"]
            rim_y_rel = rim_y - self.rim_reference["y"]
        else:
            rim_x_center_rel = rim_x_center
            rim_y_rel = rim_y
        
        # Define rim boundaries (using relative coordinates)
        rim_top = rim_y_rel - rim_height // 2  # Top of rim
        rim_bottom = rim_y_rel + rim_height // 2  # Bottom of rim
        rim_left = rim_x_center_rel - rim_width // 2
        rim_right = rim_x_center_rel + rim_width // 2
        
        # Get current ball position (relative coordinates)
        ball_x = int(self.ball_location[0])
        ball_y = int(self.ball_location[1])
        
        if self.rim_reference is not None:
            ball_x_rel = ball_x - self.rim_reference["x"]
            ball_y_rel = ball_y - self.rim_reference["y"]
        else:
            ball_x_rel = ball_x
            ball_y_rel = ball_y
        
        # Check if ball is on same vertical line (horizontally aligned with rim)
        is_same_vertical = (rim_left <= ball_x_rel <= rim_right)
        
        # Track frames where ball is inside rim horizontally
        if is_same_vertical:
            # Ball is inside rim horizontally
            if frame_number not in self.inside_rim_horizontal_frames:
                self.inside_rim_horizontal_frames.append(frame_number)
        
        # Check if ball is definitely below the rim (below the bottom line of the rim)
        is_definitely_below_rim = (ball_y_rel > rim_bottom)
        
        # When ball is definitely below the rim, check previous 30 frames for goal success
        if is_definitely_below_rim:
            # Trigger goal checking
            self.goal_checking_triggered = True
            self.goal_checking_triggered_frame = frame_number
            
            # Get all tracking points in previous 30 frames
            previous_30_frames = [f for f in range(max(0, frame_number - 30), frame_number)]
            tracking_in_previous_30 = [(x_rel, y_rel, track_frame) for x_rel, y_rel, track_frame in self.ball_tracking 
                                      if track_frame in previous_30_frames]
            
            # Get frames where ball was horizontally between top and bottom lines of rim
            frames_between_rim_lines = []
            for x_rel, y_rel, track_frame in tracking_in_previous_30:
                # Check if ball was horizontally aligned with rim (between left and right)
                if rim_left <= x_rel <= rim_right:
                    # Check if ball was between top and bottom lines of rim
                    if rim_top <= y_rel <= rim_bottom:
                        frames_between_rim_lines.append(track_frame)
            
            # Log all ratio values for frames where ball is between upper and lower rim lines (from previous 30 frames)
            # Store ratio values for display in frame
            self.goal_checking_ratio_values = []
            previous_30_frames_list = [f for f in range(max(0, frame_number - 30), frame_number)]
            frames_between_rim_lines_in_30 = sorted([f for f in previous_30_frames_list 
                                                     if f in self.within_rim_frames and f in self.ball_tracking_ratios])
            
            if len(frames_between_rim_lines_in_30) > 0:
                print(f"[Goal Checking Triggered - Frame {frame_number}] Frames where ball is between rim lines (previous 30 frames): {frames_between_rim_lines_in_30}")
                ratio_values_log = []
                for check_frame in frames_between_rim_lines_in_30:
                    ratio = self.ball_tracking_ratios[check_frame]
                    ratio_values_log.append(f"Frame {check_frame}: ratio = {ratio:.3f} (abs = {abs(ratio):.3f})")
                    # Store for display
                    self.goal_checking_ratio_values.append((check_frame, ratio))
                
                if ratio_values_log:
                    print(f"[Goal Checking Triggered - Frame {frame_number}] All ratio values (frames between rim lines, previous 30 frames):")
                    for ratio_log in ratio_values_log:
                        print(f"  {ratio_log}")
                else:
                    print(f"[Goal Checking Triggered - Frame {frame_number}] No ratio values available for frames between rim lines in previous 30 frames")
            else:
                # No frames between rim lines in previous 30 frames, clear ratio values
                self.goal_checking_ratio_values = []
                print(f"[Goal Checking Triggered - Frame {frame_number}] No frames where ball is between rim lines in previous 30 frames")
            
                    # ============================================
            # Goal Verification Logic:
            # 1. Check if ball's direction is from above to bottom (downward) in previous 30 frames
            # 2. Check if ALL ratios of frames where ball is inside rim (from previous 30 frames) are less than 2
                    # ============================================
            
            # Check 1: Ball's direction must be from above to bottom (downward)
            is_moving_downward = False
            if len(tracking_in_previous_30) >= 2:
                # Sort by frame number
                tracking_in_previous_30.sort(key=lambda x: x[2])
                # Compare first and last frame in previous 30 frames
                first_frame_y = tracking_in_previous_30[0][1]  # y position of first frame
                last_frame_y = tracking_in_previous_30[-1][1]  # y position of last frame
                is_moving_downward = (last_frame_y > first_frame_y)  # Positive means moving down
            else:
                # Not enough tracking data to determine direction
                print(f"[Frame {frame_number}] Goal rejected: Not enough tracking data in previous 30 frames to determine ball direction")
                # Don't return early - keep flag set so display can show
                return False, None
            
            if not is_moving_downward:
                print(f"[Frame {frame_number}] Goal rejected: Ball is not moving from above to bottom in previous 30 frames")
                # Don't return early - keep flag set so display can show
                return False, None
                    
            # Check 2: ALL ratios of frames where ball is between upper and lower rim lines (from previous 30 frames) must be less than 2
            all_ratios_valid = True
            invalid_ratio_frames = []
            
            # Get frames in previous 30 frames where ball is between upper and lower rim lines
            previous_30_frames_list = [f for f in range(max(0, frame_number - 30), frame_number)]
            frames_between_rim_lines_in_30 = [f for f in previous_30_frames_list 
                                              if f in self.within_rim_frames and f in self.ball_tracking_ratios]
            
            if len(frames_between_rim_lines_in_30) == 0:
                print(f"[Frame {frame_number}] Goal rejected: No frames where ball is between upper and lower rim lines in previous 30 frames with ratio values")
                return False, None
            
            # Check all ratios for frames where ball is between rim lines (from previous 30 frames)
            for check_frame in frames_between_rim_lines_in_30:
                ratio = self.ball_tracking_ratios[check_frame]
                if abs(ratio) >= 2:
                    all_ratios_valid = False
                    invalid_ratio_frames.append((check_frame, ratio))
            
            if all_ratios_valid and len(invalid_ratio_frames) == 0:
                # Goal success! Both conditions met:
                # 1. Ball is moving from above to bottom
                # 2. ALL ratios of frames where ball is between rim lines (from previous 30 frames) are < 2
                # Avoid duplicate detections
                if hasattr(self, 'last_goal_frame') and self.last_goal_frame is not None:
                    if frame_number - self.last_goal_frame < 20:  # Within 20 frames of last goal
                        return False, None  # Likely duplicate detection
                
                self.last_goal_frame = frame_number
                print(f"[Goal Success Detected] Ball is definitely below rim at frame {frame_number}. "
                      f"Ball moving from above to bottom: ✓, All ratios for frames between rim lines (previous 30 frames) < 2: ✓ "
                      f"(checked {len(frames_between_rim_lines_in_30)} frames). Rim: {rim_team}")
                # Clear the tracking after goal detection
                self.inside_rim_horizontal_frames = []
                self.within_rim_frames = []
                self.goal_checking_triggered = False
                self.goal_checking_triggered_frame = None
                self.goal_checking_ratio_values = []  # Clear ratio values
                return True, rim_team
            else:
                # Some frames have ratio >= 2, not a successful goal
                if len(invalid_ratio_frames) > 0:
                    print(f"[Frame {frame_number}] Goal rejected: Ball is below rim, moving downward ✓, but ratio >= 2 at frames between rim lines: {invalid_ratio_frames}")
        
        # Track frames where ball is within rim's vertical boundaries (between upper and lower rim lines)
        # Check if ball is within rim's vertical boundaries (between upper and lower rim lines)
        # This applies even if ball is not on the same vertical line (horizontally aligned)
        is_within_rim_vertical = (rim_top <= ball_y_rel <= rim_bottom)
        
        if is_within_rim_vertical:
            # Track that ball is within rim boundaries (vertically between upper and lower lines)
            if frame_number not in self.within_rim_frames:
                self.within_rim_frames.append(frame_number)
                print(f"[Frame {frame_number}] Ball is within rim boundaries (between upper and lower lines)")
        
        # Clean up old frames (keep only last 30 frames for goal success check)
        current_frame = frame_number
        self.within_rim_frames = [f for f in self.within_rim_frames if current_frame - f <= 30]
        self.inside_rim_horizontal_frames = [f for f in self.inside_rim_horizontal_frames if current_frame - f <= 30]
        
        # Keep all ratios from first frame to current frame (no cleanup)
        # This allows displaying all ratio values in the current frame
        # Note: For very long videos, this may use more memory, but allows full history
        
        # ============================================
        # OLD Enhanced Goal Detection Logic - DISABLED
        # This logic has been replaced by the new goal checking logic that triggers
        # when the ball is definitely below the rim. Only the new logic should detect goals.
        # ============================================
        # The old logic is disabled to ensure goals are only detected through
        # the new goal checking mechanism that displays "Goal checking triggered"
        
        return False, None
    
    def process_image(self, image_path, output_path=None, save_image=True, frame_number=None):
        """
        Process a single image and detect ball, rim, and players
        Ball tracking is performed to draw tracking lines
        
        Args:
            image_path: Path to input image file
            output_path: Optional path to save annotated output image
            save_image: Whether to save detection images
            frame_number: Frame number for tracking (default: uses length of ball_tracking)
            
        Returns:
            Dictionary with detection results
        """
        # Read image
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Use frame_number if provided, otherwise use current tracking length
        if frame_number is None:
            frame_number = len(self.ball_tracking)
        
        print(f"Processing image: {image_path} (frame_number: {frame_number})")
        
        # Detect rim_a and rim_b
        rim_detected = self.detect_rim(frame, frame_number=frame_number, save_image=save_image, fps=30)
        if self.rim_a_detected:
            print(f"  rim_a detected at: {self.rim_a_location}")
            print(f"  rim_a belongs to: {self.rim_a_team} (left side = Team A, right side = Team B)")
        if self.rim_b_detected:
            print(f"  rim_b detected at: {self.rim_b_location}")
            print(f"  rim_b belongs to: {self.rim_b_team} (left side = Team A, right side = Team B)")
        if not rim_detected:
            print("  No rims detected")
        
        # Detect ball
        ball_position = self.detect_ball(frame, frame_number=frame_number, save_image=save_image, fps=30)
        if ball_position:
            print(f"  Ball detected at: {ball_position}")
            # Track ball in every frame (from first detection to current frame)
            # Use rim location as reference axis (relative coordinates)
            if self.ball_location is not None:
                ball_x = int(self.ball_location[0])
                ball_y = int(self.ball_location[1])
                
                # Convert to relative coordinates using rim as reference
                if self.rim_reference is not None:
                    rel_x = ball_x - self.rim_reference["x"]
                    rel_y = ball_y - self.rim_reference["y"]
                    tracked_point = (rel_x, rel_y, frame_number)
                else:
                    # Fallback to absolute coordinates if no rim reference
                    tracked_point = (ball_x, ball_y, frame_number)
                
                self.ball_tracking.append(tracked_point)
                
                # Calculate and store ratio for this frame
                # Use the closest rim (rim_a or rim_b) for ratio calculation
                ratio = None
                if self.rim_a_location is not None:
                    ratio_a = self.calculate_horizontal_distance_ratio(self.ball_location, self.rim_a_location)
                    if ratio_a is not None:
                        ratio = ratio_a
                
                if self.rim_b_location is not None:
                    ratio_b = self.calculate_horizontal_distance_ratio(self.ball_location, self.rim_b_location)
                    if ratio_b is not None:
                        # If both rims are available, use the one with smaller absolute ratio
                        if ratio is None or abs(ratio_b) < abs(ratio):
                            ratio = ratio_b
                
                if ratio is not None:
                    self.ball_tracking_ratios[frame_number] = ratio
                
                print(f"  Ball tracked (relative to rim): {tracked_point} | Total tracking points: {len(self.ball_tracking)}")
        else:
            print("  Ball not detected")
        
        # Update previous ball location for trajectory drawing
        if self.ball_location is not None:
            self.previous_ball_location = self.ball_location.copy()
        
        # Check for goal (same logic as video processing)
        goal_scored, team = self.check_goal(frame_number)
        
        if goal_scored and team:
            self.team_goals[team] += 1
            self.goal_history.append({
                'frame': frame_number,
                'team': team,
                'time': frame_number / 30.0  # Assume 30 fps for images
            })
            # Set recent goal for display (show for 30 frames for visibility)
            self.recent_goal = {
                'team': team,
                'frame': frame_number,
                'display_until': frame_number + 30  # Display for 30 frames for better visibility
            }
            print(f"GOAL! {team} scored at frame {frame_number}")
        
        # Detect players
        players = self.detect_players(frame)
        print(f"  Players detected: {len(players)}")
        for i, player in enumerate(players):
            print(f"    Player {i+1}: {player['class_name']} (confidence: {player['confidence']:.2f})")
        
        # Draw annotations (includes ball tracking lines and goal notification)
        annotated_frame = self.draw_annotations(frame, ball_position, players, frame_number=frame_number)
        
        # Save annotated image if output path is provided
        if output_path:
            cv2.imwrite(output_path, annotated_frame)
            print(f"Annotated image saved to: {output_path}")
        
        # Return detection results
        return {
            'rim_detected': rim_detected,
            'rim_a_detected': self.rim_a_detected,
            'rim_a_location': self.rim_a_location,
            'rim_a_team': self.rim_a_team,
            'rim_b_detected': self.rim_b_detected,
            'rim_b_location': self.rim_b_location,
            'rim_b_team': self.rim_b_team,
            'ball_detected': ball_position is not None,
            'ball_position': ball_position,
            'players_detected': len(players),
            'players': players
        }
    
    def process_video(self, video_path, output_path=None):
        """
        Process basketball video and count goals
        
        Args:
            video_path: Path to input video file
            output_path: Optional path to save annotated output video
        """
        # Open video
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Set frame_to_count based on FPS (track for 2 seconds after shot)
        self.frame_to_count = fps * 2
        
        # Setup video writer if output path is provided
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # First, detect rim_a and rim_b by scanning through initial frames
        print("Detecting rim_a and rim_b...")
        rim_detected = False
        scan_frame_number = 0
        for _ in range(min(300, total_frames)):  # Try first 300 frames or total frames
            ret, frame = cap.read()
            if not ret:
                break
            
            scan_frame_number += 1
            if self.detect_rim(frame, frame_number=scan_frame_number, save_image=True, fps=fps):
                rim_detected = True
                if self.rim_a_detected:
                    print(f"rim_a detected at: {self.rim_a_location}")
                    print(f"rim_a belongs to: {self.rim_a_team} (left side = Team A, right side = Team B)")
                if self.rim_b_detected:
                    print(f"rim_b detected at: {self.rim_b_location}")
                    print(f"rim_b belongs to: {self.rim_b_team} (left side = Team A, right side = Team B)")
                break
        
        if not rim_detected:
            print("Warning: No rims detected. Goal detection may be inaccurate.")
            # Clear rim detection data
            self.rim_a_detected = False
            self.rim_a_location = None
            self.rim_a_bounding_box = None
            self.rim_a_standard_line = None
            self.rim_a_team = None
            self.rim_b_detected = False
            self.rim_b_location = None
            self.rim_b_bounding_box = None
            self.rim_b_standard_line = None
            self.rim_b_team = None
        
        # Reset video to beginning
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        frame_number = 0
        color = tuple(np.random.randint(0, 255, size=3, dtype="uint8").tolist())
        
        print("Processing video...")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_number += 1
            
            # Update shot detection frame count
            if self.shot_detected:
                self.frame_count += 1
            
            # Detect shot
            if self.detect_shot(frame):
                self.shot_detected = True
                self.frame_count = 0
            
            # Detect ball
            ball_position = self.detect_ball(frame, frame_number=frame_number, save_image=True, fps=fps)
            
            # Track ball in every frame (from first detection to current frame)
            # Use rim location as reference axis (relative coordinates)
            if self.ball_location is not None:
                ball_x = int(self.ball_location[0])
                ball_y = int(self.ball_location[1])
                
                # Convert to relative coordinates using rim as reference
                if self.rim_reference is not None:
                    rel_x = ball_x - self.rim_reference["x"]
                    rel_y = ball_y - self.rim_reference["y"]
                    tracked_point = (rel_x, rel_y, frame_number)
                else:
                    # Fallback to absolute coordinates if no rim reference
                    tracked_point = (ball_x, ball_y, frame_number)
                
                self.ball_tracking.append(tracked_point)
                
                # Calculate and store ratio for this frame
                # Use the closest rim (rim_a or rim_b) for ratio calculation
                ratio = None
                if self.rim_a_location is not None:
                    ratio_a = self.calculate_horizontal_distance_ratio(self.ball_location, self.rim_a_location)
                    if ratio_a is not None:
                        ratio = ratio_a
                
                if self.rim_b_location is not None:
                    ratio_b = self.calculate_horizontal_distance_ratio(self.ball_location, self.rim_b_location)
                    if ratio_b is not None:
                        # If both rims are available, use the one with smaller absolute ratio
                        if ratio is None or abs(ratio_b) < abs(ratio):
                            ratio = ratio_b
                
                if ratio is not None:
                    self.ball_tracking_ratios[frame_number] = ratio
                
                # Log ball tracking data
                print(f"[Frame {frame_number}] Ball tracked (relative to rim): {tracked_point} | Total tracking points: {len(self.ball_tracking)}")
            
            # Check for goal
            goal_scored, team = self.check_goal(frame_number)
            
            if goal_scored and team:
                self.team_goals[team] += 1
                self.goal_history.append({
                    'frame': frame_number,
                    'team': team,
                    'time': frame_number / fps
                })
                # Set recent goal for display (show for 30 frames for visibility)
                self.recent_goal = {
                    'team': team,
                    'frame': frame_number,
                    'display_until': frame_number + 30  # Display for 30 frames for better visibility
                }
                print(f"GOAL! {team} scored at frame {frame_number} ({frame_number/fps:.2f}s)")
                
                # Save ball tracking history for this goal
                if len(self.ball_tracking) > 0:
                    history = {
                        "ball_tracking": self.ball_tracking.copy(),
                        "color": color
                    }
                    self.ball_tracking_history.append(history)
                
                # Reset shot detection but keep ball tracking for continuous line drawing
                self.frame_count = 0
                # Don't clear ball_tracking - keep it for continuous tracking from first frame
                self.shot_detected = False
                color = tuple(np.random.randint(0, 255, size=3, dtype="uint8").tolist())
            
            # Clear shot detection after frame_to_count frames, but keep ball tracking
            if self.frame_count > self.frame_to_count:
                # Don't clear ball_tracking - keep it for continuous tracking
                self.shot_detected = False
                self.frame_count = 0
            
            # Detect players
            players = self.detect_players(frame)
            
            # Draw annotations on frame
            annotated_frame = self.draw_annotations(frame, ball_position, players, frame_number)
            
            # Write frame if output is specified
            if out:
                out.write(annotated_frame)
            
            # Update previous ball location
            if self.ball_location is not None:
                self.previous_ball_location = self.ball_location.copy()
            
            # Progress update
            if frame_number % 100 == 0:
                progress = (frame_number / total_frames) * 100
                print(f"Progress: {progress:.1f}% - Team A: {self.team_goals['Team A']}, "
                      f"Team B: {self.team_goals['Team B']}")
        
        cap.release()
        if out:
            out.release()
        
        print("\n" + "="*50)
        print("ANALYSIS COMPLETE")
        print("="*50)
        print(f"Team A Goals: {self.team_goals['Team A']}")
        print(f"Team B Goals: {self.team_goals['Team B']}")
        print(f"Total Goals: {sum(self.team_goals.values())}")
        print(f"\nGoal Events:")
        for goal in self.goal_history:
            print(f"  Frame {goal['frame']} ({goal['time']:.2f}s): {goal['team']}")
    
    def draw_annotations(self, frame, ball_position, players, frame_number=None):
        """
        Draw annotations on frame (ball, rim, players, scores, trajectories, goal notifications)
        
        Args:
            frame: Input frame
            ball_position: Current ball position
            players: List of player detections
            frame_number: Current frame number (optional, for goal display)
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        # Draw rim_a if detected
        if self.rim_a_detected and self.rim_a_bounding_box is not None:
            x1, y1, x2, y2 = self.rim_a_bounding_box
            color = (0, 255, 0)  # Green for rim_a (BGR format)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw label on rim_a
            label = "rim_a"
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            # cv2.rectangle(
            #     annotated,
            #     (x1, y1 - text_height - 10),
            #     (x1 + text_width, y1),
            #     color,
            #     -1
            # )
            # cv2.putText(
            #     annotated,
            #     label,
            #     (x1, y1 - 5),
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     0.6,
            #     (255, 255, 255),  # White text
            #     2
            # )
            
            # Draw standard line across rim_a
            if self.rim_a_standard_line is not None:
                cv2.line(annotated, self.rim_a_standard_line[0], self.rim_a_standard_line[1], color, 2)
        
        # Draw rim_b if detected
        if self.rim_b_detected and self.rim_b_bounding_box is not None:
            x1, y1, x2, y2 = self.rim_b_bounding_box
            color = (0, 255, 255)  # Yellow for rim_b (BGR format)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw label on rim_b
            label = "rim_b"
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            # cv2.rectangle(
            #     annotated,
            #     (x1, y1 - text_height - 10),
            #     (x1 + text_width, y1),
            #     color,
            #     -1
            # )
            # cv2.putText(
            #     annotated,
            #     label,
            #     (x1, y1 - 5),
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     0.6,
            #     (255, 255, 255),  # White text
            #     2
            # )
            
            # Draw standard line across rim_b
            if self.rim_b_standard_line is not None:
                cv2.line(annotated, self.rim_b_standard_line[0], self.rim_b_standard_line[1], color, 2)
        
        # Ball tracking lines removed - tracking data is still collected for goal detection
        
        # Draw ball position
        if ball_position:
            cv2.circle(annotated, ball_position, 10, (0, 0, 255), -1)
            cv2.circle(annotated, ball_position, 15, (0, 0, 255), 2)
        
        # Draw players
        for player in players:
            x1, y1, x2, y2 = player['bbox']
            conf = player['confidence']
            cls_name = player.get('class_name', 'Player')
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 255, 0), 2)
            label = f"{cls_name}: {conf:.2f}"
            cv2.putText(annotated, label, 
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        # Draw shot detection indicator
        if self.shot_detected:
            cv2.putText(annotated, "Shot Detected", (10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Draw goal notification if recent goal detected (display for 30 frames)
        if self.recent_goal is not None and frame_number is not None:
            if frame_number >= self.recent_goal['frame'] and frame_number <= self.recent_goal['display_until']:
                goal_text = f"GOAL! {self.recent_goal['team']}"
                # Get frame dimensions for centering
                frame_height, frame_width = annotated.shape[:2]
                # Calculate text size for centering
                (text_width, text_height), baseline = cv2.getTextSize(
                    goal_text, cv2.FONT_HERSHEY_SIMPLEX, 2, 4
                )
                text_x = (frame_width - text_width) // 2
                text_y = (frame_height + text_height) // 2
                
                # Draw text with outline (shadow effect)
                cv2.putText(annotated, goal_text, (text_x, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 6)  # Black outline
                cv2.putText(annotated, goal_text, (text_x, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)  # Green text
            elif frame_number > self.recent_goal['display_until']:
                # Clear recent goal if display time expired
                self.recent_goal = None
        
        # Calculate and display horizontal distance from ball to rim in every frame
        # Display distance whenever both ball and rim are detected
        horizontal_distance = None  # Horizontal distance in pixels
        current_rim = None  # Track which rim we're measuring against
        
        if self.ball_location is not None:
            # Calculate distances to both rims and choose the closest one
            distance_to_rim_a = None
            distance_to_rim_b = None
            
            # Check against rim_a
            if self.rim_a_location is not None:
                # Calculate horizontal distance to rim_a
                distance_to_rim_a = self.calculate_horizontal_distance(self.ball_location, self.rim_a_location)
                if current_rim is None:
                    current_rim = "rim_a"
                    horizontal_distance = distance_to_rim_a
            
            # Check against rim_b
            if self.rim_b_location is not None:
                # Calculate horizontal distance to rim_b
                distance_to_rim_b = self.calculate_horizontal_distance(self.ball_location, self.rim_b_location)
                
                # If both rims are detected, choose the one with smaller absolute distance
                if distance_to_rim_a is not None and distance_to_rim_b is not None:
                    if abs(distance_to_rim_b) < abs(distance_to_rim_a):
                        current_rim = "rim_b"
                        horizontal_distance = distance_to_rim_b
                    else:
                        current_rim = "rim_a"
                        horizontal_distance = distance_to_rim_a
                elif distance_to_rim_b is not None:
                    current_rim = "rim_b"
                    horizontal_distance = distance_to_rim_b
            
            # Calculate ratio (horizontal_distance / ball_radius)
            horizontal_distance_ratio = None
            if horizontal_distance is not None and current_rim is not None:
                # Get the rim location for ratio calculation
                rim_location = None
                if current_rim == "rim_a" and self.rim_a_location is not None:
                    rim_location = self.rim_a_location
                elif current_rim == "rim_b" and self.rim_b_location is not None:
                    rim_location = self.rim_b_location
                
                if rim_location is not None:
                    horizontal_distance_ratio = self.calculate_horizontal_distance_ratio(self.ball_location, rim_location)
            
            # Store horizontal distance and ratio in instance variables
            if horizontal_distance is not None:
                self.ball_rim_horizontal_distance = horizontal_distance
                self.ball_rim_horizontal_distance_rim = current_rim
                self.ball_rim_horizontal_distance_ratio = horizontal_distance_ratio
            else:
                # Reset if no rim detected
                self.ball_rim_horizontal_distance = None
                self.ball_rim_horizontal_distance_rim = None
                self.ball_rim_horizontal_distance_ratio = None
            
            # Check if ball is outside rim's vertical boundaries (above upper line or below lower line)
            # and check if ball is horizontally aligned with rim (inside rim horizontally)
            ball_out_of_rim_vertical = False
            ball_inside_rim_horizontal = False
            if current_rim is not None and self.ball_location is not None:
                # Get the rim location
                rim_location = None
                if current_rim == "rim_a" and self.rim_a_location is not None:
                    rim_location = self.rim_a_location
                elif current_rim == "rim_b" and self.rim_b_location is not None:
                    rim_location = self.rim_b_location
                
                if rim_location is not None:
                    rim_x = rim_location["x"]
                    rim_y = rim_location["y"]
                    rim_width = rim_location["w"]
                    rim_height = rim_location["h"]
                    rim_left = rim_x - rim_width // 2  # Left rim boundary
                    rim_right = rim_x + rim_width // 2  # Right rim boundary
                    rim_top = rim_y - rim_height // 2  # Upper rim line
                    rim_bottom = rim_y + rim_height // 2  # Lower rim line
                    
                    ball_x = int(self.ball_location[0])
                    ball_y = int(self.ball_location[1])
                    
                    # Check if ball is outside rim's vertical boundaries
                    if ball_y < rim_top or ball_y > rim_bottom:
                        ball_out_of_rim_vertical = True
                    
                    # Check if ball is horizontally aligned with rim (inside rim horizontally)
                    if rim_left <= ball_x <= rim_right:
                        ball_inside_rim_horizontal = True
            
            # Display horizontal distance, ball radius, and ratio in every frame when both ball and rim are detected
            if horizontal_distance is not None:
                # Get ball radius
                ball_radius = self.get_ball_radius(self.ball_location)
                
                # Format distance: positive = ball to the right of rim, negative = ball to the left of rim
                distance_text = f"Distance: {abs(horizontal_distance):.1f}px"
                if horizontal_distance < 0:
                    distance_text += " (left)"
                else:
                    distance_text += " (right)"
                
                # Add ball radius
                if ball_radius is not None:
                    distance_text += f" | Ball Radius: {ball_radius:.1f}px"
                
                # Add ratio if available
                if horizontal_distance_ratio is not None:
                    distance_text += f" | Ratio: {horizontal_distance_ratio:.2f}"
                
                # Add rim identifier
                if current_rim:
                    distance_text += f" [{current_rim}]"
                
                (text_width, text_height), baseline = cv2.getTextSize(
                    distance_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                )
                # Position text near top-left, below score
                text_x = 10
                text_y = 80
                
                # # Draw background rectangle
                # cv2.rectangle(
                #     annotated,
                #     (text_x - 5, text_y - text_height - 5),
                #     (text_x + text_width + 5, text_y + 5),
                #     (0, 0, 0),  # Black background
                #     -1
                # )
                # # Draw text
                # cv2.putText(
                #     annotated,
                #     distance_text,
                #     (text_x, text_y),
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     0.7,
                #     (0, 255, 255),  # Yellow text
                #     2
                # )
                
                # Display "Ball is out of rim as horizontal" if ball is outside rim's vertical boundaries
                if ball_out_of_rim_vertical:
                    out_of_rim_text = "Ball is out of rim as horizontal"
                    (out_text_width, out_text_height), baseline = cv2.getTextSize(
                        out_of_rim_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                    )
                    # Position text below the distance text
                    out_text_x = 10
                    out_text_y = text_y + out_text_height + 15
                    
                    # # Draw background rectangle
                    # cv2.rectangle(
                    #     annotated,
                    #     (out_text_x - 5, out_text_y - out_text_height - 5),
                    #     (out_text_x + out_text_width + 5, out_text_y + 5),
                    #     (0, 0, 0),  # Black background
                    #     -1
                    # )
                    # # Draw text
                    # cv2.putText(
                    #     annotated,
                    #     out_of_rim_text,
                    #     (out_text_x, out_text_y),
                    #     cv2.FONT_HERSHEY_SIMPLEX,
                    #     0.7,
                    #     (0, 0, 255),  # Red text to indicate warning
                    #     2
                    # )
                
                # Display "Ball is inside of rim" if ball is horizontally aligned with rim
                if ball_inside_rim_horizontal:
                    inside_rim_text = "Ball is inside of rim"
                    (inside_text_width, inside_text_height), baseline = cv2.getTextSize(
                        inside_rim_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                    )
                    # Position text below the distance text (or below "out of rim" message if it exists)
                    inside_text_x = 10
                    if ball_out_of_rim_vertical:
                        # Calculate position based on out_of_rim_text height
                        (out_text_width_temp, out_text_height_temp), _ = cv2.getTextSize(
                            "Ball is out of rim as horizontal", cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                        )
                        inside_text_y = text_y + (out_text_height_temp + 15) * 2
                    else:
                        inside_text_y = text_y + inside_text_height + 15
                    
                    # # Draw background rectangle
                    # cv2.rectangle(
                    #     annotated,
                    #     (inside_text_x - 5, inside_text_y - inside_text_height - 5),
                    #     (inside_text_x + inside_text_width + 5, inside_text_y + 5),
                    #     (0, 0, 0),  # Black background
                    #     -1
                    # )
                    # # Draw text
                    # cv2.putText(
                    #     annotated,
                    #     inside_rim_text,
                    #     (inside_text_x, inside_text_y),
                    #     cv2.FONT_HERSHEY_SIMPLEX,
                    #     0.7,
                    #     (0, 255, 0),  # Green text to indicate positive status
                    #     2
                    # )
        
        # Display "Goal checking triggered" when goal checking logic is triggered
        if self.goal_checking_triggered:
            goal_checking_text = ""
            (gc_text_width, gc_text_height), baseline = cv2.getTextSize(
                goal_checking_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
            )
            # Position text in the center-top area
            frame_height, frame_width = annotated.shape[:2]
            gc_text_x = (frame_width - gc_text_width) // 2
            gc_text_y = 50
            
            
            # Display ratio values for frames where ball is between upper and lower rim lines
            if len(self.goal_checking_ratio_values) > 0:
                # Position ratio values display below "Goal checking triggered" text
                ratio_display_y = gc_text_y + gc_text_height + 30
                ratio_display_x = 10  # Left side of frame
                
                # Display header
                header_text = "Ratio values (frames between rim lines):"
                (header_width, header_height), _ = cv2.getTextSize(
                    header_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                
                # # Draw background for header
                # cv2.rectangle(
                #     annotated,
                #     (ratio_display_x - 5, ratio_display_y - header_height - 5),
                #     (ratio_display_x + header_width + 5, ratio_display_y + 5),
                #     (0, 0, 0),  # Black background
                #     -1
                # )
                # cv2.putText(
                #     annotated,
                #     header_text,
                #     (ratio_display_x, ratio_display_y),
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     0.6,
                #     (255, 255, 255),  # White text
                #     2
                # )
                
                # Display each ratio value
                current_y = ratio_display_y + header_height + 15
                max_width = 0
                ratio_lines = []
                
                for frame_num, ratio in self.goal_checking_ratio_values:
                    if ratio is not None:
                        ratio_text = f"Frame {frame_num}: ratio = {ratio:.3f} (abs = {abs(ratio):.3f})"
                        # Color code: green if abs < 2, red if abs >= 2
                        ratio_color = (0, 255, 0) if abs(ratio) < 2 else (0, 0, 255)
                    else:
                        ratio_text = f"Frame {frame_num}: ratio = N/A"
                        ratio_color = (128, 128, 128)  # Gray for N/A
                    
                    ratio_lines.append((ratio_text, ratio_color))
                    (text_width, text_height), _ = cv2.getTextSize(
                        ratio_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                    )
                    max_width = max(max_width, text_width)
                
                # Draw background rectangle for all ratio values
                if ratio_lines:
                    total_height = len(ratio_lines) * (text_height + 5) + 10
                    # cv2.rectangle(
                    #     annotated,
                    #     (ratio_display_x - 5, ratio_display_y + header_height + 10),
                    #     (ratio_display_x + max_width + 10, ratio_display_y + header_height + 10 + total_height),
                    #     (0, 0, 0),  # Black background
                    #     -1
                    # )
                    
                    # Draw each ratio line
                    for ratio_text, ratio_color in ratio_lines:
                        # cv2.putText(
                        #     annotated,
                        #     ratio_text,
                        #     (ratio_display_x, current_y),
                        #     cv2.FONT_HERSHEY_SIMPLEX,
                        #     0.5,
                        #     ratio_color,
                        #     1
                        # )
                        current_y += text_height + 5
        
        # Display all ratio values from first frame to current frame
        # Determine current frame number
        current_display_frame = frame_number
        if current_display_frame is None:
            # If frame_number not provided, use the maximum frame from ratios or ball_tracking
            if len(self.ball_tracking_ratios) > 0:
                current_display_frame = max(self.ball_tracking_ratios.keys())
            elif len(self.ball_tracking) > 0:
                current_display_frame = max([f for _, _, f in self.ball_tracking])
            else:
                current_display_frame = 0
        
        if current_display_frame is not None and len(self.ball_tracking_ratios) > 0:
            # Always show last 30 frames where ball is between upper and lower rim lines
            start_frame = max(0, current_display_frame - 30)
            previous_30_frames_list = [f for f in range(start_frame, current_display_frame + 1)]
            
            # Get frames where ball is between upper and lower rim lines from last 30 frames
            frames_between_rim_lines_last_30 = sorted([f for f in previous_30_frames_list 
                                                      if f in self.within_rim_frames and f in self.ball_tracking_ratios])
            
            all_frames_with_ratios = frames_between_rim_lines_last_30
            header_text = f"Ratios (Last 30 frames between rim lines: {start_frame} to {current_display_frame}):"
            
            # Always display the header and ratios (if any)
            # Position on the right side of the frame
            frame_height, frame_width = annotated.shape[:2]
            all_ratios_x = frame_width - 300  # Right side with margin
            all_ratios_y = 100  # Start below score
            
            # Display header
            (header_width, header_height), _ = cv2.getTextSize(
                header_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            
            # # Draw background for header
            # cv2.rectangle(
            #     annotated,
            #     (all_ratios_x - 5, all_ratios_y - header_height - 5),
            #     (all_ratios_x + header_width + 5, all_ratios_y + 5),
            #     (0, 0, 0),  # Black background
            #     -1
            # )
            # cv2.putText(
            #     annotated,
            #     header_text,
            #     (all_ratios_x, all_ratios_y),
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     0.5,
            #     (255, 255, 255),  # White text
            #     1
            # )
            
            if len(all_frames_with_ratios) > 0:
                # Display ratio values (limit to last 20 for display to avoid too much text)
                display_frames = all_frames_with_ratios[-20:] if len(all_frames_with_ratios) > 20 else all_frames_with_ratios
                current_display_y = all_ratios_y + header_height + 10
                max_display_width = 0
                all_ratio_lines = []
                
                for display_frame in display_frames:
                    ratio = self.ball_tracking_ratios[display_frame]
                    ratio_text = f"F{display_frame}: {ratio:.3f} (|{ratio:.3f}|)"
                    # Color code: green if abs < 2, red if abs >= 2
                    ratio_color = (0, 255, 0) if abs(ratio) < 2 else (0, 0, 255)
                    all_ratio_lines.append((ratio_text, ratio_color))
                    
                    (text_width, text_height), _ = cv2.getTextSize(
                        ratio_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1
                    )
                    max_display_width = max(max_display_width, text_width)
                
                # Draw background rectangle for all ratio values
                if all_ratio_lines:
                    total_display_height = len(all_ratio_lines) * (text_height + 3) + 10
                    # Limit height to fit in frame
                    max_height = frame_height - all_ratios_y - 50
                    if total_display_height > max_height:
                        total_display_height = max_height
                    
                    # cv2.rectangle(
                    #     annotated,
                    #     (all_ratios_x - 5, all_ratios_y + header_height + 5),
                    #     (all_ratios_x + max_display_width + 10, all_ratios_y + header_height + 5 + total_display_height),
                    #     (0, 0, 0),  # Black background
                    #     -1
                    # )
                    
                    # Draw each ratio line
                    for ratio_text, ratio_color in all_ratio_lines:
                        if current_display_y > frame_height - 50:  # Stop if near bottom of frame
                            break
                        # cv2.putText(
                        #     annotated,
                        #     ratio_text,
                        #     (all_ratios_x, current_display_y),
                        #     cv2.FONT_HERSHEY_SIMPLEX,
                        #     0.4,
                        #     ratio_color,
                        #     1
                        # )
                        current_display_y += text_height + 3
                    
                    # Show count if more frames exist
                    if len(all_frames_with_ratios) > 20:
                        count_text = f"... ({len(all_frames_with_ratios) - 20} more frames)"
                        # cv2.putText(
                        #     annotated,
                        #     count_text,
                        #     (all_ratios_x, current_display_y),
                        #     cv2.FONT_HERSHEY_SIMPLEX,
                        #     0.4,
                        #     (128, 128, 128),  # Gray text
                        #     1
                        # )
            else:
                # No frames between rim lines in last 30 frames
                no_frames_text = "No frames between rim lines"
                (text_width, text_height), _ = cv2.getTextSize(
                    no_frames_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1
                )
                current_display_y = all_ratios_y + header_height + 10
                
                # cv2.rectangle(
                #     annotated,
                #     (all_ratios_x - 5, current_display_y - text_height - 5),
                #     (all_ratios_x + text_width + 5, current_display_y + 5),
                #     (0, 0, 0),  # Black background
                #     -1
                # )
                # cv2.putText(
                #     annotated,
                #     no_frames_text,
                #     (all_ratios_x, current_display_y),
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     0.4,
                #     (128, 128, 128),  # Gray text
                #     1
                # )
        
        # Draw score
        score_text = f"Team A: {self.team_goals['Team A']}  |  Team B: {self.team_goals['Team B']}"
        cv2.putText(annotated, score_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
        cv2.putText(annotated, score_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        return annotated


def main():
    parser = argparse.ArgumentParser(description='Basketball Video/Image Analysis with YOLOv8')
    parser.add_argument('--video', type=str, default=None,
                       help='Path to input basketball video file')
    parser.add_argument('--image', type=str, default=None,
                       help='Path to input basketball image file')
    parser.add_argument('--images-dir', type=str, default=None,
                       help='Directory containing images to process')
    parser.add_argument('--ball_rim_model', type=str, default=None,
                       help='Path to ball/rim detection model (.pt). Default: checks model/ folder first, then sample_project')
    parser.add_argument('--shot_model', type=str, default=None,
                       help='Path to shot detection model (.pt). Default: checks model/ folder first, then sample_project')
    parser.add_argument('--player_model', type=str, default=None,
                       help='Path to player detection model (.pt). Default: uses best.pt if available, else yolov8n.pt')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save annotated output video/image (optional)')
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
    
    args = parser.parse_args()
    
    # Check if input is provided
    if not args.video and not args.image and not args.images_dir:
        parser.error("Please provide --video, --image, or --images-dir")
    
    # Initialize analyzer
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
    
    # Process based on input type
    if args.video:
        # Process video
        analyzer.process_video(args.video, args.output)
    elif args.image:
        # Process single image
        analyzer.process_image(args.image, args.output)
    elif args.images_dir:
        # Process directory of images
        images_dir = Path(args.images_dir)
        if not images_dir.exists():
            print(f"Error: Directory not found: {args.images_dir}")
            return analyzer
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(images_dir.glob(f'*{ext}'))
            image_files.extend(images_dir.glob(f'*{ext.upper()}'))
        
        image_files = sorted(image_files)
        
        if not image_files:
            print(f"No image files found in {args.images_dir}")
            return analyzer
        
        print(f"Processing {len(image_files)} images from {args.images_dir}")
        
        # Create output directory if specified
        output_dir = None
        if args.output:
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each image with frame numbers for ball tracking
        for i, image_file in enumerate(image_files):
            print(f"\n[{i+1}/{len(image_files)}] Processing: {image_file.name}")
            output_path = output_dir / f"detected_{image_file.name}" if output_dir else None
            # Process image with frame number for continuous ball tracking
            analyzer.process_image(str(image_file), str(output_path) if output_path else None, frame_number=i)
    
    return analyzer


if __name__ == "__main__":
    analyzer = main()
