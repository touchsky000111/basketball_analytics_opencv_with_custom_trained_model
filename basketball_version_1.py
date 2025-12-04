"""
Basketball Video Analysis Script
Uses YOLOv8 models to detect ball, players, and hoops
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
    def __init__(self, ball_rim_model_path=None, player_model_path=None, confidence_threshold=0.5, device=None, use_yolov8n_for_ball=False, use_ball_detect_module=True, ball_model_path="model/best.pt"):
        """
        Initialize the Basketball Analyzer
        
        Args:
            ball_rim_model_path: Path to ball/rim detection model (.pt). 
                                 Classes: 0=rim_a, 1=basketball, 2=rim_b
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
        
        # Rim detection - track rim_a and rim_b as separate objects
        # Model best.pt detects: 0=rim_a, 1=basketball, 2=rim_b
        self.rim_a_location = None  # xywh: [x, y, w, h] center of the box, width, height
        self.rim_a_bounding_box = None  # xyxy: [x1, y1, x2, y2]
        self.rim_a_standard_line = None  # Line across rim_a
        self.rim_a_detected = False  # Flag to track if rim_a was detected
        self.rim_a_team = None  # Which team's rim_a: "Team A" (left) or "Team B" (right) - kept for compatibility
        
        self.rim_b_location = None  # xywh: [x, y, w, h] center of the box, width, height
        self.rim_b_bounding_box = None  # xyxy: [x1, y1, x2, y2]
        self.rim_b_standard_line = None  # Line across rim_b
        self.rim_b_detected = False  # Flag to track if rim_b was detected
        self.rim_b_team = None  # Which team's rim_b: "Team A" (left) or "Team B" (right)
        
        self.frame_width = None  # Store frame width to determine rim position
        
        # Ball location (current frame only)
        self.ball_location = None  # xywh: [x, y, w, h]
        
        # Ball tracking for goal detection
        # List of (frame_number, ball_x, ball_y, rim_x, rim_y, rim_left, rim_right, rim_type)
        self.ball_tracking = []
        
        # Goal tracking
        self.team_goals = defaultdict(int)  # Track goals per team
        self.goal_history = []  # Store goal events with frame numbers
        self.recent_goal = None  # Store recent goal info for display: {'team': str, 'frame': int, 'display_until': int}
        self.last_goal_frame = None  # Track last goal frame to avoid duplicates
        
        # Image capture tracking
        self.rim_image_saved = False  # Track if rim image has been saved
        self.last_ball_image_frame = -1  # Track last frame when ball image was saved
        self.ball_image_save_interval = 30  # Save ball image every N frames when detected
        self.rim_image_saved_frame = -1  # Track frame when rim image was saved
    
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
            verbose=False,
            save=False  # Disable YOLO's built-in saving/plotting
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
                
                # Create standard line across rim
                standard_line = [
                    (rim_location["x"] - rim_location["w"] // 2, rim_location["y"]),
                    (rim_location["x"] + rim_location["w"] // 2, rim_location["y"])
                ]
                
                # Store rim_a or rim_b separately
                # Note: rim_a and rim_b are just different objects
                if cls_id == 0:  # rim_a
                    self.rim_a_location = rim_location
                    self.rim_a_bounding_box = np.array([x1, y1, x2, y2])
                    self.rim_a_standard_line = standard_line
                    self.rim_a_detected = True
                    # Determine which team's basket this is based on position
                    rim_center_x = rim_location["x"]
                    frame_midpoint = self.frame_width / 2
                    self.rim_a_team = "Team A" if rim_center_x < frame_midpoint else "Team B"
                elif cls_id == 2:  # rim_b
                    self.rim_b_location = rim_location
                    self.rim_b_bounding_box = np.array([x1, y1, x2, y2])
                    self.rim_b_standard_line = standard_line
                    self.rim_b_detected = True
                    # Determine which team's basket this is based on position
                    rim_center_x = rim_location["x"]
                    frame_midpoint = self.frame_width / 2
                    self.rim_b_team = "Team A" if rim_center_x < frame_midpoint else "Team B"
            
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
            ball_results = self.ball_model.predict(frame, classes=[1], max_det=1, conf=self.confidence_threshold, verbose=False, save=False)

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
        player_results = self.player_model.predict(frame, conf=self.confidence_threshold, verbose=False, save=False)
        
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
    
    def check_goal(self, frame_number):
        """
        Simple goal detection logic:
        In 30 frames, if ball direction is from above rim to below rim,
        and during this, if ball is between left line and right line of rim, that is goal.
        
        Args:
            frame_number: Current frame number
            
        Returns:
            Tuple (goal_scored: bool, team: str or None)
        """
        # Need at least 2 frames of tracking data
        if len(self.ball_tracking) < 2:
            return False, None
        
        # Get last 30 frames of tracking
        recent_tracking = [t for t in self.ball_tracking if frame_number - t[0] <= 30]
        
        if len(recent_tracking) < 2:
            return False, None
        
        # Check against both rim_a and rim_b
        for rim_type in ['rim_a', 'rim_b']:
            rim_location = self.rim_a_location if rim_type == 'rim_a' else self.rim_b_location
            rim_team = self.rim_a_team if rim_type == 'rim_a' else self.rim_b_team
            
            if rim_location is None or rim_team is None:
                continue
            
            # Filter tracking data for this rim
            # Tracking tuple: (frame_number, ball_x, ball_y, rim_x, rim_y, rim_left, rim_right, rim_type)
            rim_tracking = [t for t in recent_tracking if t[7] == rim_type]
            
            if len(rim_tracking) < 2:
                continue
            
            # Sort by frame number
            rim_tracking.sort(key=lambda x: x[0])
            
            # Check if ball moved from above rim to below rim
            first_frame = rim_tracking[0]
            last_frame = rim_tracking[-1]
            
            first_ball_y = first_frame[2]  # ball_y
            last_ball_y = last_frame[2]    # ball_y
            rim_y_pos = first_frame[4]     # rim_y (should be same for all frames)
            
            # Ball must move from above rim (ball_y < rim_y) to below rim (ball_y > rim_y)
            if first_ball_y >= rim_y_pos or last_ball_y <= rim_y_pos:
                continue  # Not moving from above to below
            
            # Check if ball was between left and right lines during this movement
            # Use rim boundaries from tracking data (indices 5 and 6)
            ball_between_lines = False
            for track in rim_tracking:
                ball_x = track[1]  # ball_x
                track_rim_left = track[5]  # rim_left from tracking data
                track_rim_right = track[6]  # rim_right from tracking data
                if track_rim_left <= ball_x <= track_rim_right:
                    ball_between_lines = True
                    break
            
            if not ball_between_lines:
                continue  # Ball was never between left and right lines
            
            # Goal detected! Avoid duplicate detections
            if self.last_goal_frame is not None:
                if frame_number - self.last_goal_frame < 30:  # Within 30 frames of last goal
                    return False, None  # Likely duplicate detection
            
            self.last_goal_frame = frame_number
            
            # Teams score on opponent's basket, so invert the team
            scoring_team = "Team B" if rim_team == "Team A" else "Team A"
            
            print(f"GOAL! {scoring_team} scored at frame {frame_number} (rim: {rim_type})")
            return True, scoring_team
        
        return False, None
    
    def process_image(self, image_path, output_path=None, save_image=True, frame_number=None):
        """
        Process a single image and detect ball, rim, and players
        
        Args:
            image_path: Path to input image file
            output_path: Optional path to save annotated output image
            save_image: Whether to save detection images
            frame_number: Frame number (optional, for image saving)
            
        Returns:
            Dictionary with detection results
        """
        # Read image
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        if frame_number is None:
            frame_number = 0
        
        print(f"Processing image: {image_path}")
        
        # Detect rim_a and rim_b
        rim_detected = self.detect_rim(frame, frame_number=frame_number, save_image=save_image, fps=30)
        if self.rim_a_detected:
            print(f"  rim_a detected at: {self.rim_a_location}")
        if self.rim_b_detected:
            print(f"  rim_b detected at: {self.rim_b_location}")
        if not rim_detected:
            print("  No rims detected")
        
        # Detect ball
        ball_position = self.detect_ball(frame, frame_number=frame_number, save_image=save_image, fps=30)
        if ball_position:
            print(f"  Ball detected at: {ball_position}")
            
            # Track ball position relative to rim for goal detection
            if self.ball_location is not None:
                ball_x = int(self.ball_location[0])
                ball_y = int(self.ball_location[1])
                
                # Track against rim_a if detected
                if self.rim_a_detected and self.rim_a_location is not None:
                    rim_x = self.rim_a_location["x"]
                    rim_y = self.rim_a_location["y"]
                    rim_width = self.rim_a_location["w"]
                    rim_left = rim_x - rim_width // 2
                    rim_right = rim_x + rim_width // 2
                    self.ball_tracking.append((frame_number, ball_x, ball_y, rim_x, rim_y, rim_left, rim_right, 'rim_a'))
                
                # Track against rim_b if detected
                if self.rim_b_detected and self.rim_b_location is not None:
                    rim_x = self.rim_b_location["x"]
                    rim_y = self.rim_b_location["y"]
                    rim_width = self.rim_b_location["w"]
                    rim_left = rim_x - rim_width // 2
                    rim_right = rim_x + rim_width // 2
                    self.ball_tracking.append((frame_number, ball_x, ball_y, rim_x, rim_y, rim_left, rim_right, 'rim_b'))
                
                # Keep only last 30 frames of tracking
                self.ball_tracking = [t for t in self.ball_tracking if frame_number - t[0] <= 30]
        else:
            print("  Ball not detected")
        
        # Check for goal
        goal_scored, team = self.check_goal(frame_number)
        if goal_scored and team:
            self.team_goals[team] += 1
            self.goal_history.append({
                'frame': frame_number,
                'team': team,
                'time': frame_number / 30.0  # Assume 30 fps for images
            })
            # Set recent goal for display (show for 30 frames)
            self.recent_goal = {
                'team': team,
                'frame': frame_number,
                'display_until': frame_number + 30
            }
        
        # Detect players
        players = self.detect_players(frame)
        print(f"  Players detected: {len(players)}")
        for i, player in enumerate(players):
            print(f"    Player {i+1}: {player['class_name']} (confidence: {player['confidence']:.2f})")
        
        # Draw annotations
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
            'rim_b_detected': self.rim_b_detected,
            'rim_b_location': self.rim_b_location,
            'ball_detected': ball_position is not None,
            'ball_position': ball_position,
            'players_detected': len(players),
            'players': players
        }
    
    def process_video(self, video_path, output_path=None):
        """
        Process basketball video
        
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
            print("Warning: No rims detected.")
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
        
        print("Processing video...")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_number += 1
            
            # Detect ball
            ball_position = self.detect_ball(frame, frame_number=frame_number, save_image=True, fps=fps)
            
            # Track ball position relative to rim for goal detection
            if self.ball_location is not None:
                ball_x = int(self.ball_location[0])
                ball_y = int(self.ball_location[1])
                
                # Track against rim_a if detected
                if self.rim_a_detected and self.rim_a_location is not None:
                    rim_x = self.rim_a_location["x"]
                    rim_y = self.rim_a_location["y"]
                    rim_width = self.rim_a_location["w"]
                    rim_left = rim_x - rim_width // 2
                    rim_right = rim_x + rim_width // 2
                    self.ball_tracking.append((frame_number, ball_x, ball_y, rim_x, rim_y, rim_left, rim_right, 'rim_a'))
                
                # Track against rim_b if detected
                if self.rim_b_detected and self.rim_b_location is not None:
                    rim_x = self.rim_b_location["x"]
                    rim_y = self.rim_b_location["y"]
                    rim_width = self.rim_b_location["w"]
                    rim_left = rim_x - rim_width // 2
                    rim_right = rim_x + rim_width // 2
                    self.ball_tracking.append((frame_number, ball_x, ball_y, rim_x, rim_y, rim_left, rim_right, 'rim_b'))
                
                # Keep only last 30 frames of tracking
                self.ball_tracking = [t for t in self.ball_tracking if frame_number - t[0] <= 30]
            
            # Check for goal
            goal_scored, team = self.check_goal(frame_number)
            if goal_scored and team:
                self.team_goals[team] += 1
                self.goal_history.append({
                    'frame': frame_number,
                    'team': team,
                    'time': frame_number / fps
                })
                # Set recent goal for display (show for 30 frames)
                self.recent_goal = {
                    'team': team,
                    'frame': frame_number,
                    'display_until': frame_number + 30
                }
            
            # Detect players
            players = self.detect_players(frame)
            
            # Draw annotations on frame
            annotated_frame = self.draw_annotations(frame, ball_position, players, frame_number)
            
            # Write frame if output is specified
            if out:
                out.write(annotated_frame)
            
            # Progress update
            if frame_number % 100 == 0:
                progress = (frame_number / total_frames) * 100
                print(f"Progress: {progress:.1f}%")
        
        cap.release()
        if out:
            out.release()
        
        print("\n" + "="*50)
        print("ANALYSIS COMPLETE")
        print("="*50)
        
        # Print goal summary
        print(f"\nTeam A Goals: {self.team_goals['Team A']}")
        print(f"Team B Goals: {self.team_goals['Team B']}")
        print(f"Total Goals: {self.team_goals['Team A'] + self.team_goals['Team B']}")
        
        if self.goal_history:
            print("\nGoal Events:")
            for goal in self.goal_history:
                print(f"  Frame {goal['frame']}: {goal['team']} scored at {goal['time']:.2f}s")
    
    def draw_annotations(self, frame, ball_position, players, frame_number=None):
        """
        Draw annotations on frame (ball, rim, players, trajectories)
        
        Args:
            frame: Input frame
            ball_position: Current ball position
            players: List of player detections
            frame_number: Current frame number (optional)
            
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
            cv2.rectangle(
                annotated,
                (x1, y1 - text_height - 10),
                (x1 + text_width, y1),
                color,
                -1
            )
            cv2.putText(
                annotated,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),  # White text
                2
            )
            
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
            cv2.rectangle(
                annotated,
                (x1, y1 - text_height - 10),
                (x1 + text_width, y1),
                color,
                -1
            )
            cv2.putText(
                annotated,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),  # White text
                2
            )
            
            # Draw standard line across rim_b
            if self.rim_b_standard_line is not None:
                cv2.line(annotated, self.rim_b_standard_line[0], self.rim_b_standard_line[1], color, 2)
        
        # Draw ball position
        if ball_position:
            cv2.circle(annotated, ball_position, 10, (0, 0, 255), -1)
            cv2.circle(annotated, ball_position, 15, (0, 0, 255), 2)
            
            # Draw label on ball
            label = "basketball"
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            ball_x, ball_y = ball_position
            # Position label above the ball
            label_x = ball_x - text_width // 2
            label_y = ball_y - 20
            cv2.rectangle(
                annotated,
                (label_x - 5, label_y - text_height - 5),
                (label_x + text_width + 5, label_y + 5),
                (0, 0, 255),  # Red background (same as ball color)
                -1
            )
            cv2.putText(
                annotated,
                label,
                (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),  # White text
                2
            )
        
        # Draw players
        for player in players:
            x1, y1, x2, y2 = player['bbox']
            conf = player['confidence']
            cls_name = player.get('class_name', 'Player')
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 255, 0), 2)
            label = f"{cls_name}: {conf:.2f}"
            cv2.putText(annotated, label, 
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
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
    parser.add_argument('--player_model', type=str, default=None,
                       help='Path to player detection model (.pt). Default: uses best.pt if available, else yolov8n.pt')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save annotated output video/image (optional)')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Confidence threshold for detections (default: 0.5)')
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
