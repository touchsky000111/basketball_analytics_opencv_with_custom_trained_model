"""
Basketball Detection using YOLO Model or Color Detection Method
Detects basketballs in images using:
  - YOLO model (recommended): Uses trained YOLO model (e.g., model/best.pt)
  - Color detection: Detects orange basketballs using HSV color space (fallback)
"""

import cv2
import numpy as np
import os
from pathlib import Path
import argparse
import json
from analyze_basketball_color import analyze_basketball_color

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not available. YOLO model detection will be disabled.")


class BasketballConfig:
    """Configuration class for basketball color and shape parameters"""
    
    # Default basketball color in HSV (orange basketball)
    # Optimized based on analysis of reference image (path/ball.png)
    # HSV Statistics: Hue 9.0-15.0, Saturation 129.0-255.0, Value 85.0-149.4
    DEFAULT_LOWER_COLOR = [4, 109, 65]     # Lower HSV bound: analyzed from reference image
    DEFAULT_UPPER_COLOR = [20, 255, 159]   # Upper HSV bound: analyzed from reference image
    
    # Default shape parameters
    DEFAULT_MIN_AREA = 70                # Minimum area in pixels
    DEFAULT_MAX_AREA = 80               # Maximum area in pixels
    DEFAULT_MIN_CIRCULARITY = 0.5          # Minimum circularity (0-1, 1 = perfect circle)
    DEFAULT_MIN_ASPECT_RATIO = 0.5         # Minimum width/height ratio
    DEFAULT_MAX_ASPECT_RATIO = 2.0         # Maximum width/height ratio
    
    # Morphological operations
    DEFAULT_KERNEL_SIZE = 5                # Kernel size for noise removal
    DEFAULT_MORPH_ITERATIONS = 2            # Number of iterations for morphological operations
    
    def __init__(self, 
                 lower_color=None, 
                 upper_color=None,
                 min_area=None,
                 max_area=None,
                 min_circularity=None,
                 min_aspect_ratio=None,
                 max_aspect_ratio=None,
                 kernel_size=None,
                 morph_iterations=None):
        """
        Initialize basketball detection configuration
        
        Args:
            lower_color: Lower HSV color bound [H, S, V] (default: [5, 50, 50])
            upper_color: Upper HSV color bound [H, S, V] (default: [25, 255, 255])
            min_area: Minimum area in pixels (default: 100)
            max_area: Maximum area in pixels (default: 50000)
            min_circularity: Minimum circularity 0-1 (default: 0.5)
            min_aspect_ratio: Minimum width/height ratio (default: 0.5)
            max_aspect_ratio: Maximum width/height ratio (default: 2.0)
            kernel_size: Kernel size for morphological operations (default: 5)
            morph_iterations: Iterations for morphological operations (default: 2)
        """
        self.lower_color = np.array(lower_color if lower_color else self.DEFAULT_LOWER_COLOR)
        self.upper_color = np.array(upper_color if upper_color else self.DEFAULT_UPPER_COLOR)
        self.min_area = min_area if min_area is not None else self.DEFAULT_MIN_AREA
        self.max_area = max_area if max_area is not None else self.DEFAULT_MAX_AREA
        self.min_circularity = min_circularity if min_circularity is not None else self.DEFAULT_MIN_CIRCULARITY
        self.min_aspect_ratio = min_aspect_ratio if min_aspect_ratio is not None else self.DEFAULT_MIN_ASPECT_RATIO
        self.max_aspect_ratio = max_aspect_ratio if max_aspect_ratio is not None else self.DEFAULT_MAX_ASPECT_RATIO
        self.kernel_size = kernel_size if kernel_size is not None else self.DEFAULT_KERNEL_SIZE
        self.morph_iterations = morph_iterations if morph_iterations is not None else self.DEFAULT_MORPH_ITERATIONS
    
    @classmethod
    def from_dict(cls, config_dict):
        """Create config from dictionary"""
        return cls(**config_dict)
    
    @classmethod
    def from_json(cls, json_path):
        """Load config from JSON file"""
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_reference_image(cls, image_path, **kwargs):
        """
        Create config by analyzing a reference basketball image
        
        Args:
            image_path: Path to reference basketball image
            **kwargs: Additional config parameters to override defaults
            
        Returns:
            BasketballConfig object with color values from image analysis
        """
        analysis_result = analyze_basketball_color(image_path, show_mask=False)
        
        if analysis_result is None:
            print("Warning: Could not analyze reference image, using defaults")
            return cls(**kwargs)
        
        # Use analyzed color values
        config_params = {
            'lower_color': analysis_result['lower_color'],
            'upper_color': analysis_result['upper_color'],
            **kwargs  # Allow overriding other parameters
        }
        
        return cls(**config_params)
    
    def to_dict(self):
        """Convert config to dictionary"""
        return {
            'lower_color': self.lower_color.tolist(),
            'upper_color': self.upper_color.tolist(),
            'min_area': self.min_area,
            'max_area': self.max_area,
            'min_circularity': self.min_circularity,
            'min_aspect_ratio': self.min_aspect_ratio,
            'max_aspect_ratio': self.max_aspect_ratio,
            'kernel_size': self.kernel_size,
            'morph_iterations': self.morph_iterations
        }
    
    def save_json(self, json_path):
        """Save config to JSON file"""
        with open(json_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)
    
    def print_config(self):
        """Print current configuration"""
        print("="*60)
        print("Basketball Detection Configuration")
        print("="*60)
        print(f"Color (HSV):")
        print(f"  Lower bound: {self.lower_color}")
        print(f"  Upper bound: {self.upper_color}")
        print(f"Shape Parameters:")
        print(f"  Area range: {self.min_area} - {self.max_area} pixels")
        print(f"  Min circularity: {self.min_circularity}")
        print(f"  Aspect ratio: {self.min_aspect_ratio} - {self.max_aspect_ratio}")
        print(f"Morphological Operations:")
        print(f"  Kernel size: {self.kernel_size}x{self.kernel_size}")
        print(f"  Iterations: {self.morph_iterations}")
        print("="*60)


class BasketballColorDetector:
    def __init__(self, config=None, model_path="model/best.pt", use_yolo=True, confidence_threshold=0.3, **kwargs):
        """
        Initialize the Basketball Detector
        
        Args:
            config: BasketballConfig object (optional, for color detection)
            model_path: Path to YOLO model file (default: 'model/best.pt'). If None, tries to find best.pt
            use_yolo: Whether to use YOLO model (True) or color detection (False)
            confidence_threshold: Confidence threshold for YOLO detections (default: 0.3)
            **kwargs: Configuration parameters for color detection (will create config if not provided)
                     - lower_color, upper_color: HSV color bounds
                     - min_area, max_area: Area thresholds
                     - min_circularity: Minimum circularity threshold
                     - min_aspect_ratio, max_aspect_ratio: Aspect ratio bounds
                     - kernel_size, morph_iterations: Morphological operation parameters
        """
        self.use_yolo = use_yolo and YOLO_AVAILABLE
        self.confidence_threshold = confidence_threshold
        self.yolo_model = None
        
        # Initialize YOLO model if requested - ONLY use best.pt
        if self.use_yolo:
            # ONLY use best.pt - no fallbacks to other models
            if model_path is None:
                model_path = "model/best.pt"
            
            # Load best.pt model
            if Path(model_path).exists():
                try:
                    self.yolo_model = YOLO(model_path)
                    print(f"Loaded YOLO model: {model_path}")
                    print(f"Model classes: {self.yolo_model.names}")
                except Exception as e:
                    print(f"Error loading YOLO model: {e}")
                    print("Falling back to color detection.")
                    self.use_yolo = False
            else:
                print(f"Warning: best.pt model not found at {model_path}. Using color detection.")
                self.use_yolo = False
        
        # Initialize color detection config
        if config is None:
            self.config = BasketballConfig(**kwargs)
        else:
            self.config = config
        
        # For backward compatibility
        self.lower_orange = self.config.lower_color
        self.upper_orange = self.config.upper_color
        self.min_area = self.config.min_area
        self.max_area = self.config.max_area
    
    def detect_ball(self, image):
        """
        Detect basketball in an image using YOLO model or color detection
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            list: List of detected basketballs, each as (x, y, w, h, confidence)
                  where (x, y) is top-left corner, (w, h) is width and height
        """
        if self.use_yolo and self.yolo_model is not None:
            return self._detect_ball_yolo(image)
        else:
            return self._detect_ball_color(image)
    
    def _detect_ball_yolo(self, image):
        """
        Detect basketball using YOLO model
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            list: List of detected basketballs, each as (x, y, w, h, confidence)
        """
        detections = []
        
        try:
            # Run YOLO inference
            # For best.pt model: 0=rim_a, 1=basketball, 2=rim_b
            # Filter to only detect basketball (class 1)
            results = self.yolo_model.predict(
                image,
                classes=[1],  # Only detect basketball (class 1)
                conf=self.confidence_threshold,
                verbose=False
            )
            
            # Extract detections
            boxes = results[0].boxes
            for box in boxes:
                # Get bounding box coordinates (xyxy format)
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = float(box.conf[0])
                cls = int(box.cls[0])
                
                # Only include basketball detections (class 1)
                # This is already filtered by classes=[1] above, but double-check
                if cls == 1:
                    # Convert to (x, y, w, h) format
                    x = int(x1)
                    y = int(y1)
                    w = int(x2 - x1)
                    h = int(y2 - y1)
                    
                    detections.append((x, y, w, h, confidence))
        
        except Exception as e:
            print(f"Error during YOLO detection: {e}")
        
        return detections
    
    def _detect_ball_color(self, image):
        """
        Detect basketball in an image using color detection
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            list: List of detected basketballs, each as (x, y, w, h, confidence)
                  where (x, y) is top-left corner, (w, h) is width and height
        """
        # Convert BGR to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Create mask for basketball color
        mask = cv2.inRange(hsv, self.config.lower_color, self.config.upper_color)
        
        # Apply morphological operations to remove noise
        kernel = np.ones((self.config.kernel_size, self.config.kernel_size), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=self.config.morph_iterations)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=self.config.morph_iterations)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area
            if self.config.min_area <= area <= self.config.max_area:
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calculate circularity (basketballs are roughly circular)
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    
                    # Filter by circularity
                    if circularity >= self.config.min_circularity:
                        # Calculate aspect ratio (should be close to 1 for circles)
                        aspect_ratio = float(w) / h if h > 0 else 0
                        
                        # Filter by aspect ratio
                        if self.config.min_aspect_ratio <= aspect_ratio <= self.config.max_aspect_ratio:
                            # Confidence based on circularity and area
                            # Higher circularity and reasonable area = higher confidence
                            area_confidence = min(area / self.config.max_area, 1.0) if self.config.max_area > 0 else 0.5
                            confidence = min(circularity * 0.7 + area_confidence * 0.3, 1.0)
                            detections.append((x, y, w, h, confidence))
        
        return detections
    
    def draw_detections(self, image, detections):
        """
        Draw bounding boxes on image for detected basketballs
        Only draws the detection with the highest confidence
        
        Args:
            image: Input image
            detections: List of detections from detect_ball()
            
        Returns:
            Image with bounding boxes drawn
        """
        result_image = image.copy()
        
        if not detections:
            return result_image
        
        # Find the detection with the highest confidence
        best_detection = max(detections, key=lambda d: d[4])  # d[4] is confidence
        x, y, w, h, confidence = best_detection
        
        # Draw bounding box for the best detection only
        cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Draw confidence text
        label = f"basketball {confidence:.2f}"
        cv2.putText(result_image, label, (x, y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return result_image
    
    def process_image(self, image_path, output_path=None, show_result=False):
        """
        Process a single image and detect basketballs
        
        Args:
            image_path: Path to input image
            output_path: Path to save output image (optional)
            show_result: Whether to display the result
            
        Returns:
            Number of basketballs detected
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image {image_path}")
            return 0
        
        # Detect basketballs
        detections = self.detect_ball(image)
        
        # Draw detections
        result_image = self.draw_detections(image, detections)
        
        # Save if output path is provided
        if output_path:
            cv2.imwrite(output_path, result_image)
            print(f"Processed {image_path}: Found {len(detections)} basketball(s). Saved to {output_path}")
        else:
            print(f"Processed {image_path}: Found {len(detections)} basketball(s)")
        
        # Show result if requested
        if show_result:
            cv2.imshow("Basketball Detection", result_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return len(detections)
    
    def process_folder(self, input_folder, output_folder=None, show_results=False):
        """
        Process all images in a folder
        
        Args:
            input_folder: Path to folder containing images
            output_folder: Path to save output images (optional)
            show_results: Whether to display results
            
        Returns:
            Dictionary with statistics
        """
        input_path = Path(input_folder)
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = [f for f in input_path.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        if not image_files:
            print(f"No image files found in {input_folder}")
            return {}
        
        # Create output folder if specified
        if output_folder:
            output_path = Path(output_folder)
            output_path.mkdir(parents=True, exist_ok=True)
        
        total_detections = 0
        images_with_balls = 0
        
        print(f"Processing {len(image_files)} images...")
        
        for img_file in sorted(image_files):
            if output_folder:
                output_file = output_path / f"detected_{img_file.name}"
            else:
                output_file = None
            
            num_detections = self.process_image(
                str(img_file), 
                str(output_file) if output_file else None,
                show_results
            )
            
            total_detections += num_detections
            if num_detections > 0:
                images_with_balls += 1
        
        stats = {
            'total_images': len(image_files),
            'images_with_balls': images_with_balls,
            'total_detections': total_detections,
            'avg_detections_per_image': total_detections / len(image_files) if image_files else 0
        }
        
        print("\n" + "="*50)
        print("Detection Statistics:")
        print(f"Total images processed: {stats['total_images']}")
        print(f"Images with basketballs: {stats['images_with_balls']}")
        print(f"Total basketballs detected: {stats['total_detections']}")
        print(f"Average detections per image: {stats['avg_detections_per_image']:.2f}")
        print("="*50)
        
        return stats


def main():
    parser = argparse.ArgumentParser(description='Detect basketballs in images using YOLO model or color detection')
    parser.add_argument('--input', '-i', type=str, default='images',
                       help='Input folder or image path (default: images)')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output folder for processed images (optional)')
    parser.add_argument('--show', '-s', action='store_true',
                       help='Show detection results')
    parser.add_argument('--config', '-c', type=str, default=None,
                       help='Path to JSON configuration file')
    parser.add_argument('--lower-color', nargs=3, type=int, default=None,
                       help='Lower HSV bounds (H S V) for basketball color')
    parser.add_argument('--upper-color', nargs=3, type=int, default=None,
                       help='Upper HSV bounds (H S V) for basketball color')
    parser.add_argument('--min-area', type=int, default=None,
                       help='Minimum area for detection')
    parser.add_argument('--max-area', type=int, default=None,
                       help='Maximum area for detection')
    parser.add_argument('--min-circularity', type=float, default=None,
                       help='Minimum circularity (0-1, default: 0.5)')
    parser.add_argument('--min-aspect-ratio', type=float, default=None,
                       help='Minimum aspect ratio (default: 0.5)')
    parser.add_argument('--max-aspect-ratio', type=float, default=None,
                       help='Maximum aspect ratio (default: 2.0)')
    parser.add_argument('--print-config', action='store_true',
                       help='Print current configuration and exit')
    parser.add_argument('--save-config', type=str, default=None,
                       help='Save current configuration to JSON file')
    parser.add_argument('--model', '-m', type=str, default='model/best.pt',
                       help='Path to YOLO model file (default: model/best.pt)')
    parser.add_argument('--use-yolo', action='store_true', default=True,
                       help='Use YOLO model for detection (default: True if model found)')
    parser.add_argument('--use-color', action='store_true', default=False,
                       help='Use color detection instead of YOLO model')
    parser.add_argument('--confidence', type=float, default=0.3,
                       help='Confidence threshold for YOLO detections (default: 0.3)')
    
    args = parser.parse_args()
    
    # Determine detection method
    use_yolo = args.use_yolo and not args.use_color
    
    # Load configuration
    if args.config:
        config = BasketballConfig.from_json(args.config)
    else:
        # Create config from command line arguments
        config_kwargs = {}
        if args.lower_color:
            config_kwargs['lower_color'] = args.lower_color
        if args.upper_color:
            config_kwargs['upper_color'] = args.upper_color
        if args.min_area is not None:
            config_kwargs['min_area'] = args.min_area
        if args.max_area is not None:
            config_kwargs['max_area'] = args.max_area
        if args.min_circularity is not None:
            config_kwargs['min_circularity'] = args.min_circularity
        if args.min_aspect_ratio is not None:
            config_kwargs['min_aspect_ratio'] = args.min_aspect_ratio
        if args.max_aspect_ratio is not None:
            config_kwargs['max_aspect_ratio'] = args.max_aspect_ratio
        
        config = BasketballConfig(**config_kwargs)
    
    # Print config if requested
    if args.print_config:
        config.print_config()
        return
    
    # Save config if requested
    if args.save_config:
        config.save_json(args.save_config)
        print(f"Configuration saved to {args.save_config}")
        return
    
    # Handle model path - convert empty string or "None" to None
    model_path = args.model
    if model_path and (model_path.lower() == 'none' or model_path.strip() == ''):
        model_path = None
    
    # Initialize detector
    detector = BasketballColorDetector(
        config=config,
        model_path=model_path,
        use_yolo=use_yolo,
        confidence_threshold=args.confidence
    )
    
    # Print detection method
    if detector.use_yolo:
        print(f"Using YOLO model: {args.model}")
    else:
        print("Using color detection method")
    
    input_path = Path(args.input)
    
    # Check if input is a file or folder
    if input_path.is_file():
        # Process single image
        detector.process_image(str(input_path), args.output, args.show)
    elif input_path.is_dir():
        # Process folder
        detector.process_folder(str(input_path), args.output, args.show)
    else:
        print(f"Error: {args.input} is not a valid file or directory")


if __name__ == "__main__":
    main()

