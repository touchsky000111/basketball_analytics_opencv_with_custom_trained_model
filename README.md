# Basketball Video Analytics

A comprehensive basketball video analysis system that automatically detects balls, players, rims, and counts goals for each team using YOLOv8 deep learning models.

## Features

- 🏀 **Ball Detection**: Detects basketball using custom YOLOv8 models
- 🏀 **Rim Detection**: Detects basketball rims (rim_a and rim_b) 
- 👥 **Player Detection**: Detects players on the court
- 🎯 **Goal Detection**: Automatically counts goals for Team A and Team B
- 📊 **Video Processing**: Complete pipeline from video input to annotated output
- 🎥 **Frame-by-Frame Analysis**: Detailed analysis with tracking and annotations

## Installation

1. Clone or download this repository

2. Install required dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

Process a basketball video with default settings:

```bash
python main.py --video input/input_video.mp4 --output output/video.mp4
```

This will:
1. Extract frames from the input video
2. Analyze each frame for ball, rim, and player detection
3. Detect and count goals
4. Generate an annotated output video
5. Automatically clean up intermediate files

## Command Line Arguments

### Required Arguments

- `--video`: Path to input video file
- `--output`: Path to output video file

### Frame Extraction Options

- `--interval`: Extract every Nth frame (default: 1, extracts all frames)
- `--start`: Frame number to start extraction from (default: 0)
- `--end`: Frame number to stop extraction at (default: end of video)
- `--format`: Image format for extracted frames - jpg, jpeg, png, bmp, tiff (default: jpg)
- `--prefix`: Prefix for extracted frame filenames (default: frame)

### Directory Options

- `--images-dir`: Directory to save extracted frames (default: images)
- `--output-images-dir`: Directory to save analyzed frames (default: output_images)

### Model Options

- `--ball_rim_model`: Path to ball/rim detection model (.pt file)
- `--shot_model`: Path to shot detection model (.pt file)
- `--player_model`: Path to player detection model (.pt file)
- `--confidence`: Confidence threshold for detections (default: 0.3)
- `--device`: Device to use - "cuda", "cpu", or None for auto-detect (default: None)
- `--ball-model`: Path to ball detection model (default: model/best.pt)

### Other Options

- `--keep-intermediate`: Keep intermediate frame directories after processing (default: deleted automatically)

## Usage Examples

### Process entire video
```bash
python main.py --video input/input_video.mp4 --output output/video.mp4
```

### Process with custom confidence threshold
```bash
python main.py --video input/input_video.mp4 --output output/video.mp4 --confidence 0.5
```

### Process specific frame range
```bash
python main.py --video input/input_video.mp4 --output output/video.mp4 --start 1000 --end 2000
```

### Extract every 2nd frame (faster processing)
```bash
python main.py --video input/input_video.mp4 --output output/video.mp4 --interval 2
```

### Use GPU acceleration
```bash
python main.py --video input/input_video.mp4 --output output/video.mp4 --device cuda
```

### Keep intermediate files for debugging
```bash
python main.py --video input/input_video.mp4 --output output/video.mp4 --keep-intermediate
```

### Use custom model paths
```bash
python main.py --video input/input_video.mp4 --output output/video.mp4 \
    --ball_rim_model model/best.pt \
    --confidence 0.5 \
    --device cuda
```

## Project Structure

```
BasketBall_Analytics/
├── main.py                 # Main processing script (unified pipeline)
├── basketball_analyzer.py  # Core analysis engine
├── extract_frames.py        # Frame extraction utility
├── generate_video.py       # Video generation utility
├── requirements.txt        # Python dependencies
├── model/                  # YOLOv8 model files
│   ├── best.pt            # Ball/rim detection model
│   └── ...
├── input/                  # Input video files
│   └── input_video.mp4
└── output/                # Output video files
    └── video.mp4
```

## How It Works

The processing pipeline consists of three main steps:

1. **Frame Extraction**: Extracts frames from the input video and saves them as images
2. **Frame Analysis**: 
   - Detects basketball, rims (rim_a and rim_b), and players
   - Tracks ball movement and position relative to rims
   - Detects goals based on ball trajectory through rim
   - Annotates frames with bounding boxes, scores, and goal notifications
3. **Video Generation**: Combines analyzed frames into final output video

## Output

The script provides:
- Annotated output video with:
  - Ball detection (red circle)
  - Rim detection (green/yellow boxes)
  - Player detection (cyan boxes)
  - Score display (Team A vs Team B)
  - Goal notifications
- Console output with:
  - Processing progress
  - Goal detection results
  - Final score summary
  - Goal event timeline

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended for faster processing)
- YOLOv8 models (included in `model/` directory)

## Notes

- Intermediate directories (`images/` and `output_images/`) are automatically deleted after processing unless `--keep-intermediate` is used
- The script automatically detects and uses GPU if available
- Frame extraction interval can be adjusted to speed up processing (higher interval = fewer frames = faster but less detailed)

## Troubleshooting

**Video writer error**: Make sure the output directory exists or the script will create it automatically.

**Low detection accuracy**: Try adjusting the `--confidence` threshold or ensure your models are properly trained.

**Out of memory**: Use `--interval` to process fewer frames or reduce video resolution.

## License

This project is for basketball video analysis and research purposes.

