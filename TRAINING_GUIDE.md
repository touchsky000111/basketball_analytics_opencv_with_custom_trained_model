# Training Guide: Custom Basketball Detection Model

## What's Required to Train an AI Model for Basketball Detection

### 1. **Labeled Dataset** (Most Important!)
You need to annotate your images with bounding boxes around basketballs.

**Options:**
- **Manual Labeling**: Use tools like:
  - [LabelImg](https://github.com/HumanSignal/labelImg) (GUI tool)
  - [CVAT](https://cvat.org/) (Web-based)
  - [Roboflow](https://roboflow.com/) (Online platform)
  - [YOLO Label](https://github.com/developer0hye/Yolo_Label) (Simple GUI)

**Dataset Structure:**
```
datasets/
└── basketball_detection/
    ├── train/
    │   ├── images/
    │   │   ├── frame_000000.jpg
    │   │   ├── frame_000001.jpg
    │   │   └── ...
    │   └── labels/
    │       ├── frame_000000.txt
    │       ├── frame_000001.txt
    │       └── ...
    ├── val/
    │   ├── images/
    │   └── labels/
    └── data.yaml
```

**Label Format (YOLO):**
Each `.txt` file contains one line per object:
```
class_id center_x center_y width height
```
Example: `0 0.5 0.5 0.1 0.1` (normalized coordinates 0-1)

### 2. **Dataset Configuration File (data.yaml)**
```yaml
path: /path/to/datasets/basketball_detection
train: train/images
val: val/images

names:
  0: basketball
```

### 3. **Hardware Requirements**
- **GPU (Recommended)**: NVIDIA GPU with CUDA support
  - Minimum: 4GB VRAM (GTX 1050 Ti, GTX 1060)
  - Recommended: 8GB+ VRAM (RTX 3060, RTX 3070, etc.)
- **CPU**: Can train on CPU but much slower (10-100x slower)
- **RAM**: 8GB minimum, 16GB+ recommended
- **Storage**: Enough space for dataset + model checkpoints

### 4. **Software Requirements**
- Python 3.8+
- PyTorch (with CUDA if using GPU)
- Ultralytics YOLO
- OpenCV
- NumPy

### 5. **Training Parameters**
- **Epochs**: Number of training iterations (50-300 typical)
- **Batch Size**: Images per batch (depends on GPU memory)
- **Image Size**: Input resolution (640x640 typical)
- **Learning Rate**: How fast model learns (auto-tuned by YOLO)
- **Pretrained Weights**: Start from YOLOv8n/m/s/l/x

### 6. **Dataset Split**
- **Training**: 70-80% of images
- **Validation**: 10-20% of images
- **Test**: 10% (optional)

### 7. **Quality Requirements**
- **Minimum Images**: 100+ images (more = better)
- **Diverse Conditions**: Different lighting, angles, backgrounds
- **Balanced Dataset**: Mix of easy and hard cases
- **Accurate Labels**: Precise bounding boxes

## Quick Start Steps

1. **Install Dependencies** (already done):
   ```bash
   pip install ultralytics opencv-python numpy torch torchvision
   ```

2. **Label Your Images**:
   - Use LabelImg or similar tool
   - Create bounding boxes around basketballs
   - Export in YOLO format

3. **Organize Dataset**:
   - Split images into train/val folders
   - Create data.yaml file

4. **Train Model**:
   ```bash
   python train_basketball_model.py
   ```

5. **Evaluate Results**:
   - Check training metrics (mAP, precision, recall)
   - Test on validation images
   - Adjust parameters if needed

## Tips for Better Results

1. **More Data**: 500+ images is ideal
2. **Data Augmentation**: YOLO does this automatically
3. **Pretrained Models**: Start with YOLOv8n.pt (faster) or YOLOv8m.pt (more accurate)
4. **Transfer Learning**: Use pretrained weights (already done by default)
5. **Hyperparameter Tuning**: Adjust epochs, batch size based on results

## Expected Training Time

- **GPU (RTX 3060)**: 1-4 hours for 100 epochs
- **CPU**: 10-40 hours for 100 epochs
- **Cloud (Google Colab)**: Free GPU, ~2-3 hours

## Next Steps

1. Use `prepare_dataset.py` to help organize your images
2. Use `train_basketball_model.py` to train the model
3. Use `evaluate_model.py` to test the trained model

