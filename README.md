# ASL Gesture Recognition System with YOLOv8 and Real-time Webcam Detection

## Project Overview

This project presents a comprehensive solution for real-time American Sign Language (ASL) alphabet gesture recognition, leveraging the power of YOLOv8 for efficient object detection. The system is designed to detect and identify 26 ASL alphabet gestures (A-Z) from various input sources, with a particular focus on real-time webcam feeds. This capability provides an automated and interactive means for assisting in sign language communication, making it a valuable tool for educational purposes, assistive technology, and interactive applications.

The core of this project integrates a pre-trained YOLOv8 model with a custom Python script (`pred_yolo.py`) to facilitate live gesture detection directly from a webcam. This allows users to interact with the system in real-time, observing the model's predictions as they perform ASL gestures. The project also includes a Jupyter Notebook (`ASL_Gesture_Recognition_with_YOLOv8.ipynb`) that details the training process, configuration management, and inference procedures for various data types, including images and videos.

## Key Features

*   **YOLOv8 Object Detection**: Employs the Ultralytics YOLOv8 model for efficient and precise gesture detection.
*   **26 ASL Alphabet Recognition**: Supports recognition of all 26 ASL alphabet gestures from 'A' to 'Z'.
*   **Real-time Webcam Inference**: Includes a dedicated script (`pred_yolo.py`) for real-time ASL gesture recognition using a webcam, displaying FPS and bounding boxes.
*   **Configurable Training Parameters**: Through the `ASLConfig` class (within the Jupyter Notebook), users can easily adjust dataset, model, training, and video processing parameters.
*   **Data Augmentation**: Integrates various data augmentation techniques (e.g., rotation, scaling, cropping, HSV adjustments) to improve model generalization capabilities.
*   **Real-time Inference (General)**: Supports real-time gesture recognition on images, videos, and webcam streams.
*   **Result Visualization**: Capable of drawing bounding boxes and class labels around detected gestures, along with displaying confidence scores.

## Technology Stack

*   **Python**: Primary development language.
*   **PyTorch**: Deep learning framework.
*   **Ultralytics YOLOv8**: Object detection model.
*   **OpenCV**: Image processing and video stream handling.
*   **NumPy, Matplotlib, Seaborn, Pandas**: Data processing and visualization.
*   **YAML**: Configuration file management.

## Setup and Installation

1.  **Clone the Repository**: 
    ```bash
    git clone <repository_url>
    cd ASL_Gesture_Recognition_with_YOLOv8
    ```

2.  **Install Dependencies**: 
    It is recommended to install all necessary Python packages using `pip`.
    ```bash
    pip install -r requirements.txt
    ```

3.  **Download Dataset**: 
    The project expects the dataset structure to be as follows:
    ```
    asl_dataset/
    ├── data.yaml
    ├── train/
    │   ├── images/
    │   └── labels/
    └── valid/
        ├── images/
        └── labels/
    ```
    Please organize your ASL gesture dataset in YOLO format and update the `asl_dataset/data.yaml` file to point to the correct paths and class information.

## Usage

### 1. System Configuration

The project manages all configurations through the `ASLConfig` class. You can directly modify its attributes in the Jupyter Notebook or save it as a YAML file.

```python
class ASLConfig:
    def __init__(self):
        # Dataset configuration
        self.dataset = {
            'name': 'american-sign-language-letters',
            'num_classes': 26,
            'class_names': ['A', 'B', 'C', ..., 'Z'],
            'data_yaml': 'asl_dataset/data.yaml'
        }

        # Model configuration
        self.model = {
            'name': 'yolov8n',
            'input_size': 640,
            'confidence_threshold': 0.5,
            'iou_threshold': 0.45,
            'max_detections': 300
        }

        # Training configuration
        self.training = {
            'epochs': 30,
            'batch_size': 16,
            'learning_rate': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3,
            'patience': 50,
            'save_period': 10,
            'augmentation': { ... } # Data augmentation parameters
        }

        # Video processing configuration
        self.video = {
            'fps': 30,
            'output_format': 'mp4',
            'codec': 'mp4v',
            'show_confidence': True,
            'show_labels': True,
            'line_thickness': 3,
            'font_scale': 0.8
        }

        # Path configuration
        self.paths = {
            'data_dir': 'data',
            'models_dir': 'models',
            'results_dir': 'results',
            'videos_dir': 'videos'
        }

        # Device configuration
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize configuration
config = ASLConfig()
```

### 2. Model Training

Execute the corresponding cells in the Jupyter Notebook to start training the model. The training process will use the parameters defined in `config.training` and save the best model to `runs/detect/<training_run_name>/weights/best.pt`.

```python
# Load YOLOv8 model
model = YOLO(config.model['name'] + '.pt') # e.g., yolov8n.pt

# Start training
results = model.train(
    data=config.dataset['data_yaml'],
    epochs=config.training['epochs'],
    batch=config.training['batch_size'],
    imgsz=config.model['input_size'],
    device=config.device,
    name=training_run_name,
    lr0=config.training['learning_rate'],
    momentum=config.training['momentum'],
    weight_decay=config.training['weight_decay'],
    warmup_epochs=config.training['warmup_epochs'],
    patience=config.training['patience'],
    save_period=config.training['save_period'],
    **config.training['augmentation']
)
```

### 3. Model Inference

After training, you can use the trained model for inference. Below are examples of how to perform inference on images, videos, or webcams:

```python
# Load the best model
model = YOLO(config.model['best_model_path'])

# Image inference
results = model.predict(
    source='path/to/your/image.jpg',
    conf=config.model['confidence_threshold'],
    iou=config.model['iou_threshold'],
    imgsz=config.model['input_size'],
    show=True, # Display results
    save=True  # Save results
)

# Video inference
results = model.predict(
    source='path/to/your/video.mp4',
    conf=config.model['confidence_threshold'],
    iou=config.model['iou_threshold'],
    imgsz=config.model['input_size'],
    show=True,
    save=True,
    stream=True # Process video stream
)

# Webcam inference (typically source=0)
# results = model.predict(
#     source=0,
#     conf=config.model['confidence_threshold'],
#     iou=config.model['iou_threshold'],
#     imgsz=config.model['input_size'],
#     show=True,
#     stream=True
# )
```

## Real-time Webcam Detection (`pred_yolo.py`)

This script provides a straightforward way to perform real-time ASL gesture recognition using your webcam. It loads a pre-trained YOLOv8 model and continuously processes frames from your webcam, displaying the detected gestures with bounding boxes and FPS.

### Prerequisites

Before running `pred_yolo.py`, ensure you have:

*   A pre-trained YOLOv8 model file (e.g., `yolo11n.pt` as specified in the script). You can train your own model using the provided Jupyter Notebook or download a pre-trained one.
*   A working webcam connected to your system.

### How to Run

1.  **Place your model file**: Make sure your YOLOv8 model file (e.g., `yolo11n.pt`) is in the same directory as `pred_yolo.py`, or update the `model = YOLO("yolo11n.pt")` line in `pred_yolo.py` to the correct path of your model.
2.  **Run the script**: Open a terminal or command prompt, navigate to the project directory, and execute the script:
    ```bash
    python pred_yolo.py
    ```
3.  **Interact**: A window will open displaying your webcam feed with real-time gesture detections. Press `q` to quit the application.

## Project Structure

```
. # Project root directory
├── ASL_Gesture_Recognition_with_YOLOv8.ipynb # Jupyter Notebook for training and general inference
├── README.md # This README file
├── pred_yolo.py # Python script for real-time webcam detection
├── requirements.txt # Python dependencies
├── asl_dataset/ # Dataset directory (to be prepared by user)
│   ├── data.yaml # Dataset configuration file in YOLO format
│   ├── train/ # Training data
│   └── valid/ # Validation data
├── runs/ # Training results and model save directory
│   └── detect/
│       └── <training_run_name>/
│           ├── weights/
│           │   ├── best.pt # Best model weights
│           │   └── last.pt # Last epoch model weights
│           └── ... (other training logs and charts)
└── videos/ # Video output directory
```

## Contributing

Contributions of any kind are welcome! If you have any suggestions, bug reports, or feature requests, please feel free to submit an Issue or Pull Request.

## License

This project is licensed under the [MIT License](LICENSE).


