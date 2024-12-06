import os
import yaml
from ultralytics import YOLO
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.keypoint_rcnn import KeypointRCNN
from torchvision.models.detection import FasterRCNN
from torchvision.models import resnet
from torchvision.ops import boxes as bbox_ops
import numpy as np
from tqdm import tqdm

def setup_gpu():
    """Configure GPU settings for stable multi-GPU training"""
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"PyTorch sees {device_count} CUDA devices")
        torch.cuda.set_device(0)
        torch.autograd.set_detect_anomaly(True)
        print("Set memory allocator settings for better stability")
        torch.cuda.empty_cache()
        print("Optional: limit memory growth")
        torch.cuda.set_per_process_memory_fraction(0.9)
        print("SetupComplete")
    else:
        print("CUDA is not available")

def create_dataset_yaml():
    """Creates the dataset.yaml file for YOLOv8"""
    classes = {
        0: 'HumanLimb',
        1: 'Insekt',
        2: 'KeinUnkraut',
        3: 'blatt',
        4: 'loewenzahn',
        5: 'loewenzahn_Bluete',
        6: 'loewenzahn_Staengel'
    }
    dataset_config = {
        'path': os.path.abspath('./yolo_dataset'),
        'train': 'train',
        'val': 'validate',
        'names': classes,
        'nc': len(classes)
    }
    try:
        with open('dataset.yaml', 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
        print("Dataset YAML created successfully")
    except Exception as e:
        print(f"Error creating dataset.yaml: {str(e)}")

def train_model():
    """Trains the YOLOv8 model with multi-GPU support"""
    try:
        # Initialize YOLOv8 model
        model = YOLO('yolov8n.pt')

        # Training parameters optimized for multi-GPU setup
        results = model.train(
            data='dataset.yaml',
            epochs=100,
            imgsz=640,
            batch=16,
            device='cuda',
            patience=20,
            save=True,
            project='training_results',
            name='train5',
            exist_ok=False,
            pretrained=True,
            verbose=True,
            workers=8,
            cache=True,
            amp=True,
            warmup_epochs=5.0,
            close_mosaic=10,
            multi_scale=True
        )

        print("Training completed successfully")
    except Exception as e:
        print(f"Training error occurred: {str(e)}")
        raise

def main():
    try:
        print("Configuring GPUs...")
        setup_gpu()
        print("Starting training process...")

        print("Creating dataset.yaml...")
        create_dataset_yaml()

        print("Starting model training...")
        train_model()
        print("Training completed successfully")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        torch.cuda.empty_cache()
        print("GPU memory cleared")

if __name__ == "__main__":
    main()
