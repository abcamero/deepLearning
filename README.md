# deepLearning
This repository contains codes for deep learning -> Handcrafted Networks and Predefined networks (Tensorflow.keras)

## PyTorch Image Classification

A unified script for training and inference on the Intel Image Classification dataset.

### Features
- Eliminates code duplication between training and inference scripts
- Fixed transform order and added normalization for consistent preprocessing
- Modular design with separate functions for training and inference
- Command-line interface for easy usage

### Usage

#### Training
```bash
python pytorch_classification.py --mode train --epochs 5 --batch_size 64
```

#### Inference
```bash
python pytorch_classification.py --mode infer --image_path path/to/image.jpg
```

### Files
- `pytorch_classification.py`: Unified training and inference script
- `PytorchClassification_Module1.py`: Original training script
- `pytorchClassification_Inference.py`: Original inference script
- `my_classifier.pth`: Trained model weights
