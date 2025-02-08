# MNIST Image Enhancement and Classification with U-Net

This repository contains a deep learning project that combines image enhancement and classification using a modified U-Net architecture. The project focuses on recovering high-quality images from deliberately degraded MNIST digits while simultaneously classifying them.

## Project Overview

The project implements two versions of the solution:
1. `enhancement.py`: Basic U-Net implementation for image enhancement only
2. `enhancement_classification.py`: Enhanced version that combines image reconstruction with digit classification

The model takes low-resolution MNIST digits as input and performs two tasks:
- `enhancement.py`: Reconstructs high-quality versions of the digits
- `enhancement_classification.py`: Reconstructs high-quality versions and Classifies the digits (0-9) in the enhanced version

## Features

- Data preprocessing with controlled quality degradation
- U-Net architecture with skip connections
- Multi-task learning (in the enhanced version)
- Real-time loss tracking during training
- Visualization tools for model performance
- Random sample testing with probability distribution display

## Requirements

- TensorFlow
- NumPy
- Matplotlib

## Model Architecture

The enhanced model (`enhancement_classification.py`) includes:

- Encoder blocks with Conv2D and MaxPooling layers
- Decoder blocks with Conv2DTranspose and skip connections
- Classification branch with Dense layers and dropout
- Dual outputs for image reconstruction and digit classification

## Output

The model provides:
- Training, validation, and testing loss plots
- Reconstructed image visualization
- Classification probability distribution for each digit (0-9)

## Model Parameters

- Input shape: 28x28x1
- Training epochs: 10
- Batch size: 64
- Validation split: First 5000 images
- Optimizer: Adam
- Loss functions: 
  - Image reconstruction: Binary crossentropy
  - Classification: Categorical crossentropy (enhanced_classification version only)

## Implementation Details

### Data Preprocessing
- Images are deliberately degraded using resizing operations
- Pixel values are normalized to [0,1] range
- Labels are one-hot encoded for classification

### Training
- Real-time loss tracking via custom callback
- Multi-task learning with weighted losses
- Skip connections for preserving spatial information

### Visualization
- Custom plotting functions for loss curves
- Image reconstruction visualization
- Classification probability distribution display

## Results

The model outputs:
1. Enhanced image quality from degraded inputs
2. Classification probabilities for each digit class
3. Comprehensive loss tracking across training, validation, and testing sets