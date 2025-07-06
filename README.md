# 🚗 Parking Space Detection using Vision Transformers

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.0%2B-yellow)](https://huggingface.co/transformers/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📖 Overview

This project was developed as a **group capstone project** for our Computer Vision coursework. Our team collaborated to explore and demonstrate the application of **Vision Transformers (ViTs)** for real-world parking space detection, comparing their performance against traditional CNN approaches. The system implements a **few-shot learning model** capable of accurate pixel-level segmentation using minimal labeled data, making it highly scalable to new parking environments without extensive annotation. This capstone project showcases our understanding of transformer-based computer vision, few-shot semantic segmentation, and intelligent transportation systems.

A **state-of-the-art parking space detection system** that uses Vision Transformers to identify free and occupied parking spaces in real-time, built with PyTorch, DINOv2, and custom SegFormer architectures.

## 🚀 What This Project Does

This project is an advanced computer vision system that lets you:

- **Detect parking spaces** in real-time from video streams
- **Segment free vs occupied spaces** with high accuracy using pixel-level classification
- **Process multiple datasets** (ACPDS, PKLot) with different annotation levels
- **Compare architectures** between Vision Transformers and traditional CNNs
- **Achieve few-shot learning** with minimal training data

It's perfect for learning how Vision Transformers, semantic segmentation, and real-time computer vision work together, and serves as a comprehensive capstone project for intelligent parking systems.

## 📝 How It Works

1. **DINOv2** extracts rich visual features from parking lot images using self-supervised learning
2. **Custom SegFormer decoder** processes these features for pixel-level classification
3. **Few-shot learning** allows the model to adapt to new parking environments with minimal data
4. **Real-time processing** analyzes video streams and outputs segmentation masks
5. **Comprehensive evaluation** measures performance using F1, Precision, Recall, and IoU metrics

## 🛠️ Tech Stack

- **PyTorch**: Deep learning framework
- **Transformers (HuggingFace)**: Pre-trained Vision Transformer models
- **DINOv2**: Self-supervised Vision Transformer backbone
- **OpenCV**: Computer vision and video processing
- **NumPy & Matplotlib**: Data processing and visualization
- **Torchmetrics**: Comprehensive evaluation metrics
- **Jupyter Notebooks**: Interactive development and experimentation

## 📦 Project Structure

```
├── src/
│   ├── model.py                       # Core model architectures
│   ├── dataset.py                     # Data loading and preprocessing
│   ├── model_test.py                  # Comprehensive evaluation framework
│   └── video_test.py                  # Real-time video processing
├── notebooks/
│   ├── dinoXformer.ipynb             # DINOv2 + Linear classifier
│   ├── segformer_acpds.ipynb         # SegFormer on ACPDS dataset
│   ├── segformer_pklot.ipynb         # SegFormer on PKLot dataset
│   └── simple_linear_probing.ipynb   # Linear probing baseline
├── weights/                          # Trained model checkpoints
├── test/                             # Test videos and outputs
├── CV_Project_Report.pdf             # Detailed technical report
└── README.md                         # Project documentation
```

## 🖥️ Getting Started

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (8GB+ VRAM recommended)
- 16GB+ RAM for dataset processing

### Install Dependencies

```bash
pip install torch torchvision transformers datasets
pip install opencv-python matplotlib pillow numpy
pip install torchmetrics tqdm evaluate
```

### Dataset Setup

**Option A: ACPDS Dataset (Fully Annotated)**

```bash
# Download ACPDS dataset and organize:
├── ACPDS/
│   ├── ACPDS/
│   │   ├── images/
│   │   └── int_masks/
```

**Option B: PKLot Dataset (Partially Annotated)**

```bash
# Download PKLot dataset and organize:
├── PKLOT/
│   ├── PKLOT/
│   │   ├── images/
│   │   └── int_masks/
```

### Train Models

**DINOv2 + Linear Probing:**

```bash
jupyter notebook dinoXformer.ipynb
```

**SegFormer Architecture:**

```bash
jupyter notebook segformer_acpds.ipynb  # For ACPDS dataset
jupyter notebook segformer_pklot.ipynb  # For PKLot dataset
```

**Baseline Comparison:**

```bash
jupyter notebook simple_linear_probing.ipynb
```

## 🧑‍💻 Usage

### Evaluate Performance

```bash
python model_test.py  # Comprehensive evaluation with metrics
```

### Real-Time Video Processing

```bash
python video_test.py  # Process parking lot videos
```

### Running Tests

To evaluate model performance:

```bash
python model_test.py
```

To process test videos:

```bash
python video_test.py
```

## 🔧 Technical Details

### Model Architecture

**Core Components:**

1. **Feature Extractor**: Pre-trained DINOv2 Vision Transformer (frozen backbone)
2. **Decoder Head**: SegFormer-style decoder with conv layers, ReLU, BatchNorm, dropout
3. **Classification**: 3 classes (background, free, occupied)

**Technical Implementation:**

```python
class Dinov2ForSemanticSegmentation(Dinov2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.dinov2 = Dinov2Model(config)  # Frozen feature extractor
        self.classifier = SegFormerHead(   # Custom decoder head
            in_channels=config.hidden_size,
            hidden_size=256,
            num_classes=config.num_labels
        )
```

### How Each Component Works

- **DINOv2 Backbone**: Extracts rich visual features using self-supervised learning
- **SegFormer Decoder**: Processes features for pixel-level segmentation with conv layers, ReLU, BatchNorm
- **Dataset Pipeline**: Handles ACPDS/PKLot data with proper preprocessing and augmentation
- **Evaluation Framework**: Comprehensive metrics including F1, Precision, Recall, IoU
- **Video Processing**: Real-time inference with visualization overlays and polygon detection

## 📊 Results

- **F1 Score**: 0.85+ on ACPDS dataset
- **Precision**: 0.88+ for occupied space detection
- **Recall**: 0.82+ for free space detection
- **Real-time Processing**: 2 FPS on standard GPU hardware
- **Few-Shot Learning**: High accuracy with minimal training data

## 🎯 Project Contributions

- **Novel Architecture**: DINOv2 + SegFormer combination for parking space detection
- **Few-Shot Learning**: Minimal data requirements for new environments
- **Comprehensive Evaluation**: Multiple experimental approaches and datasets
- **Real-World Application**: Video processing pipeline for practical deployment
- **Performance Analysis**: Detailed ViT vs CNN comparison

## 🤝 Team Collaboration & Development

This capstone project demonstrates our team's ability to:

- **Collaborative Development**: Distributed work across multiple experimental tracks and components
- **Technical Coordination**: Systematic approach to hypothesis testing and validation across team members
- **Code Quality**: Modular, well-documented, and reproducible implementations
- **Project Management**: Clear documentation, task distribution, and result integration
- **Academic Excellence**: Comprehensive approach to a complex computer vision problem

## 📚 References

- **DINOv2**: Learning Robust Visual Features without Supervision
- **SegFormer**: Simple and Efficient Design for Semantic Segmentation
- **Vision Transformers**: An Image is Worth 16x16 Words
- **ACPDS**: Annotated Car Parking Dataset for Semantic Segmentation
- **PKLot**: Parking Lot Database for Classification and Segmentation

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
