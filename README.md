# Poultry Disease Classification using Deep Learning

## Project Overview
This project implements a comprehensive deep learning solution for automated classification of poultry diseases from images. The system uses convolutional neural networks (CNNs) and transfer learning to identify common poultry diseases including Newcastle Disease, Cocci virus, and Salmonella, helping farmers with early detection and intervention.

##  Problem Statement
The global poultry industry faces significant economic losses due to infectious diseases. Traditional diagnosis methods relying on visual assessment by experts are:
- **Subjective**: Varies between practitioners
- **Slow**: Delays in diagnosis can spread disease
- **Inaccessible**: Limited veterinary services in remote areas

This project addresses these challenges by providing an automated, rapid, and accurate disease classification system.

##  Dataset
- **Total Images**: 6,812 across 6 categories
- **Classes**: Newcastle Disease, Cocci virus, Salmonella, and Healthy birds
- **Class Distribution**:
  - Newcastle Disease: 376 images (5.5%)
  - Cocci virus: [Number] images
  - Salmonella: [Number] images
  - Healthy: [Number] images

**Note**: Dataset exhibits significant class imbalance, which is addressed through data augmentation and class weighting techniques.

## 🛠️ Technical Approach

### Data Preprocessing
- Image resizing (128×128, 150×150, 224×224)
- Pixel normalization (0-1 scaling)
- Comprehensive data augmentation:
  - Rotation (±25°)
  - Width/height shifting (20%)
  - Zoom (20%)
  - Horizontal flipping
  - Brightness adjustment

### Model Architectures
Three systematic approaches were implemented:

1. **Simple CNN** (Baseline)
   - 3 convolutional layers with increasing filters (32, 64, 128)
   - Batch normalization and max-pooling
   - Fully connected layers with dropout

2. **Deeper CNN**
   - 6 convolutional layers in 3 blocks
   - Enhanced feature extraction capability
   - Advanced regularization

3. **Transfer Learning**
   - EfficientNetB0 (Primary)
   - VGG16
   - ResNet50
   - Pre-trained on ImageNet, fine-tuned for poultry diseases

### Training Strategy
- **Optimizer**: Adam with learning rates (0.001, 0.0001, 0.00001)
- **Regularization**: Dropout (0.3-0.5), L2 regularization
- **Class Imbalance Handling**: Computed class weights
- **Callbacks**: Early stopping, model checkpointing, learning rate reduction



### Key Findings
- **Transfer learning superiority**: Pre-trained models significantly outperformed custom CNNs
- **EfficientNetB0**: Achieved best balance of accuracy (94%) and computational efficiency
- **Generalization**: Minimal gap between training and validation accuracy indicates good generalization
- **Clinical relevance**: Confusion patterns align with real diagnostic challenges

##  Installation & Setup

### Prerequisites
```bash
Python 3.8+
TensorFlow 2.8+
scikit-learn
matplotlib
seaborn
pandas
numpy
