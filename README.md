# MobileNetV3 on CIFAR-10 with ChannelMixAug

## Overview
This project implements MobileNetV3-Small for image classification on the CIFAR-10 dataset. A custom feature-space augmentation, ChannelMixAug, is introduced and compared to the baseline model. The goal is to evaluate whether this augmentation can improve accuracy with minimal computational overhead.

## Dataset
- **CIFAR-10:** A benchmark dataset for image classification, consisting of 60,000 32x32 color images in 10 classes. The dataset is automatically downloaded by the notebook.

## Requirements
Install the necessary dependencies using:
```bash
pip3 install torch torchvision numpy matplotlib tqdm
```
- Python 3.8 or newer
- PyTorch
- torchvision
- numpy
- matplotlib
- tqdm

## Model Architecture
- **MobileNetV3-Small:** Efficient neural network architecture for mobile and edge devices, combining depthwise separable convolutions, SE blocks, and h-swish activations.
- **ChannelMixAug:** Custom feature-space augmentation that mixes channel groups between samples in a batch during training.

## Code Structure
- `FinalProject.ipynb`: Main Jupyter Notebook containing all code for data loading, model setup, training, augmentation, and evaluation.

## Training the Model
1. Open `FinalProject.ipynb` in Jupyter or VS Code.
2. Run all cells to train both the baseline and augmented models.
3. Training is set to 1 epochs for demonstration. You can increase or decrease epochs as needed.

## Output Verification
- The notebook prints model parameter counts, test accuracy, and training time for both models.
- Example results:

| Model         | Parameters | Test Accuracy | Time  |
|--------------|------------|---------------|-------|
| Baseline      | 1,528,106  | 9.25%         | 6.3 s |
| ChannelMixAug | 1,528,106  | 10.15%        | 6.1 s |


## References
- [Searching for MobileNetV3 (arXiv, 2019)](https://arxiv.org/abs/1905.02244)
- [TorchVision MobileNetV3](https://github.com/pytorch/vision/tree/main/torchvision/models)
- [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

## Contributors
- Mitul Patel
- Vrunda Panchal

## Contact
For any questions, please contact Mitul Patel or Vrunda Panchal.
