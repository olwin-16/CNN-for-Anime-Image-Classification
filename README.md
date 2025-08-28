# Convolutional Neural Network (CNN) for Anime Image Classification

<br>

<img width="982" height="495" alt="image" src="https://github.com/user-attachments/assets/91b96ec8-0d29-4f31-9ad3-3d89824697b2" />

<br>

## Project Overview

This project develops and trains a Convolutional Neural Network (CNN) to classify anime images into specific character categories. The objective is to design a model that can analyze visual patterns in anime images and accurately identify different characters or styles, thus advancing skills in image classification and CNN architectures using PyTorch

## Project Structure

- **Main Script**: anime_image_cnn.py containing data loading, preprocessing, CNN model definition, training loop, evaluation, and visualization utilities.
- **Dataset Handling**: Downloads and unpacks a subset of the AniWho dataset containing images of two character classes (anastasia and takao).
- **Custom Dataset Class**: Efficiently loads images and applies transformations.
- **Train/Validation Split**: Splits dataset 80/20 with DataLoader objects for batching.
- **CNN model**: Two convolutional layers with pooling followed by two fully connected layers for classification.
- **Training**: Trains model with CrossEntropyLoss and Adam optimizer across 5 epochs, visualizing training and validation loss curves.

## Dataset Details

- **Source**: Dataset described in the paper [AniWho: A Quick and Accurate Way to Classify Anime Character Faces in Images](https://arxiv.org/pdf/2208.11012v3)
- **Contents**: Subset contains 100 images (50 per class) of two characters: Anastasia and Takao.
- **Format**: RGB images resized to 64x64 pixels.
- **License**: Dataset usage subject to original authors’ terms (see linked paper).

## CNN Architecture

```bash
import torch.nn as nn
import torch.nn.functional as F

class AnimeCNN(nn.Module):
    def __init__(self):
        super(AnimeCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 2)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = AnimeCNN()
print(model)
```

- Two convolutional layers with ReLU and max pooling.
- Flattened features fed into two fully connected layers
- Output layer with 2 neurons corresponding to class scores.

## Training and Evaluation

- **Loss Function**: CrossEntropyLoss
- **Optimizer**: Adam with learning rate 0.001
- **Epochs**: 5
- **Batch Size**: 8 for training, 20 for validation
  
Training includes monitoring and printing losses for both training and validation datasets.

## Training Output Summary

```bash
Epoch 1, Train Loss: 0.7333, Val Loss: 0.1828
Epoch 2, Train Loss: 0.1857, Val Loss: 0.0011
Epoch 3, Train Loss: 0.0471, Val Loss: 0.0005
Epoch 4, Train Loss: 0.0086, Val Loss: 0.0003
Epoch 5, Train Loss: 0.0009, Val Loss: 0.0002
Finished Training
```

- Rapid decrease in loss indicates effective learning and convergence.
- Near-zero validation loss demonstrates excellent generalization on unseen data.

## Loss Curves Visualization

A plot of the training and validation loss curves is generated to illustrate model performance over epochs.

<img width="846" height="470" alt="image" src="https://github.com/user-attachments/assets/08d3ae32-d1ee-478b-bc6e-1096e43a2039" />

## Setup & Installation

## Clone the repository:

```bash
git clone https://github.com/olwin-16/CNN_for_Anime_Image_Classification.git
cd CNN_for_Anime_Image_Classification
```

## Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the training script:

```bash
python anime_image_cnn.py
```

## requirements.txt
```bash
pandas==2.2.2
numpy==1.26.4
matplotlib==3.8.0
scikit-learn==1.5.0
torch==2.3.1
torchvision==0.18.1
Pillow
requests
```

## License

Refer to the original dataset license in the [AniWho](https://arxiv.org/pdf/2208.11012v3) paper.
Code is provided under the [MIT License](LICENSE)

## Contact

Please open an issue or contact via [Email](mailto:olwinchristian1626@gmail.com) for questions or contributions.
