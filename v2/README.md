# Class-Conditional Diffusion Model for CIFAR Cat and Dog Generation

This repository implements a class-conditional diffusion model for generating high-quality cat and dog images from the CIFAR-10 dataset. It combines an autoencoder architecture with a diffusion model to learn a structured latent space and generate realistic images.

## Features

- **Class-Conditional Generation**: Generate cat or dog images by conditioning the diffusion process
- **RGB Latent Space**: Uses a 3-channel latent space for better image quality
- **Attention Mechanisms**: Implements channel attention layers for improved feature selection
- **Center Loss**: Enhances class separation in the latent space
- **Visualization Tools**:
  - Latent space visualization using PCA and t-SNE
  - Reconstruction quality visualization
  - Diffusion process animation
  - Denoising path visualization in latent space

## Requirements

- Python 3.6+
- PyTorch 1.7+
- torchvision
- numpy
- matplotlib
- tqdm
- scikit-learn
- imageio

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/cifar-cat-dog-diffusion.git
cd cifar-cat-dog-diffusion

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training the Model

```python
# Run the entire pipeline (autoencoder training + diffusion model training)
python main.py
```

The training process:
1. Trains an autoencoder with class separation capabilities
2. Trains a conditional diffusion model on the learned latent space
3. Generates visualizations at regular intervals
4. Saves models and visualizations to the `./cifar_cat_dog_conditional_v2` directory

### Using Pre-trained Models

```python
import torch
from model import SimpleAutoencoder, ConditionalUNet, ConditionalDenoiseDiffusion

# Load models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
autoencoder = SimpleAutoencoder().to(device)
autoencoder.load_state_dict(torch.load("cifar_cat_dog_autoencoder.pt"))
autoencoder.eval()

unet = ConditionalUNet().to(device)
unet.load_state_dict(torch.load("conditional_diffusion_final.pt"))
diffusion = ConditionalDenoiseDiffusion(unet, n_steps=1000, device=device)

# Generate cat samples (class_idx=0)
from utils import generate_class_samples
cat_samples = generate_class_samples(autoencoder, diffusion, target_class=0, num_samples=5)

# Generate dog samples (class_idx=1)
dog_samples = generate_class_samples(autoencoder, diffusion, target_class=1, num_samples=5)
```

## Model Architecture

### Autoencoder

- **Encoder**: Convolutional layers with channel attention mechanisms
- **Decoder**: Transposed convolutions to reconstruct original images
- **Classifier**: MLP head for class prediction
- **Center Loss**: Helps create more separated clusters in latent space

### Diffusion Model

- **UNet Architecture**: Class-conditional UNet with attention blocks
- **Time Embedding**: Sinusoidal time embeddings for diffusion timesteps
- **Class Embedding**: Embedding layer for class conditioning
- **Residual Blocks**: Combines time and class embeddings for conditional generation

## Visualizations

The model generates several types of visualizations:

1. **Reconstruction Quality**: Original vs. reconstructed images during training
2. **Latent Space**: 2D projection of the latent space showing class separation
3. **Sample Grid**: Generated samples for each class
4. **Denoising Path**: Visualization of the denoising process and latent space trajectory
5. **Diffusion Animation**: GIF animations showing the noise-to-image generation process

## Example Results

After training, you can find these visualization files in the results directory:

- `reconstruction_epoch_X.png`: Shows reconstruction quality at epoch X
- `latent_space_epoch_X.png`: t-SNE visualization of latent space at epoch X
- `diffusion_animation_class_X.gif`: Animation of the diffusion process for class X
- `denoising_path_X_final.png`: Visualization of the denoising trajectory for class X
- `sample_grid_all_classes.png`: Grid of generated samples for all classes

## Customization

You can customize various aspects of the model:

- **Latent Dimensions**: Modify the `hidden_dims` parameter in `ConditionalUNet`
- **Training Parameters**: Adjust learning rates, number of epochs, etc.
- **Diffusion Steps**: Change the `n_steps` parameter in `ConditionalDenoiseDiffusion`
