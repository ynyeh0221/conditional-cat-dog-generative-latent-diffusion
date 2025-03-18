# Class-Conditional Diffusion Model for CIFAR Cats and Dogs

## Overview

This project implements a class-conditional diffusion model for generating high-quality cat and dog images based on the CIFAR-10 dataset. The implementation combines a variational autoencoder (VAE) with a conditional diffusion model, allowing controlled generation of specific animal classes with smooth transitions between latent space representations.

## Features

- **Dual-Stage Architecture**: Combines VAE and conditional diffusion model for enhanced image generation
- **Class-Conditional Generation**: Generates images of a specific class (cat or dog) on demand
- **Attention Mechanisms**: Incorporates channel attention layers for improved feature selection
- **Rich Visualizations**: Includes tools for visualizing the latent space, denoising process, and generation steps
- **Animation Support**: Creates GIF animations showing the diffusion process from noise to image

## Requirements

- Python 3.6+
- PyTorch 1.8+
- torchvision
- matplotlib
- numpy
- tqdm
- scikit-learn
- imageio

## Installation

```bash
# Clone this repository
git clone https://github.com/yourusername/cifar-diffusion.git
cd cifar-diffusion

# Install required packages
pip install torch torchvision matplotlib numpy tqdm scikit-learn imageio
```

## Usage

### Training the Model

```python
# Import the main script
import torch
from main import main

# Run the full training pipeline
main()
```

This will:
1. Download and process the CIFAR-10 dataset
2. Train the VAE if no saved model is found
3. Train the diffusion model if no saved model is found
4. Generate sample visualizations

### Generating Images

```python
# Load pretrained models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae = VariationalAutoencoder().to(device)
vae.load_state_dict(torch.load("cifar_cat_dog_conditional_v3/cifar_cat_dog_autoencoder.pt"))

unet = ConditionalUNet(in_channels=3, hidden_dims=[32, 64, 128], num_classes=2).to(device)
unet.load_state_dict(torch.load("cifar_cat_dog_conditional_v3/conditional_diffusion_final.pt"))
diffusion = ConditionalDenoiseDiffusion(unet, n_steps=1000, device=device)

# Generate samples (0 for cat, 1 for dog)
cat_samples = generate_class_samples(vae, diffusion, target_class=0, num_samples=5, 
                                    save_path="cat_samples.png")
```

## Model Architecture

The implementation consists of two main components:

### 1. Variational Autoencoder (VAE)

- **Encoder**: Convolutional layers with channel attention blocks
- **Latent Space**: RGB latent space (3 channels, 8x8 spatial dimension)
- **Decoder**: Upsampling convolutional layers with channel attention
- **Additional Features**: 
  - Class prediction from latent space
  - Center loss for improved class separation
  - KL divergence regularization

### 2. Conditional Diffusion Model

- **Architecture**: UNet with attention blocks and residual connections
- **Conditioning**: Class embedding added to time embedding
- **Time Embedding**: Sinusoidal embeddings processed through MLP
- **Class Embedding**: Embedding layer processed through MLP
- **Noise Prediction**: UNet predicts noise at each timestep, conditioned on class

## Visualization Tools

The project includes several visualization tools:

- `visualize_reconstructions`: Shows original vs. reconstructed images
- `visualize_latent_space`: Plots the t-SNE visualization of the latent space
- `visualize_denoising_steps`: Shows the denoising process and latent space path
- `generate_samples_grid`: Creates a grid of samples for all classes
- `create_diffusion_animation`: Creates a GIF animation of the diffusion process

## Results

After training, the model produces high-quality, diverse cat and dog images with distinctive class features. The denoising visualizations show how the latent space paths move toward the target class regions during the generation process.
