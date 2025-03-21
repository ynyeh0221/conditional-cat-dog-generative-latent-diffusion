# Class-Conditional Diffusion Model for CIFAR Cat and Dog Generation

This project implements a class-conditional diffusion model specifically trained to generate cat and dog images from the CIFAR-10 dataset. The model uses a two-stage approach with an autoencoder to compress images into a latent space, followed by a diffusion model that operates in this latent space.

## Overview

The model pipeline consists of:

1. **Autoencoder** - Compresses CIFAR-10 images (32x32 pixels) into a flat latent space of dimension 128.
2. **Class-Conditional Diffusion Model** - Learns to generate latent vectors conditioned on class labels (cat or dog).

The architecture includes several advanced components:

- **Channel Attention Layers (CAL)** for improved feature selection
- **Convolutional Attention Blocks (CAB)** for better information flow
- **Center Loss** to improve class separation in the latent space
- **Conditional UNet** for noise prediction in the diffusion process

## Requirements

- Python 3.7+
- PyTorch 1.7+
- torchvision
- matplotlib
- numpy
- tqdm
- scikit-learn
- imageio

## Installation

```bash
pip install torch torchvision matplotlib numpy tqdm scikit-learn imageio
```

## Usage

### Training the Model

```python
python main.py
```

This will:
1. Download the CIFAR-10 dataset
2. Filter for only cat and dog classes
3. Train the autoencoder (if not already trained)
4. Train the diffusion model (if not already trained)
5. Generate visualizations

### Visualizations

The model generates several types of visualizations:

- **Reconstruction Quality** - Shows original vs. reconstructed images
- **Latent Space Visualization** - t-SNE plot of the latent space, showing class separation
- **Sample Grid** - Generated samples for each class
- **Denoising Animations** - GIF animations showing the progressive denoising process
- **Denoising Path Visualization** - Shows both the denoising process and the corresponding path in latent space

## Model Architecture

### Autoencoder

The autoencoder consists of:

- **Encoder**: Convolutional neural network with channel attention that maps 32x32 RGB images to 128-dimensional vectors
- **Decoder**: Deconvolutional network that reconstructs images from the latent space
- **Classifier**: MLP that predicts the class (cat or dog) from the latent representation

### Conditional Diffusion Model

The diffusion model operates in the latent space and consists of:

- **ConditionalUNet**: Neural network that predicts noise in the diffusion process
- **TimeEmbedding**: Converts diffusion timesteps into embeddings
- **ClassEmbedding**: Provides class conditioning information

## Key Functions

- `create_cat_dog_dataset`: Filters CIFAR-10 to include only cats and dogs
- `visualize_reconstructions`: Shows autoencoder reconstruction quality
- `visualize_latent_space`: Creates t-SNE visualizations of the latent space
- `generate_samples_grid`: Creates a grid of generated samples for all classes
- `visualize_denoising_steps`: Shows the denoising process and latent space path
- `create_diffusion_animation`: Creates GIF animations of the diffusion process

## Results

The model can generate high-quality cat and dog images that capture the essence of each class. The diffusion process clearly shows how random noise gradually transforms into coherent images following the features learned from the training data.

## Customization

You can customize:

- Latent space dimension (`latent_dim` parameter in `SimpleAutoencoder`)
- Diffusion steps (`n_steps` parameter in `ConditionalDenoiseDiffusion`)
- Model architecture (number of layers, hidden dimensions, etc.)
