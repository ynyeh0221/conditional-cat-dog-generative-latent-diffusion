# CIFAR-10 Cat/Dog Conditional Diffusion Model

A PyTorch implementation of a class-conditional diffusion model with an advanced VAE architecture for generating high-quality cat and dog images from the CIFAR-10 dataset.

## Overview

This project implements a sophisticated generative pipeline combining:

1. A Variational Autoencoder (VAE) with enhanced architecture features including:
   - Channel and spatial attention mechanisms
   - Residual connections
   - Layer normalization
   - Center loss for improved latent space clustering
   - Perceptual (VGG) loss for better visual quality
   - GAN component for sharper details

2. A conditional diffusion model operating in the latent space that:
   - Learns class-conditional noise prediction
   - Uses attention mechanisms in both spatial and temporal dimensions
   - Gradually denoises random latent vectors to produce class-specific samples

The model demonstrates how to effectively combine diffusion models with VAEs to generate high-quality, class-conditional images, even from the relatively low-resolution CIFAR-10 dataset.

## Features

- **Dual-stage generative pipeline**: VAE encoding/decoding + latent space diffusion
- **Advanced neural network components**:
  - Channel Attention (CA) and Spatial Attention (SA) modules
  - Swish activation functions
  - Layer normalization for improved stability
  - Multi-head self-attention in the diffusion model
- **Sophisticated loss components**:
  - VGG perceptual loss for visual quality
  - Center loss for better latent space organization
  - KL divergence for latent space regularization
  - GAN adversarial loss for sharper details
- **Visualization tools**:
  - t-SNE latent space visualization
  - Animated diffusion process
  - Class-specific sample generation
  - Denoising path visualization

## Architecture

### Autoencoder Components

- **Encoder**: Convolutional network with downsampling blocks and residual connections
- **Decoder**: Transposed convolutional network with upsampling blocks
- **Classifier**: MLP for class prediction from latent vectors
- **Discriminator**: Convolutional network for adversarial training

### Diffusion Model Components

- **UNet**: Class-conditional, time-conditional network for noise prediction
- **Time Embedding**: Sinusoidal position encoding for timestep information
- **Class Embedding**: Learnable embeddings for conditioning on cat vs dog classes
- **Attention Blocks**: Multi-head self-attention for improved generation quality

## Requirements

- Python 3.7+
- PyTorch 1.9+
- torchvision
- numpy
- matplotlib
- scikit-learn
- tqdm
- imageio

## Installation

```bash
git clone https://github.com/yourusername/cifar10-catdog-diffusion.git
cd cifar10-catdog-diffusion
pip install -r requirements.txt
```

## Usage

### Training the Models

The training process has two main stages:

1. First, train the VAE:

```python
python main.py --train_vae --epochs 1200
```

2. Then, train the diffusion model:

```python
python main.py --train_diffusion --epochs 800
```

You can also train both in sequence:

```python
python main.py --train_all --vae_epochs 1200 --diffusion_epochs 800
```

### Generating Images

To generate class-conditional samples:

```python
python generate.py --class_name dog --num_samples 10
```

To create a diffusion animation:

```python
python generate.py --create_animation --class_name cat
```

### Visualizing Results

To visualize the latent space and samples:

```python
python visualize.py --mode latent_space
python visualize.py --mode samples_grid
python visualize.py --mode denoising_steps --class_name dog
```

## Results

The model produces high-quality 32x32 images of cats and dogs from the CIFAR-10 dataset. Sample outputs and visualizations are stored in the `./cifar10_catdog_conditional_improved` directory.

Example visualizations include:
- VAE reconstructions
- Latent space organization
- Generated samples
- Denoising process animations
- Diffusion paths in latent space

 Model Component | Visualization | Description |
|-----------------|---------------|-------------|
| Autoencoder | ![Reconstructions](https://github.com/ynyeh0221/CIFAR10-cat-dog-generative-latent-diffusion/blob/main/v10/output/reconstruction/vae_reconstruction_epoch_800.png) | Original images (top) and their reconstructions (bottom) |
| Latent Space | ![Latent Space](https://github.com/ynyeh0221/CIFAR10-cat-dog-generative-latent-diffusion/blob/main/v10/output/latent_space/vae_latent_space_epoch_800.png) | t-SNE visualization of cat and dog latent representations |
| Class Samples | ![Class Samples](https://github.com/ynyeh0221/CIFAR10-cat-dog-generative-latent-diffusion/blob/main/v10/output/diffusion_sample_result/sample_class_cat_epoch_800.png)![Class Samples](https://github.com/ynyeh0221/CIFAR10-cat-dog-generative-latent-diffusion/blob/main/v10/output/diffusion_sample_result/sample_class_dog_epoch_800.png) | Generated samples for cat and dog classes |
| Denoising Process | ![Denoising Cat](https://github.com/ynyeh0221/CIFAR10-cat-dog-generative-latent-diffusion/blob/main/v10/output/diffusion_path/denoising_path_cat_epoch_800.png)![Denoising Dog](https://github.com/ynyeh0221/CIFAR10-cat-dog-generative-latent-diffusion/blob/main/v10/output/diffusion_path/denoising_path_dog_epoch_800.png) | Visualization of cat generation process and latent path |
| Animation | ![Cat Animation](https://github.com/ynyeh0221/CIFAR10-cat-dog-generative-latent-diffusion/blob/main/v10/output/diffusion_animination/diffusion_animation_cat_epoch_800.gif)![Dog Animation](https://github.com/ynyeh0221/CIFAR10-cat-dog-generative-latent-diffusion/blob/main/v10/output/diffusion_animination/diffusion_animation_dog_epoch_800.gif) | Animation of the denoising process for cat generation |

## Model Architecture Details

### VAE Enhancements

- **Channel Attention**: Recalibrates channel-wise feature responses
- **Spatial Attention**: Focuses on important spatial regions
- **Residual Connections**: Enable deeper networks with better gradient flow
- **Layer Normalization**: Stabilizes training across deeper architectures
- **Center Loss**: Explicitly pushes latent codes toward class centers

### Diffusion Process

The diffusion model operates by:
1. Progressively adding noise to latent vectors over N timesteps
2. Training a neural network to reverse this process
3. Starting from random noise and iteratively denoising to produce a class-specific sample
4. Projecting the final denoised latent vector through the VAE decoder


## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- The CIFAR-10 dataset is provided by the Canadian Institute for Advanced Research
- This implementation builds on concepts from:
  - "Denoising Diffusion Probabilistic Models" (Ho et al., 2020)
  - "Diffusion Models Beat GANs on Image Synthesis" (Dhariwal & Nichol, 2021)
  - "Learning Transferable Visual Models From Natural Language Supervision" (Radford et al., 2021)
