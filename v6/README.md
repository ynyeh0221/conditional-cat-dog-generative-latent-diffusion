# Conditional VAE-Diffusion Model for CIFAR Cat and Dog Generation

This repository implements a conditional image generation pipeline that combines a Variational Autoencoder (VAE) with a diffusion model to generate high-quality, controllable cat and dog images based on the CIFAR-10 dataset.

## Project Overview

The project implements a two-stage generative model:

1. **VAE Stage**: A variational autoencoder learns to encode images into a compact latent space and decode them back to reconstructed images, while simultaneously learning to classify different categories.

2. **Diffusion Stage**: A conditional diffusion model operates in the latent space of the VAE, learning to denoise random noise vectors into meaningful latent vectors that can be decoded into high-quality images.

## Key Features

- **Class-conditional generation**: Generate specific animals (cats or dogs) on demand
- **Progressive denoising**: Visualize the step-by-step process of image generation
- **Latent space visualization**: Explore the learned representations with t-SNE and PCA projections
- **Attention mechanisms**: Leverages channel attention for improved feature selection
- **Well-structured architecture**: Modular design with encoder, decoder, UNet, and diffusion components

## Requirements

- PyTorch
- torchvision
- matplotlib
- numpy
- tqdm
- scikit-learn (for t-SNE and PCA visualization)
- imageio (for creating GIF animations)

## Model Architecture

### VAE Architecture

- **Encoder**: Convolutional backbone with attention blocks to encode images into a latent distribution
- **Reparameterization Trick**: Samples from the latent distribution in a differentiable way
- **Decoder**: Transforms latent vectors back to image space
- **Classifier**: Classifies latent vectors into cat or dog classes
- **Center Loss**: Encourages better class separation in the latent space

### Diffusion Model Architecture

- **UNet**: MLP-based noise prediction network for the latent space
- **Time Embedding**: Encodes diffusion timesteps as sinusoidal embeddings
- **Class Embedding**: Incorporates class information for conditional generation
- **Denoising Process**: Gradual transformation from random noise to structured latent vectors

## Usage

The main function runs the entire pipeline non-interactively:

```python
python main.py
```

This will:
1. Load and prepare the CIFAR-10 dataset, filtering for cat and dog classes
2. Train the VAE model (or load a pre-trained one if available)
3. Train the diffusion model (or load a pre-trained one if available)
4. Generate visualizations of the model's outputs

## Visualization Capabilities

The model includes several visualization functions:

- `visualize_reconstructions`: Shows original vs. VAE-reconstructed images
- `visualize_latent_space`: Displays t-SNE projections of the latent space
- `visualize_denoising_steps`: Shows the diffusion path from noise to image
- `create_diffusion_animation`: Creates GIF animations of the denoising process
- `generate_samples_grid`: Produces a grid of generated samples for all classes

## Results

After training, you can expect:

- Clear separation between cat and dog representations in the latent space
- High-quality reconstructions of input images
- Realistic generated samples of both cats and dogs
- Smooth denoising animations showing the generation process

 Model Component | Visualization | Description |
|-----------------|---------------|-------------|
| Autoencoder | ![Reconstructions](https://github.com/ynyeh0221/CIFAR10-cat-dog-generative-latent-diffusion/blob/main/v6/output/reconstruction/vae_reconstruction_epoch_300.png) | Original images (top) and their reconstructions (bottom) |
| Latent Space | ![Latent Space](https://github.com/ynyeh0221/CIFAR10-cat-dog-generative-latent-diffusion/blob/main/v6/output/latent_space/vae_latent_space_epoch_300.png) | t-SNE visualization of cat and dog latent representations |
| Class Samples | ![Class Samples](https://github.com/ynyeh0221/CIFAR10-cat-dog-generative-latent-diffusion/blob/main/v6/output/diffusion_sample_result/sample_class_Cat_epoch_800.png)![Class Samples](https://github.com/ynyeh0221/CIFAR10-cat-dog-generative-latent-diffusion/blob/main/v6/output/diffusion_sample_result/sample_class_Dog_epoch_800.png) | Generated samples for cat and dog classes |
| Denoising Process | ![Denoising Cat](https://github.com/ynyeh0221/CIFAR10-cat-dog-generative-latent-diffusion/blob/main/v6/output/denoising_path_Cat_final.png)![Denoising Dog](https://github.com/ynyeh0221/CIFAR10-cat-dog-generative-latent-diffusion/blob/main/v6/output/denoising_path_Dog_final.png) | Visualization of cat generation process and latent path |
| Animation | ![Cat Animation](https://github.com/ynyeh0221/CIFAR10-cat-dog-generative-latent-diffusion/blob/main/v6/diffusion_animation_class_Cat_epoch_800.gif)![Dog Animation](https://github.com/ynyeh0221/CIFAR10-cat-dog-generative-latent-diffusion/blob/main/v6/diffusion_animation_class_Dog_epoch_800.gif) | Animation of the denoising process for cat generation |


## Project Structure

- `SimpleAutoencoder`: The VAE implementation with conditioning capabilities
- `ConditionalUNet`: The noise prediction network for the diffusion model
- `ConditionalDenoiseDiffusion`: The diffusion process handler
- Various visualization and utility functions
