# Class-Conditional VAE-Diffusion Model for STL10 Cat/Dog Image Generation

This repository contains an implementation of a Class-Conditional Variational Autoencoder (VAE) with Diffusion Model for generating high-quality cat and dog images from the STL10 dataset.

## Project Overview

This project implements a state-of-the-art generative model that combines the strengths of VAEs and diffusion models for conditional image generation. The model is trained specifically on the cat and dog classes from the STL10 dataset, with images resized from 96×96 to 48×48 pixels.

## Model Architecture

The implementation consists of two main components:

### 1. VAE with Perceptual Loss and GAN Enhancement

- **Encoder**: A convolutional neural network with residual blocks, channel and spatial attention, and layer normalization
- **Decoder**: A mirrored architecture with transposed convolutions for upsampling
- **Discriminator**: For adversarial training to improve perceptual quality
- **Perceptual Loss**: Using VGG16 features to capture high-level semantic information
- **Classifier**: For class conditioning in the latent space

### 2. Conditional Diffusion Model

- **UNet Architecture**: Operating on the 2D reshape of the VAE latent space
- **Time Embedding**: For conditioning on the noise level
- **Class Embedding**: For conditioning on the target class (cat or dog)
- **Conditional Denoising Process**: Gradually removing noise from randomly sampled latent vectors

## Key Features

- **Dual-layer Generative Process**: VAE for learning a structured latent space, diffusion model for high-quality sampling
- **Multi-level Conditioning**: Time step and class conditioning at each stage of the diffusion process
- **Advanced Regularization**: Center loss, KL divergence, and adversarial loss for better latent space structure
- **Progressive Training Strategy**: Gradually introducing different loss components
- **Comprehensive Visualization Tools**: For both VAE reconstructions and diffusion sampling

## Requirements

```
torch>=1.8.0
torchvision>=0.9.0
matplotlib
numpy
scikit-learn
tqdm
imageio
```

## Usage

### Training

```python
# Train the VAE
python main.py --mode vae --epochs 10000 --lr 1e-3 --lambda_vgg 0.8 --save_dir ./results

# Train the diffusion model using a pre-trained VAE
python main.py --mode diffusion --vae_path ./results/vae_gan_best.pt --epochs 2000 --lr 1e-3 --save_dir ./results
```

### Generation

```python
# Generate cat images
python generate.py --class_name cat --num_samples 10 --model_path ./results/conditional_diffusion_final.pt

# Generate dog images
python generate.py --class_name dog --num_samples 10 --model_path ./results/conditional_diffusion_final.pt
```

## Results

The model can generate high-quality, diverse images of cats and dogs. Some notable features:

- Clear separation of classes in the latent space
- Smooth transitions in the diffusion process
- High-fidelity structural details in generated images
- Preservation of class-specific features

## Visualization Tools

The codebase includes tools for visualizing:

- VAE reconstructions and latent space embeddings
- The denoising process as animated GIFs
- Latent space paths during the diffusion process
- Comparisons between original images, VAE reconstructions, and diffusion-generated samples

Model Component | Visualization | Description |
|-----------------|---------------|-------------|
| GAN-VAE | ![Reconstructions](https://github.com/ynyeh0221/CIFAR10-cat-dog-generative-latent-diffusion/blob/main/v12/output/reconstruction/vae_reconstruction_epoch_10000.png) | Original images (top) and their reconstructions (bottom) |
| Latent Space | ![Latent Space](https://github.com/ynyeh0221/CIFAR10-cat-dog-generative-latent-diffusion/blob/main/v12/output/latent_space/vae_latent_space_epoch_10000.png) | t-SNE visualization of cat and dog latent representations |
| Original image v.s. Reconstruction v.s. Reconstruction from noised/denoised latent space | ![](https://github.com/ynyeh0221/CIFAR10-cat-dog-generative-latent-diffusion/blob/main/v12/output/diffusion_latent_space_comparison/latent_comparison_epoch_6700.png) | First row is the reconstructed image from original image's latent space. Second row is the reconstructed image from original image's latent space with noising/denoising. Third row is the original image |
| Class Samples | ![Class Samples](https://github.com/ynyeh0221/CIFAR10-cat-dog-generative-latent-diffusion/blob/main/v12/output/diffusion_sample_result/sample_class_cat_epoch_6700.png)![Class Samples](https://github.com/ynyeh0221/CIFAR10-cat-dog-generative-latent-diffusion/blob/main/v12/output/diffusion_sample_result/sample_class_dog_epoch_6700.png) | Generated samples for cat and dog classes |
| Denoising Process | ![Denoising Cat](https://github.com/ynyeh0221/CIFAR10-cat-dog-generative-latent-diffusion/blob/main/v11/output/diffusion_path/denoising_path_cat_epoch_6700.png)![Denoising Dog](https://github.com/ynyeh0221/CIFAR10-cat-dog-generative-latent-diffusion/blob/main/v12/output/diffusion_path/denoising_path_dog_epoch_6700.png) | Visualization of cat generation process and latent path |
| Animation | ![Cat Animation](https://github.com/ynyeh0221/CIFAR10-cat-dog-generative-latent-diffusion/blob/main/v12/animination/diffusion_animation_cat_epoch_6700.gif)![Dog Animation](https://github.com/ynyeh0221/CIFAR10-cat-dog-generative-latent-diffusion/blob/main/v12/animination/diffusion_animation_dog_epoch_6700.gif) | Animation of the denoising process for cat generation |

## Model Weights

Pre-trained model weights are available:
- VAE: `results/vae_gan_final.pt`
- Diffusion Model: `results/conditional_diffusion_final.pt`

## Acknowledgments

- The STL10 dataset providers
- PyTorch and torchvision libraries
- NVIDIA for GPU computing resources
