# CIFAR Cat-Dog Conditional Diffusion Model

## Project Overview

This project implements a class-conditional diffusion model for generating cat and dog images based on the CIFAR-10 dataset. It combines an autoencoder with RGB latent space and a conditional denoising diffusion probabilistic model (DDPM) to create high-quality synthetic images with class control.

## Features

- RGB latent space autoencoder with attention modules
- Class-conditional UNet with attention for noise prediction
- Denoising diffusion process with 1000 timesteps
- Extensive visualization tools for model understanding
- Latent space exploration with PCA projections

## Requirements

```
torch
torchvision
matplotlib
numpy
tqdm
scikit-learn
imageio
```

## Model Architecture

### Autoencoder

The autoencoder uses a deep convolutional architecture with Channel Attention Blocks (CABs) to learn an RGB latent representation:

- **Encoder**: Converts 32×32×3 images to 8×8×3 RGB latent codes
- **Decoder**: Reconstructs original images from latent codes
- **Attention**: Uses Channel Attention Layers (CALs) to enhance feature selection

### Diffusion Model

The diffusion model is conditional, allowing class-specific generation:

- **UNet**: Deep conditional UNet with residual blocks and attention
- **Time Embedding**: Sinusoidal embeddings to condition on diffusion timestep
- **Class Embedding**: Learnable embeddings for class conditioning
- **Forward Process**: Gradually adds noise according to a schedule
- **Reverse Process**: Learns to denoise images step by step with class guidance

## Training Process

The training happens in two phases:

1. **Autoencoder Training**: Learn to compress images to RGB latent space and reconstruct them
2. **Diffusion Model Training**: Train noise predictor conditioned on class labels

## Results

| Model Component | Visualization | Description |
|-----------------|---------------|-------------|
| Autoencoder | ![Reconstructions](https://github.com/ynyeh0221/CIFAR10-cat-dog-generative-latent-diffusion/blob/main/v1/output/reconstruction/reconstruction_epoch_100.png) | Original images (top) and their reconstructions (bottom) |
| Latent Space | ![Latent Space](https://github.com/ynyeh0221/CIFAR10-cat-dog-generative-latent-diffusion/blob/main/v1/output/latent_space/latent_space_epoch_100.png) | t-SNE visualization of cat and dog latent representations |
| Class Samples | ![Class Samples](https://github.com/ynyeh0221/CIFAR10-cat-dog-generative-latent-diffusion/blob/main/v1/output/diffusion_result_sample/sample_class_Cat_epoch_100.png)![Class Samples](https://github.com/ynyeh0221/CIFAR10-cat-dog-generative-latent-diffusion/blob/main/v1/output/diffusion_result_sample/sample_class_Dog_epoch_100.png) | Generated samples for cat and dog classes |
| Denoising Process | ![Denoising Cat](https://github.com/ynyeh0221/CIFAR10-cat-dog-generative-latent-diffusion/blob/main/v1/output/denosing_path/denoising_path_Cat_epoch_100.png)![Denoising Dog](https://github.com/ynyeh0221/CIFAR10-cat-dog-generative-latent-diffusion/blob/main/v1/output/denosing_path/denoising_path_Dog_epoch_100.png) | Visualization of cat generation process and latent path |
| Animation | ![Cat Animation](https://github.com/ynyeh0221/CIFAR10-cat-dog-generative-latent-diffusion/blob/main/v1/diffusion_animation_class_Cat_epoch_100.gif)![Dog Animation](https://github.com/ynyeh0221/CIFAR10-cat-dog-generative-latent-diffusion/blob/main/v1/diffusion_animation_class_Dog_epoch_100.gif) | Animation of the denoising process for cat generation |

## Usage

### Training from Scratch

```python
python cifar_cat_dog_diffusion.py
```

The script will:
1. Download and prepare the CIFAR-10 dataset (extracting cats and dogs)
2. Train the autoencoder if no saved model exists
3. Train the diffusion model if no saved model exists
4. Generate visualizations in the `cifar_cat_dog_conditional` directory

### Using Pre-trained Models

The script automatically loads pre-trained models if they exist at:
- `./cifar_cat_dog_conditional/cifar_cat_dog_autoencoder.pt`
- `./cifar_cat_dog_conditional/conditional_diffusion_final.pt`

## Key Components Explanation

### RGB Latent Space

Unlike typical VAEs or diffusion models, this implementation uses an RGB latent space (3×8×8), making the latent variables inherently visual and interpretable.

### Attention Mechanisms

- **Channel Attention Layers**: Focus on important feature channels
- **UNet Attention Blocks**: Enable long-range dependencies in the diffusion model

### Visualization Tools

The project includes extensive visualization capabilities:
- Autoencoder reconstructions
- t-SNE/PCA projections of latent space
- Denoising process visualization with latent space paths
- Animation of diffusion process
- Class-conditional sample grids

## Notes

- Training the full model requires significant computational resources
- The models use Euclidean distance loss rather than traditional MSE
- The diffusion process uses 1000 timesteps for high-quality generation

## Future Work

- Extend to more CIFAR-10 classes
- Implement classifier-free guidance for improved control
- Add interpolation in latent space between classes
- Explore higher resolution generation
