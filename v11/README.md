# STL10 Cat/Dog Conditional VAE-Diffusion Model

A deep learning project that combines Variational Autoencoders (VAE), Generative Adversarial Networks (GAN), and Conditional Diffusion models to generate high-quality images of cats and dogs based on the STL10 dataset.

## Overview

This project implements a class-conditional image generation system using a combination of state-of-the-art deep learning techniques:

1. **VAE-GAN Architecture**: A hybrid model that combines the reconstruction capabilities of VAEs with the sharpness and realism of GANs
2. **Conditional Diffusion Model**: Operating in the latent space of the VAE to generate high-quality, diverse images conditioned on class labels
3. **Perceptual Loss**: Using VGG16 features to enhance the quality of reconstructed images

The system is trained on a binary classification subset of the STL10 dataset, focusing only on cat and dog classes.

## Features

- **Enhanced VAE Architecture**: Using residual blocks, attention mechanisms, and layer normalization for better image quality
- **Comprehensive Loss Functions**: Combining reconstruction, KL divergence, perceptual, GAN, and center losses
- **Class-Conditional Generation**: Generate images from a specific class (cat or dog) with high fidelity
- **Visualization Tools**: Includes utilities to visualize the latent space, denoising process, and generation quality

## Model Architecture

### SimpleAutoencoder (VAE-GAN)
- **Encoder**: Downsamples 48×48 images through 3 stages to a 6×6 feature map, then to a latent vector
- **Decoder**: Reconstructs images from the latent space with transposed convolutions
- **Classifier**: Additional network to predict class labels from latent vectors
- **Discriminator**: Classic GAN discriminator to enhance image quality

### ConditionalUNet
- Operates in the latent space of the VAE
- Includes time and class embeddings for conditional generation
- Leverages attention mechanisms and residual connections for stable training

## Requirements

- PyTorch
- torchvision
- matplotlib
- numpy
- scikit-learn
- tqdm
- imageio

## Usage

### Training the VAE-GAN

```python
# First train the VAE-GAN model
autoencoder = SimpleAutoencoder(in_channels=3, latent_dim=256, num_classes=2).to(device)
autoencoder, discriminator, ae_losses = train_autoencoder(
    autoencoder,
    train_loader,
    num_epochs=300,
    lr=5e-4,
    lambda_cls=0.3,
    lambda_center=0.1,
    lambda_vgg=0.4,
    visualize_every=10,
    save_dir="./results"
)
```

### Training the Diffusion Model

```python
# Then train the diffusion model using the frozen VAE
conditional_unet = ConditionalUNet(
    latent_dim=256,
    hidden_dims=[256, 512, 1024, 512, 256],
    time_emb_dim=256,
    num_classes=2
).to(device)

conditional_unet, diffusion, diff_losses = train_conditional_diffusion(
    autoencoder, 
    conditional_unet, 
    train_loader, 
    num_epochs=100, 
    lr=1e-3,
    visualize_every=10,
    save_dir="./results",
    device=device
)
```

### Generating Images

```python
# Generate 5 cat images (class index 0)
cat_samples = generate_class_samples(
    autoencoder, 
    diffusion, 
    target_class=0,  # 0 for cat, 1 for dog
    num_samples=5, 
    save_path="generated_cats.png"
)

# Create a diffusion animation showing the denoising process
animation_path = create_diffusion_animation(
    autoencoder, 
    diffusion, 
    class_idx=0,  # 0 for cat, 1 for dog
    num_frames=50, 
    fps=15,
    save_path="cat_animation.gif"
)
```

## Visualization Examples

The project includes multiple visualization capabilities:

1. **Reconstruction Visualization**: Compare original images with their VAE reconstructions
2. **Latent Space Visualization**: t-SNE or PCA projection of the latent space
3. **Denoising Process Visualization**: Track how noisy latents become clear images
4. **Sample Grid Generation**: Create a grid of generated samples from each class
5. **Diffusion Animation**: Animate the denoising process from random noise to a clear image

## Customization

- Modify the `class_names` list to work with different classes
- Adjust the architecture size by changing the dimensions in the model initializations
- Tune hyperparameters like learning rates and loss weights to balance different aspects of training

## License

This project is provided for educational and research purposes.

## Acknowledgments

- The STL10 dataset is used for training
- Architecture design inspired by modern VAE, GAN, and Diffusion model research
- VGG16 is used for perceptual loss computation
