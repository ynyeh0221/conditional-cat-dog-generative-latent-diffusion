# CIFAR Cat-Dog Conditional VAE-Diffusion Model

This project implements a class-conditional generative model that combines Variational Autoencoders (VAE) and Diffusion Models to generate high-quality images of cats and dogs from the CIFAR-10 dataset.

## Overview

This implementation features:

- A VAE with an advanced architecture incorporating residual blocks, attention mechanisms, and skip connections
- A conditional diffusion model that operates in the VAE's latent space
- Training pipeline with progressive training strategies and loss balancing
- Visualization tools to monitor the latent space, denoising process, and generation quality

## Model Architecture

### Variational Autoencoder (VAE)

The VAE consists of several key components:

- **Encoder**: Converts input images to a latent distribution (mu and logvar)
  - Uses residual blocks with channel and spatial attention
  - Progressive downsampling with skip connections
  - Outputs latent vectors of dimension 256

- **Decoder**: Reconstructs images from latent samples
  - Mirror architecture of encoder with upsampling layers
  - Includes residual connections for better gradient flow

- **Classifier**: Performs classification directly from the latent space
  - Allows the model to learn class-separable latent representations

- **Center Loss**: Encourages tight clusters in latent space by class
  - Improves the quality of the generated samples by creating a more organized latent space

### Conditional Diffusion Model

The diffusion model operates in the VAE's latent space with:

- **Noise Prediction Network**: A UNet-inspired architecture with:
  - Time embeddings to condition on diffusion timestep
  - Class embeddings for conditional generation
  - Attention mechanisms to capture long-range dependencies
  - Residual connections for improved gradient flow

- **Forward and Reverse Processes**:
  - Forward: Adds noise to latent vectors according to a predefined schedule
  - Reverse: Progressively removes noise to generate new samples

## Requirements

- Python 3.8+
- PyTorch 1.9+
- torchvision
- matplotlib
- numpy
- tqdm
- scikit-learn
- imageio

## Installation

```bash
git clone [repository-url]
cd cifar-cat-dog-conditional-vae-diffusion
pip install -r requirements.txt
```

## Usage

### Training the Models

To train the complete model pipeline:

```bash
python main.py
```

This will:
1. Download the CIFAR-10 dataset and filter for cats and dogs
2. Train the VAE model (if no pretrained model exists)
3. Train the conditional diffusion model (if no pretrained model exists)
4. Generate visualizations and samples

### Generating Images

After training, you can generate images using:

```python
# Load trained models
autoencoder = SimpleAutoencoder(in_channels=3, latent_dim=256, num_classes=2).to(device)
autoencoder.load_state_dict(torch.load("./cifar_cat_dog_conditional_improved/cifar_cat_dog_autoencoder.pt"))

conditional_unet = ConditionalUNet(latent_dim=256, hidden_dims=[256, 512, 1024, 512, 256], 
                                  time_emb_dim=256, num_classes=2).to(device)
conditional_unet.load_state_dict(torch.load("./cifar_cat_dog_conditional_improved/conditional_diffusion_final.pt"))

diffusion = ConditionalDenoiseDiffusion(conditional_unet, n_steps=1000, device=device)

# Generate cat images (class_idx=0)
cat_samples = generate_class_samples(autoencoder, diffusion, target_class=0, num_samples=5)

# Generate dog images (class_idx=1)
dog_samples = generate_class_samples(autoencoder, diffusion, target_class=1, num_samples=5)
```

### Visualization Functions

The project includes several visualization tools:

- `visualize_reconstructions`: Shows original and reconstructed images
- `visualize_latent_space`: Creates t-SNE visualizations of the latent space
- `visualize_denoising_steps`: Illustrates the denoising process and path in latent space
- `create_diffusion_animation`: Creates a GIF animation of the diffusion process
- `generate_samples_grid`: Produces a grid of samples for all classes

## Model Training Details

### VAE Training Strategy

The VAE is trained with a progressive strategy:
1. Initial focus on reconstruction
2. Gradually introduce KL divergence loss
3. Add classification and center loss components
4. Fine-tuning with all components

Loss components:
- Reconstruction loss (pixel-wise)
- VGG perceptual loss for better visual quality
- KL divergence for proper latent distribution
- Classification loss for class-separable features
- Center loss for improved clustering

### Diffusion Model Training

The diffusion model is trained to denoise latent vectors with:
- Class conditioning for controlled generation
- Residual scaling for improved stability
- Cosine annealing scheduler for better convergence

## Results

After training, the model can generate high-quality, class-specific images of cats and dogs. The results directory contains:

- Sample grids showing generated images for each class
- Animations showing the denoising process
- Latent space visualizations showing class separation
- Denoising path visualizations showing how samples evolve

 Model Component | Visualization | Description |
|-----------------|---------------|-------------|
| Autoencoder | ![Reconstructions](https://github.com/ynyeh0221/CIFAR10-cat-dog-generative-latent-diffusion/blob/main/v9/output/reconstruction/vae_reconstruction_epoch_950.png) | Original images (top) and their reconstructions (bottom) |
| Latent Space | ![Latent Space](https://github.com/ynyeh0221/CIFAR10-cat-dog-generative-latent-diffusion/blob/main/v9/output/latent_space/vae_latent_space_epoch_950.png) | t-SNE visualization of cat and dog latent representations |
| Class Samples | ![Class Samples](https://github.com/ynyeh0221/CIFAR10-cat-dog-generative-latent-diffusion/blob/main/v9/output/diffusion_sample_result/sample_class_Cat_epoch_3300.png)![Class Samples](https://github.com/ynyeh0221/CIFAR10-cat-dog-generative-latent-diffusion/blob/main/v9/output/diffusion_sample_result/sample_class_Dog_epoch_3300.png) | Generated samples for cat and dog classes |
| Denoising Process | ![Denoising Cat](https://github.com/ynyeh0221/CIFAR10-cat-dog-generative-latent-diffusion/blob/main/v9/output/diffusion_path/denoising_path_Cat_epoch_3300.png)![Denoising Dog](https://github.com/ynyeh0221/CIFAR10-cat-dog-generative-latent-diffusion/blob/main/v6/output/diffusion_path/denoising_path_Dog_epoch_3300.png) | Visualization of cat generation process and latent path |
| Animation | ![Cat Animation](https://github.com/ynyeh0221/CIFAR10-cat-dog-generative-latent-diffusion/blob/main/v9/diffusion_animation_class_Cat_epoch_3300.gif)![Dog Animation](https://github.com/ynyeh0221/CIFAR10-cat-dog-generative-latent-diffusion/blob/main/v6/diffusion_animation_class_Dog_epoch_3300.gif) | Animation of the denoising process for cat generation |
