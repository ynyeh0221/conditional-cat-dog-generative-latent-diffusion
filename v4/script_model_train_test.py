import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import math
import torch.nn.functional as F
import os
import numpy as np
from tqdm.auto import tqdm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import imageio
from torch.cuda.amp import autocast, GradScaler

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Set image size for CIFAR (32x32 RGB images)
img_size = 32

# Data preprocessing
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),  # This already scales to [0,1] range
])

# Batch size for training
batch_size = 512

# Define class names for CIFAR cats and dogs
class_names = ['Cat', 'Dog']


# Function to create a CIFAR Cat-Dog dataset
class CatDogDataset:
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img, label = self.dataset[self.indices[idx]]
        # Map labels: 3 (cat) -> 0, 5 (dog) -> 1
        if label == 3:
            return img, 0
        elif label == 5:
            return img, 1
        else:
            raise ValueError(f"Unexpected label: {label}")


# Function to create a CIFAR Cat-Dog dataset
def create_cat_dog_dataset(cifar_dataset):
    """Extract only cat and dog classes from CIFAR-10"""
    # In CIFAR-10, cat is class 3 and dog is class 5
    cat_indices = [i for i, (_, label) in enumerate(cifar_dataset) if label == 3]
    dog_indices = [i for i, (_, label) in enumerate(cifar_dataset) if label == 5]

    # Combine indices
    cat_dog_indices = cat_indices + dog_indices

    # Create dataset using the global class
    return CatDogDataset(cifar_dataset, cat_dog_indices)


# Channel Attention Layer for improved feature selection
class CALayer(nn.Module):
    def __init__(self, channel, reduction=8, bias=False):
        super(CALayer, self).__init__()
        # Global average pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Global max pooling for capturing important features
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
        )

        # Sigmoid activation for attention weights
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y_avg = self.avg_pool(x)
        y_max = self.max_pool(x)

        y_avg = self.conv_du(y_avg)
        y_max = self.conv_du(y_max)

        # Combine average and max pool features
        y = self.sigmoid(y_avg + y_max)

        return x * y


# Convolutional Attention Block - keeping the original name
class CAB(nn.Module):
    def __init__(self, n_feat, reduction=8, bias=False):
        super(CAB, self).__init__()
        self.conv1 = nn.Conv2d(n_feat, n_feat, 3, padding=1, bias=bias)
        self.bn1 = nn.BatchNorm2d(n_feat)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = nn.Conv2d(n_feat, n_feat, 3, padding=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(n_feat)

        self.ca = CALayer(n_feat, reduction, bias=bias)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.ca(out)

        # Residual connection
        out += residual

        return out


# Reparameterization module - keeping original name
class Reparameterize(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, mu, logvar):
        # During training use reparameterization trick
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            # During testing just return the mean
            return mu


# Enhanced Encoder with increased capacity - keeping original name VAEEncoder
class VAEEncoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=192):
        super().__init__()

        # Initial convolution with increased channels
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, stride=1, padding=1),  # Increased from 32
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # First downsampling block with more channels
        self.down1 = nn.Sequential(
            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # 32x32 -> 16x16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            CAB(128, reduction=8)
        )

        # Second downsampling block with more channels
        self.down2 = nn.Sequential(
            nn.Conv2d(128, 256, 4, stride=2, padding=1),  # 16x16 -> 8x8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            CAB(256, reduction=8),
            CAB(256, reduction=8)
        )

        # New: Third downsampling block for deeper feature extraction
        self.down3 = nn.Sequential(
            nn.Conv2d(256, 512, 4, stride=2, padding=1),  # 8x8 -> 4x4
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            CAB(512, reduction=8)
        )

        # Flatten layer for converting spatial features to vector
        self.flatten = nn.Flatten()

        # Latent space mapping (traditional design instead of RGB latent space)
        self.fc_mu = nn.Linear(512 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(512 * 4 * 4, latent_dim)

        # Class features branch
        self.class_branch = nn.Sequential(
            nn.Linear(512 * 4 * 4, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3)
        )

        # Reparameterization module
        self.reparameterize = Reparameterize()

    def forward(self, x):
        # Save intermediate feature maps for skip connections
        x1 = self.initial(x)  # 64 x 32 x 32
        x2 = self.down1(x1)  # 128 x 16 x 16
        x3 = self.down2(x2)  # 256 x 8 x 8
        x4 = self.down3(x3)  # 512 x 4 x 4

        # Flatten features
        x_flat = self.flatten(x4)  # B x (512*4*4)

        # Generate latent representation parameters
        mu = self.fc_mu(x_flat)  # B x latent_dim
        logvar = self.fc_logvar(x_flat)  # B x latent_dim

        # Class features
        class_features = self.class_branch(x_flat)  # B x 256

        # Apply reparameterization
        z = self.reparameterize(mu, logvar)  # B x latent_dim

        # Return all necessary outputs including skip features
        return z, mu, logvar, class_features, (x1, x2, x3, x4)


# Enhanced Decoder with skip connections - keeping original name
class Decoder(nn.Module):
    def __init__(self, latent_dim=256, out_channels=3):
        super().__init__()

        # Project from latent space to spatial features
        self.latent_proj = nn.Sequential(
            nn.Linear(latent_dim, 512 * 4 * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Initial processing
        self.initial = nn.Sequential(
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            CAB(512, reduction=8)
        )

        # First upsampling block: 4x4 -> 8x8
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            CAB(256, reduction=8)
        )

        # Note: skip_connections[3] has shape [B, 512, 4, 4]
        # Fusion layer after first skip connection
        self.fusion1 = nn.Sequential(
            nn.Conv2d(256 + 256, 256, 3, padding=1),  # 256 (up1) + 256 (skip[2]) = 512 channels
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Second upsampling block: 8x8 -> 16x16
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            CAB(128, reduction=8)
        )

        # Fusion layer after second skip connection
        self.fusion2 = nn.Sequential(
            nn.Conv2d(128 + 128, 128, 3, padding=1),  # 128 (up2) + 128 (skip[1]) = 256 channels
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Third upsampling block: 16x16 -> 32x32
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            CAB(64, reduction=8)
        )

        # Fusion layer after third skip connection
        self.fusion3 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, 3, padding=1),  # 64 (up3) + 64 (skip[0]) = 128 channels
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Final output layer
        self.final = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, out_channels, 3, padding=1),
            nn.Sigmoid()  # Use Sigmoid instead of Tanh, directly outputs [0,1] range
        )

    def forward(self, z, skip_connections=None):
        # Reconstruct initial feature map from latent vector
        x = self.latent_proj(z)
        x = x.view(-1, 512, 4, 4)
        x = self.initial(x)

        # First upsampling: 4x4 -> 8x8
        x = self.up1(x)
        if skip_connections is not None:
            # Skip connection indexing needs to match the spatial dimensions
            # skip_connections = (x1:32x32, x2:16x16, x3:8x8, x4:4x4)
            x = torch.cat([x, skip_connections[2]], dim=1)  # Connect 8x8 features
            x = self.fusion1(x)

        # Second upsampling: 8x8 -> 16x16
        x = self.up2(x)
        if skip_connections is not None:
            x = torch.cat([x, skip_connections[1]], dim=1)  # Connect 16x16 features
            x = self.fusion2(x)

        # Third upsampling: 16x16 -> 32x32
        x = self.up3(x)
        if skip_connections is not None:
            x = torch.cat([x, skip_connections[0]], dim=1)  # Connect 32x32 features
            x = self.fusion3(x)

        # Final output
        return self.final(x)


# Center Loss implementation - keeping original name
class CenterLoss(nn.Module):
    def __init__(self, num_classes=2, feat_dim=256):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, x, labels):
        batch_size = x.size(0)

        # Apply L2 normalization
        x_norm = F.normalize(x, p=2, dim=1)
        centers_norm = F.normalize(self.centers, p=2, dim=1)

        # Calculate distance matrix
        distmat = torch.pow(x_norm, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(centers_norm, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()

        # Use updated addmm_ syntax
        distmat.addmm_(x_norm, centers_norm.t(), beta=1, alpha=-2)

        # Get class mask
        classes = torch.arange(self.num_classes).to(labels.device)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        # Apply mask and calculate loss
        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss


# Complete VAE model with skip connections - keeping original name
class VariationalAutoencoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=192, num_classes=2):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder and decoder
        self.encoder = VAEEncoder(in_channels, latent_dim)
        self.decoder = Decoder(latent_dim, in_channels)

        # Classifier for class features
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(64, num_classes)
        )

        # Center loss for better clustering
        self.center_loss = CenterLoss(num_classes=num_classes, feat_dim=256)

    def encode(self, x):
        z, mu, logvar, class_features, _ = self.encoder(x)
        return z, class_features

    def decode(self, z, skip_connections=None):
        return self.decoder(z, skip_connections)

    def classify(self, class_features):
        return self.classifier(class_features)

    def compute_center_loss(self, class_features, labels):
        return self.center_loss(class_features, labels)

    def get_latent_params(self, x):
        _, mu, logvar, _, _ = self.encoder(x)
        return mu, logvar

    def forward(self, x):
        # Encode input
        z, mu, logvar, class_features, skip_connections = self.encoder(x)

        # Decode with skip connections
        reconstructed = self.decoder(z, skip_connections)

        return reconstructed, z, mu, logvar, class_features

    def compute_kl_loss(self, mu, logvar):
        # Compute KL divergence loss with numerical stability
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp() + 1e-8)

        batch_size = mu.size(0)
        kl_loss = kl_loss / batch_size

        return kl_loss


# Enhanced reconstruction loss function
def euclidean_distance_loss(x, y, reduction='mean', spectral_weight=0.1):
    """
    Enhanced Euclidean distance loss with spectral component for better detail preservation.

    Args:
        x: First tensor (reconstructed image)
        y: Second tensor (original image)
        reduction: 'mean', 'sum', or 'none'
        spectral_weight: Weight of the spectral component

    Returns:
        Enhanced Euclidean distance loss
    """
    # Calculate standard Euclidean distance
    squared_diff = (x - y) ** 2
    squared_dist = squared_diff.view(x.size(0), -1).sum(dim=1)
    euclidean_dist = torch.sqrt(squared_dist + 1e-8)  # Add small epsilon for numerical stability

    # Add perceptual component using MSE with SSIM (structural similarity)
    if spectral_weight > 0:
        # Simple structure similarity component
        def ssim_component(x, y, window_size=11):
            C1 = 0.01 ** 2
            C2 = 0.03 ** 2

            # Calculate means
            mu_x = F.avg_pool2d(x, window_size, stride=1, padding=window_size // 2)
            mu_y = F.avg_pool2d(y, window_size, stride=1, padding=window_size // 2)

            mu_x_sq = mu_x.pow(2)
            mu_y_sq = mu_y.pow(2)
            mu_xy = mu_x * mu_y

            # Calculate variances and covariance
            sigma_x_sq = F.avg_pool2d(x.pow(2), window_size, stride=1, padding=window_size // 2) - mu_x_sq
            sigma_y_sq = F.avg_pool2d(y.pow(2), window_size, stride=1, padding=window_size // 2) - mu_y_sq
            sigma_xy = F.avg_pool2d(x * y, window_size, stride=1, padding=window_size // 2) - mu_xy

            # SSIM formula
            ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / \
                       ((mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2))

            return 1 - ssim_map.mean(dim=1).mean(dim=1).mean(dim=1)

        # Calculate SSIM loss
        perceptual_dist = ssim_component(x, y)

        # Combine distances
        combined_dist = euclidean_dist + spectral_weight * perceptual_dist
    else:
        combined_dist = euclidean_dist

    # Apply reduction
    if reduction == 'mean':
        return combined_dist.mean()
    elif reduction == 'sum':
        return combined_dist.sum()
    else:  # 'none'
        return combined_dist


# Function to denormalize tensors for visualization
def denormalize_tanh(tensor):
    """
    Denormalize tensor that was normalized with tanh activation.
    (Already handled by decoder's output layer, but useful for debugging)
    """
    return (tensor + 1) / 2


# Swish activation function
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


# Time embedding for diffusion model
class TimeEmbedding(nn.Module):
    def __init__(self, n_channels=16):
        super().__init__()
        self.n_channels = n_channels
        self.lin1 = nn.Linear(self.n_channels, self.n_channels)
        self.act = Swish()
        self.lin2 = nn.Linear(self.n_channels, self.n_channels)

    def forward(self, t):
        # Sinusoidal time embedding similar to positional encoding
        half_dim = self.n_channels // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)

        # Process through MLP
        return self.lin2(self.act(self.lin1(emb)))


# Class embedding for conditional diffusion
class ClassEmbedding(nn.Module):
    def __init__(self, num_classes=2, n_channels=16):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, n_channels)
        self.lin1 = nn.Linear(n_channels, n_channels)
        self.act = Swish()
        self.lin2 = nn.Linear(n_channels, n_channels)

    def forward(self, c):
        # Get class embeddings
        emb = self.embedding(c)
        # Process through MLP (same structure as time embedding)
        return self.lin2(self.act(self.lin1(emb)))


# Attention block for UNet
class UNetAttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads

        self.norm = nn.GroupNorm(1, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        residual = x

        # Normalize input
        x = self.norm(x)

        # QKV projection
        qkv = self.qkv(x).reshape(b, 3, self.num_heads, c // self.num_heads, h * w)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        # Reshape for attention computation
        q = q.permute(0, 1, 3, 2)  # [b, heads, h*w, c//heads]
        k = k.permute(0, 1, 2, 3)  # [b, heads, c//heads, h*w]
        v = v.permute(0, 1, 3, 2)  # [b, heads, h*w, c//heads]

        # Compute attention
        scale = (c // self.num_heads) ** -0.5
        attn = torch.matmul(q, k) * scale
        attn = F.softmax(attn, dim=-1)

        # Apply attention
        out = torch.matmul(attn, v)  # [b, heads, h*w, c//heads]
        out = out.permute(0, 3, 1, 2)  # [b, c//heads, heads, h*w]
        out = out.reshape(b, c, h, w)

        # Project and add residual
        return self.proj(out) + residual


# Residual block for UNet with class conditioning
class UNetResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, d_time=16, num_groups=8, dropout_rate=0.2):
        super().__init__()

        # Feature normalization and convolution
        self.norm1 = nn.GroupNorm(min(num_groups, in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Time and class embedding projections
        self.time_emb = nn.Linear(d_time, out_channels)
        self.class_emb = nn.Linear(d_time, out_channels)  # Same dimension as time embedding
        self.act = Swish()

        self.dropout = nn.Dropout(dropout_rate)

        # Second convolution
        self.norm2 = nn.GroupNorm(min(num_groups, out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # Residual connection handling
        self.residual = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x, t, c=None):
        # First part
        h = self.act(self.norm1(x))
        h = self.conv1(h)

        # Add time embedding
        t_emb = self.act(self.time_emb(t))
        h = h + t_emb.view(-1, t_emb.shape[1], 1, 1)

        # Add class embedding if provided
        if c is not None:
            c_emb = self.act(self.class_emb(c))
            h = h + c_emb.view(-1, c_emb.shape[1], 1, 1)

        # Second part
        h = self.act(self.norm2(h))
        h = self.dropout(h)
        h = self.conv2(h)

        # Residual connection
        return h + self.residual(x)


# Switch Sequential for handling time and class embeddings
class SwitchSequential(nn.Sequential):
    def forward(self, x, t=None, c=None):
        for layer in self:
            if isinstance(layer, UNetResidualBlock):
                x = layer(x, t, c)
            elif isinstance(layer, UNetAttentionBlock):
                x = layer(x)
            else:
                x = layer(x)
        return x


# Class-Conditional UNet for noise prediction
class ConditionalUNet(nn.Module):
    def __init__(self, in_channels=3, hidden_dims=[16, 32, 64], num_classes=2, dropout_rate=0.2):
        super().__init__()

        # Time embedding
        self.time_emb = TimeEmbedding(n_channels=16)

        # Class embedding
        self.class_emb = ClassEmbedding(num_classes=num_classes, n_channels=16)

        # Downsampling path (encoder)
        self.down_blocks = nn.ModuleList()

        # Initial convolution
        self.initial_conv = nn.Conv2d(in_channels, hidden_dims[0], 3, padding=1)

        # Downsampling blocks
        input_dim = hidden_dims[0]
        for dim in hidden_dims[1:]:
            self.down_blocks.append(
                nn.ModuleList([
                    UNetResidualBlock(input_dim, input_dim),
                    UNetResidualBlock(input_dim, input_dim),
                    nn.Conv2d(input_dim, dim, 4, stride=2, padding=1)  # Downsample
                ])
            )
            input_dim = dim

        self.dropout_mid = nn.Dropout(dropout_rate)

        # Middle block (bottleneck)
        self.middle_blocks = nn.ModuleList([
            UNetResidualBlock(hidden_dims[-1], hidden_dims[-1]),
            UNetAttentionBlock(hidden_dims[-1]),
            UNetResidualBlock(hidden_dims[-1], hidden_dims[-1])
        ])

        # Upsampling path (decoder)
        self.up_blocks = nn.ModuleList()

        # Upsampling blocks
        for dim in reversed(hidden_dims[:-1]):
            self.up_blocks.append(
                nn.ModuleList([
                    nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1], 4, stride=2, padding=1),
                    UNetResidualBlock(hidden_dims[-1] + dim, dim),
                    UNetResidualBlock(dim, dim),
                ])
            )
            hidden_dims[-1] = dim

        self.dropout_final = nn.Dropout(dropout_rate)

        # Final blocks
        self.final_block = SwitchSequential(
            UNetResidualBlock(hidden_dims[0] * 2, hidden_dims[0]),
            nn.Conv2d(hidden_dims[0], in_channels, 3, padding=1)
        )

    def forward(self, x, t, c=None):
        # Time embedding
        t_emb = self.time_emb(t)

        # Class embedding (if provided)
        c_emb = None
        if c is not None:
            c_emb = self.class_emb(c)

        # Initial convolution
        x = self.initial_conv(x)

        # Store skip connections
        skip_connections = [x]

        # Downsampling
        for resblock1, resblock2, downsample in self.down_blocks:
            x = resblock1(x, t_emb, c_emb)
            x = resblock2(x, t_emb, c_emb)
            skip_connections.append(x)
            x = downsample(x)

        x = self.dropout_mid(x)

        # Middle blocks
        for block in self.middle_blocks:
            if isinstance(block, UNetAttentionBlock):
                x = block(x)
            else:
                x = block(x, t_emb, c_emb)

        # Upsampling
        for upsample, resblock1, resblock2 in self.up_blocks:
            x = upsample(x)  # Upsample first
            x = torch.cat([x, skip_connections.pop()], dim=1)  # Then concatenate
            x = resblock1(x, t_emb, c_emb)  # Then process
            x = resblock2(x, t_emb, c_emb)

        x = self.dropout_final(x)

        # Final blocks
        x = torch.cat([x, skip_connections.pop()], dim=1)
        x = self.final_block(x, t_emb, c_emb)

        return x


# Class-conditional diffusion model
class ConditionalDenoiseDiffusion():
    def __init__(self, eps_model, n_steps=1000, device=None):
        super().__init__()
        self.eps_model = eps_model
        self.device = device

        # Linear beta schedule
        self.beta = torch.linspace(0.0001, 0.02, n_steps).to(device)
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.n_steps = n_steps

    def q_sample(self, x0, t, eps=None):
        """Forward diffusion process: add noise to data"""
        if eps is None:
            eps = torch.randn_like(x0)

        alpha_bar_t = self.alpha_bar[t].reshape(-1, 1, 1, 1)
        return torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * eps

    def p_sample(self, xt, t, c=None, guidance_scale=3.0):
        """Single denoising step with enhanced classifier guidance"""
        # Convert time to tensor format expected by model
        if not isinstance(t, torch.Tensor):
            t = torch.tensor([t], device=xt.device)

        eps_uncond = self.eps_model(xt, t, None)

        if c is not None:
            eps_cond = self.eps_model(xt, t, c)
            eps_theta = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
        else:
            eps_theta = eps_uncond

        alpha_t = self.alpha[t].reshape(-1, 1, 1, 1)
        alpha_bar_t = self.alpha_bar[t].reshape(-1, 1, 1, 1)

        mean = (xt - (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t) * eps_theta) / torch.sqrt(alpha_t)

        var = self.beta[t].reshape(-1, 1, 1, 1)

        if t[0] > 0:
            noise = torch.randn_like(xt)
            return mean + torch.sqrt(var) * noise
        else:
            return mean

    def sample(self, shape, device, c=None, guidance_scale=3.0):
        """Generate samples with enhanced classifier guidance"""
        x = torch.randn(shape, device=device)

        for t in tqdm(reversed(range(self.n_steps)), desc="Sampling"):
            x = self.p_sample(x, t, c, guidance_scale=guidance_scale)

        return x

    def loss(self, x0, labels=None):
        """Calculate noise prediction loss with optional class conditioning"""
        batch_size = x0.shape[0]

        # Random timestep for each sample
        t = torch.randint(0, self.n_steps, (batch_size,), device=x0.device, dtype=torch.long)

        # Add noise
        eps = torch.randn_like(x0)
        xt = self.q_sample(x0, t, eps)

        # Predict noise (with class conditioning if labels provided)
        eps_theta = self.eps_model(xt, t, labels)

        return euclidean_distance_loss(eps, eps_theta)


# Function to generate a grid of samples for all classes
def generate_samples_grid(vae, diffusion, n_per_class=5, save_dir="./results"):
    """Generate a grid of samples with n_per_class samples for each class"""
    os.makedirs(save_dir, exist_ok=True)
    device = next(vae.parameters()).device

    # Set models to evaluation mode
    vae.eval()
    diffusion.eps_model.eval()

    n_classes = len(class_names)
    # Create figure with extra column for class labels
    fig, axes = plt.subplots(n_classes, n_per_class + 1, figsize=((n_per_class + 1) * 2, n_classes * 2))

    # Add a title to explain what the figure shows
    fig.suptitle(f'CIFAR Cat and Dog Samples Generated by Diffusion Model',
                 fontsize=16, y=0.98)

    for i in range(n_classes):
        # Create a text-only cell for the class name
        axes[i, 0].text(0.5, 0.5, class_names[i],
                        fontsize=14, fontweight='bold',
                        ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.7))
        axes[i, 0].axis('off')

        # Generate samples with class conditioning
        class_tensor = torch.tensor([i] * n_per_class, device=device)
        latent_shape = (n_per_class, 3, 8, 8)  # 3 channels for RGB

        # Sample from the diffusion model with class conditioning
        samples = diffusion.sample(latent_shape, device, class_tensor)

        # Decode samples
        with torch.no_grad():
            samples_flat = samples.view(n_per_class, -1)
            decoded = vae.decode(samples_flat)

        # Plot samples (starting from column 1, as column 0 is for class names)
        for j in range(n_per_class):
            img = decoded[j].cpu().permute(1, 2, 0).numpy()  # Change from [C,H,W] to [H,W,C] for plotting
            axes[i, j + 1].imshow(img)  # No cmap for RGB images

            # Remove axis ticks
            axes[i, j + 1].axis('off')

            # Add sample numbers above the first row
            if i == 0:
                axes[i, j + 1].set_title(f'Sample {j + 1}', fontsize=9)

    # Add a text box explaining the visualization
    description = (
        "This visualization shows cat and dog images generated by the conditional diffusion model.\n"
        "The model creates new, synthetic images based on learned patterns from CIFAR-10.\n"
        "Each row corresponds to a different animal category as labeled."
    )
    plt.figtext(0.5, 0.01, description, ha='center', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7))

    plt.tight_layout(rect=[0, 0.03, 1, 0.92])  # Adjust layout to make room for titles
    plt.savefig(f"{save_dir}/samples_grid_all_classes.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Set models back to training mode
    vae.train()
    diffusion.eps_model.train()

    print(f"Generated sample grid for all classes with clearly labeled categories")
    return f"{save_dir}/samples_grid_all_classes.png"


# Visualize latent space denoising process for a specific class
def visualize_denoising_steps(vae, diffusion, class_idx, save_path=None):
    """
    Visualize both the denoising process and the corresponding path in latent space.

    Args:
        vae: Trained autoencoder model
        diffusion: Trained diffusion model
        class_idx: Target class index (0-1 for cat/dog)
        save_dir: Directory to save visualizations
    """
    device = next(vae.parameters()).device

    # Set models to evaluation mode
    vae.eval()
    diffusion.eps_model.eval()

    # ===== PART 1: Setup dimensionality reduction for latent space =====
    print(f"Generating latent space projection for class {class_names[class_idx]}...")

    # Load CIFAR-10 test data
    cifar_test = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    # Filter to cat/dog classes
    test_dataset = create_cat_dog_dataset(cifar_test)
    test_loader = DataLoader(test_dataset, batch_size=500, shuffle=False)

    # Extract features and labels
    all_latents = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            latents = vae.encode(images)
            # 在这里添加展平操作
            latents = latents.view(latents.size(0), -1)  # 将 [batch_size, 3, 8, 8] 展平为 [batch_size, 192]
            all_latents.append(latents.detach().cpu().numpy())
            all_labels.append(labels.numpy())

    # Combine batches
    all_latents = np.vstack(all_latents)
    all_labels = np.concatenate(all_labels)

    # Use PCA for dimensionality reduction
    print("Computing PCA projection...")
    pca = PCA(n_components=2, random_state=42)
    latents_2d = pca.fit_transform(all_latents)

    # ===== PART 2: Setup denoising visualization =====
    # Parameters for visualization
    n_samples = 5  # Number of samples to generate
    steps_to_show = 8  # Number of denoising steps to visualize
    step_size = diffusion.n_steps // steps_to_show
    timesteps = list(range(0, diffusion.n_steps, step_size))[::-1]

    # Generate sample from pure noise with class conditioning
    class_tensor = torch.tensor([class_idx] * n_samples, device=device)
    x = torch.randn((n_samples, 3, 8, 8), device=device)  # 3 channels for RGB

    # Store denoised samples at each timestep
    samples_per_step = []
    # Track latent path for the first sample
    path_latents = []

    # ===== PART 3: Perform denoising and track path =====
    with torch.no_grad():
        for t in timesteps:
            # Current denoised state
            current_x = x.clone()

            # Denoise from current step to t=0 with class conditioning
            for time_step in range(t, -1, -1):
                current_x = diffusion.p_sample(current_x, torch.tensor([time_step], device=device), class_tensor)

            # Store the latent vector for the first sample for path visualization
            path_latents.append(current_x[0:1].view(1, -1).detach().cpu().numpy())

            # Decode to images
            current_x_flat = current_x.view(n_samples, -1)
            decoded = vae.decode(current_x_flat)

            # Add to samples
            samples_per_step.append(decoded.cpu())

        # Add final denoised state to path
        path_latents.append(current_x[0:1].view(1, -1).detach().cpu().numpy())

    # Stack path latents
    path_latents = np.vstack(path_latents)

    # Project path points to PCA space
    path_2d = pca.transform(path_latents)

    # ===== PART 4: Create combined visualization =====
    # Create a figure with 2 subplots: denoising process (top) and latent path (bottom)
    fig = plt.figure(figsize=(16, 16))

    # Configure subplot layout
    gs = plt.GridSpec(2, 1, height_ratios=[1.5, 1], hspace=0.3)

    # ===== PART 5: Plot denoising process (top subplot) =====
    ax_denoising = fig.add_subplot(gs[0])

    # Create a grid for the denoising visualization
    grid_rows = n_samples
    grid_cols = len(timesteps)

    # Set title for the denoising subplot
    ax_denoising.set_title(f"Diffusion Model Denoising Process for {class_names[class_idx]}",
                           fontsize=16, pad=10)

    # Hide axis ticks
    ax_denoising.set_xticks([])
    ax_denoising.set_yticks([])

    # Create nested gridspec for the denoising images
    gs_denoising = gs[0].subgridspec(grid_rows, grid_cols, wspace=0.1, hspace=0.1)

    # Plot each denoising step
    for i in range(n_samples):
        for j, t in enumerate(timesteps):
            ax = fig.add_subplot(gs_denoising[i, j])
            img = samples_per_step[j][i].permute(1, 2, 0).numpy()  # Change from [C,H,W] to [H,W,C] for plotting
            ax.imshow(img)  # No cmap for RGB images

            # Add timestep labels only to the top row
            if i == 0:
                ax.set_title(f't={t}', fontsize=9)

            # Add sample labels only to the leftmost column
            if j == 0:
                ax.set_ylabel(f"Sample {i + 1}", fontsize=9)

            # Highlight the first sample that corresponds to the path
            if i == 0:
                ax.spines['bottom'].set_color('red')
                ax.spines['top'].set_color('red')
                ax.spines['left'].set_color('red')
                ax.spines['right'].set_color('red')
                ax.spines['bottom'].set_linewidth(2)
                ax.spines['top'].set_linewidth(2)
                ax.spines['left'].set_linewidth(2)
                ax.spines['right'].set_linewidth(2)

            ax.set_xticks([])
            ax.set_yticks([])

    # Add text indicating the first row corresponds to the latent path
    plt.figtext(0.02, 0.65, "Path Tracked →", fontsize=12, color='red',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='red'))

    # ===== PART 6: Plot latent space path (bottom subplot) =====
    ax_latent = fig.add_subplot(gs[1])

    # Plot each class with alpha transparency
    for i in range(len(class_names)):
        mask = all_labels == i
        alpha = 0.3 if i != class_idx else 0.8  # Highlight target class
        size = 20 if i != class_idx else 40  # Larger points for target class
        ax_latent.scatter(
            latents_2d[mask, 0],
            latents_2d[mask, 1],
            label=class_names[i],
            alpha=alpha,
            s=size
        )

    # Plot the diffusion path
    ax_latent.plot(
        path_2d[:, 0],
        path_2d[:, 1],
        'r-o',
        linewidth=2.5,
        markersize=8,
        label=f"Diffusion Path",
        zorder=10  # Ensure path is drawn on top
    )

    # Add arrows to show direction
    for i in range(len(path_2d) - 1):
        ax_latent.annotate(
            "",
            xy=(path_2d[i + 1, 0], path_2d[i + 1, 1]),
            xytext=(path_2d[i, 0], path_2d[i, 1]),
            arrowprops=dict(arrowstyle="->", color="darkred", lw=1.5)
        )

    # Add timestep labels along the path
    for i, t in enumerate(timesteps):
        ax_latent.annotate(
            f"t={t}",
            xy=(path_2d[i, 0], path_2d[i, 1]),
            xytext=(path_2d[i, 0] + 2, path_2d[i, 1] + 2),
            fontsize=8,
            color='darkred'
        )

    # Add markers for start and end points
    ax_latent.scatter(path_2d[0, 0], path_2d[0, 1], c='black', s=100, marker='x', label="Start (Noise)", zorder=11)
    ax_latent.scatter(path_2d[-1, 0], path_2d[-1, 1], c='green', s=100, marker='*', label="End (Generated)",
                      zorder=11)

    # Highlight target class area
    target_mask = all_labels == class_idx
    target_center = np.mean(latents_2d[target_mask], axis=0)
    ax_latent.scatter(target_center[0], target_center[1], c='green', s=300, marker='*',
                      edgecolor='black', alpha=0.7, zorder=9)
    ax_latent.annotate(
        f"TARGET: {class_names[class_idx]}",
        xy=(target_center[0], target_center[1]),
        xytext=(target_center[0] + 5, target_center[1] + 5),
        fontsize=14,
        fontweight='bold',
        color='darkgreen',
        bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8)
    )

    ax_latent.set_title(f"Diffusion Path in Latent Space for {class_names[class_idx]}", fontsize=16)
    ax_latent.legend(fontsize=10, loc='best')
    ax_latent.grid(True, linestyle='--', alpha=0.7)

    # Add explanatory text
    plt.figtext(
        0.5, 0.01,
        "This visualization shows the denoising process (top) and the corresponding path in latent space (bottom).\n"
        "The first row of images (highlighted in red) corresponds to the red path in the latent space plot below.",
        ha='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

    # Save the figure
    fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0.3, wspace=0.1)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Denoising visualization for {class_names[class_idx]} saved to {save_path}")

    # Set models back to training mode
    vae.train()
    diffusion.eps_model.train()

    return save_path


# Visualize autoencoder reconstructions
def visualize_reconstructions(vae, epoch, save_dir="./results"):
    """Visualize original and reconstructed images at each epoch"""

    os.makedirs(save_dir, exist_ok=True)
    device = next(vae.parameters()).device

    # Get a batch of test data
    cifar_test = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    test_dataset = create_cat_dog_dataset(cifar_test)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True)

    test_images, test_labels = next(iter(test_loader))
    test_images = test_images.to(device)

    # Generate reconstructions
    vae.eval()
    with torch.no_grad():
        # Handle the updated model structure which might return multiple values
        reconstructed_output = vae(test_images)

        # Check what the model returns and extract reconstructed images
        if isinstance(reconstructed_output, tuple):
            # If the forward method returns a tuple, the first element should be reconstructions
            reconstructed = reconstructed_output[0]
        else:
            # If it returns a single tensor
            reconstructed = reconstructed_output

    # Create visualization
    fig, axes = plt.subplots(2, 8, figsize=(16, 4))

    for i in range(8):
        # Original image
        img = test_images[i].cpu().permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        axes[0, i].imshow(img)
        axes[0, i].set_title(f'Original: {class_names[test_labels[i]]}')
        axes[0, i].axis('off')

        # Reconstruction
        recon_img = reconstructed[i].cpu().permute(1, 2, 0).numpy()
        recon_img = np.clip(recon_img, 0, 1)
        axes[1, i].imshow(recon_img)
        axes[1, i].set_title('Reconstruction')
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.savefig(f"{save_dir}/reconstruction_epoch_{epoch}.png")
    plt.close()
    vae.train()

# Visualize latent space with t-SNE
def visualize_latent_space(vae, epoch, save_dir="./results"):
    """Visualize the latent space of the autoencoder using t-SNE"""
    os.makedirs(save_dir, exist_ok=True)
    device = next(vae.parameters()).device

    # Get test data
    cifar_test = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    test_dataset = create_cat_dog_dataset(cifar_test)
    test_loader = DataLoader(test_dataset, batch_size=500, shuffle=False)

    # Extract features and labels
    vae.eval()
    all_latents = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)

            # Handle the encode method which might return multiple values
            encode_output = vae.encode(images)

            # Check what the encode method returns and extract latent features
            if isinstance(encode_output, tuple):
                # If the encode method returns a tuple, the first element should be the latent representation
                latents = encode_output[0]
            else:
                # If it returns a single tensor
                latents = encode_output

            # Ensure latents is flattened to 2D for t-SNE
            if len(latents.shape) > 2:
                latents = latents.view(latents.size(0), -1)

            all_latents.append(latents.cpu().numpy())
            all_labels.append(labels.numpy())

    # Combine batches
    all_latents = np.vstack(all_latents)
    all_labels = np.concatenate(all_labels)

    # Use t-SNE for dimensionality reduction
    try:
        tsne = TSNE(n_components=2, random_state=42)
        latents_2d = tsne.fit_transform(all_latents)

        # Plot the 2D latent space
        plt.figure(figsize=(10, 8))
        for i in range(len(class_names)):  # 2 classes (cat/dog)
            mask = all_labels == i
            plt.scatter(latents_2d[mask, 0], latents_2d[mask, 1], label=class_names[i], alpha=0.6)

        plt.title(f"t-SNE Visualization of Latent Space (Epoch {epoch})")
        plt.legend()
        plt.savefig(f"{save_dir}/latent_space_epoch_{epoch}.png")
        plt.close()
    except Exception as e:
        print(f"t-SNE visualization error: {e}")

    vae.train()


# Function to generate samples of a specific class (need this for training)
def generate_class_samples(vae, diffusion, target_class, num_samples=5, save_path=None):
    """
    Generate samples of a specific target class

    Args:
        vae: Trained autoencoder model
        diffusion: Trained conditional diffusion model
        target_class: Index of the target class (0-1) or class name
        num_samples: Number of samples to generate
        save_path: Path to save the generated samples

    Returns:
        Tensor of generated samples
    """
    device = next(vae.parameters()).device

    # Set models to evaluation mode
    vae.eval()
    diffusion.eps_model.eval()

    # Convert class name to index if string is provided
    if isinstance(target_class, str):
        if target_class in class_names:
            target_class = class_names.index(target_class)
        else:
            raise ValueError(f"Invalid class name: {target_class}. Must be one of {class_names}")

    # Create class conditioning tensor
    class_tensor = torch.tensor([target_class] * num_samples, device=device)

    # Generate samples
    latent_shape = (num_samples, 3, 8, 8)  # 3 channels for RGB
    with torch.no_grad():
        # Sample from the diffusion model with class conditioning
        latent_samples = diffusion.sample(latent_shape, device, class_tensor)

        # Decode latents to images
        latent_samples_flat = latent_samples.view(num_samples, -1)
        samples = vae.decode(latent_samples_flat)

    # Save samples if path provided
    if save_path:
        plt.figure(figsize=(num_samples * 2, 3))
        for i in range(num_samples):
            plt.subplot(1, num_samples, i + 1)
            img = samples[i].cpu().permute(1, 2, 0).numpy()  # Convert from [C,H,W] to [H,W,C]
            plt.imshow(img)  # No cmap for RGB images
            plt.axis('off')
            plt.title(f"{class_names[target_class]}")

        plt.suptitle(f"Generated {class_names[target_class]} Samples")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    return samples


def create_diffusion_animation(vae, diffusion, class_idx, num_frames=50, seed=42,
                               save_path=None, temp_dir=None, fps=10, reverse=False):
    """
    Create a GIF animation showing the diffusion process.

    Args:
        vae: Trained autoencoder model
        diffusion: Trained diffusion model
        class_idx: Target class index (0-1) or class name
        num_frames: Number of frames to include in the animation
        seed: Random seed for reproducibility
        save_path: Path to save the output GIF
        temp_dir: Directory to save temporary frames (will be created if None)
        fps: Frames per second in the output GIF
        reverse: If False (default), show t=0→1000 (image to noise), otherwise t=1000→0 (noise to image)

    Returns:
        Path to the created GIF file
    """
    device = next(vae.parameters()).device

    # Set models to evaluation mode
    vae.eval()
    diffusion.eps_model.eval()

    # Convert class name to index if string is provided
    if isinstance(class_idx, str):
        if class_idx in class_names:
            class_idx = class_names.index(class_idx)
        else:
            raise ValueError(f"Invalid class name: {class_idx}. Must be one of {class_names}")

    # Create temp directory if needed
    if temp_dir is None:
        temp_dir = os.path.join('./temp_frames', f'class_{class_idx}_{seed}')
    os.makedirs(temp_dir, exist_ok=True)

    # Default save path if none provided
    if save_path is None:
        save_dir = './results'
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'diffusion_animation_{class_names[class_idx]}.gif')

    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create class conditioning tensor
    class_tensor = torch.tensor([class_idx], device=device)

    # Calculate timesteps to sample
    total_steps = diffusion.n_steps
    if num_frames >= total_steps:
        timesteps = list(range(total_steps))
    else:
        # Select evenly spaced timesteps
        step_size = total_steps // num_frames
        timesteps = list(range(0, total_steps, step_size))
        if timesteps[-1] != total_steps - 1:
            timesteps.append(total_steps - 1)

    # Sort timesteps for proper diffusion order
    if reverse:
        # From noise to image (t=1000 to t=0)
        timesteps = sorted(timesteps, reverse=True)
    else:
        # From image to noise (t=0 to t=1000)
        timesteps = sorted(timesteps)

    # For the looping effect, we'll add frames going backward too
    if not reverse:
        # Add timestamps going back from high to low, excluding endpoints to avoid duplicates
        backward_timesteps = sorted(timesteps[1:-1], reverse=True)
        timesteps = timesteps + backward_timesteps

    print(f"Creating diffusion animation for class '{class_names[class_idx]}'...")
    frame_paths = []

    with torch.no_grad():
        # First, generate a proper clean image at t=0 by denoising from pure noise
        print("Generating initial clean image...")
        # Start from pure noise
        x = torch.randn((1, 3, 8, 8), device=device)  # 3 channels for RGB

        # Denoise completely to get clean image at t=0
        for time_step in tqdm(range(total_steps - 1, -1, -1), desc="Denoising"):
            x = diffusion.p_sample(x, torch.tensor([time_step], device=device), class_tensor)

        # Now we have a clean, denoised image at t=0
        clean_x = x.clone()

        # Generate frames for animation
        print("Generating animation frames...")
        for i, t in enumerate(tqdm(timesteps)):
            # Start with clean image
            current_x = clean_x.clone()

            if t > 0:  # Skip adding noise at t=0
                # Apply forward diffusion up to timestep t
                # Generate a fixed noise vector for consistency
                torch.manual_seed(seed)  # Ensure same noise pattern
                eps = torch.randn_like(current_x)

                # Apply noise according to diffusion schedule
                alpha_bar_t = diffusion.alpha_bar[t].reshape(-1, 1, 1, 1)
                current_x = torch.sqrt(alpha_bar_t) * current_x + torch.sqrt(1 - alpha_bar_t) * eps

            # Decode to image
            current_x_flat = current_x.view(1, -1)
            decoded = vae.decode(current_x_flat)

            # Convert to numpy for saving
            img = decoded[0].cpu().permute(1, 2, 0).numpy()  # Convert from [C,H,W] to [H,W,C]

            # Create and save frame
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(img)  # No cmap for RGB images
            ax.axis('off')

            # Add timestep information
            progress = (t / total_steps) * 100
            title = f'Class: {class_names[class_idx]} (t={t}, {progress:.1f}% noise)'
            ax.set_title(title)

            # Save frame
            frame_path = os.path.join(temp_dir, f'frame_{i:04d}.png')
            plt.savefig(frame_path, bbox_inches='tight')
            plt.close(fig)
            frame_paths.append(frame_path)

    # Create GIF from frames
    print(f"Creating GIF animation at {fps} fps...")
    with imageio.get_writer(save_path, mode='I', fps=fps, loop=0) as writer:
        for frame_path in frame_paths:
            image = imageio.imread(frame_path)
            writer.append_data(image)

    # Clean up temporary frames
    print("Cleaning up temporary files...")
    for frame_path in frame_paths:
        os.remove(frame_path)

    # Try to remove the temp directory (if empty)
    try:
        os.rmdir(temp_dir)
    except OSError:
        pass  # Directory not empty or other error

    print(f"Animation saved to {save_path}")
    return save_path


# Main function
def main():
    """Main function to run the entire pipeline non-interactively"""
    print("Starting class-conditional diffusion model for CIFAR cats and dogs")

    # Set device
    device = torch.device("mps" if torch.backends.mps.is_available() else
                         ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    # Create results directory
    results_dir = "./cifar_cat_dog_conditional_v4"
    os.makedirs(results_dir, exist_ok=True)

    # Load CIFAR-10 dataset and filter for cats and dogs
    print("Loading and filtering CIFAR-10 dataset for cats and dogs...")
    cifar_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_dataset = create_cat_dog_dataset(cifar_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    # Paths for saved models
    vae_path = f"{results_dir}/cifar_cat_dog_autoencoder.pt"
    diffusion_path = f"{results_dir}/conditional_diffusion_final.pt"

    # Create vae
    vae = VariationalAutoencoder(in_channels=3).to(device)  # 3 channels for RGB

    # Check if trained autoencoder exists
    if os.path.exists(vae_path):
        print(f"Loading existing vae from {vae_path}")
        vae.load_state_dict(torch.load(vae_path, map_location=device))
        vae.eval()
    else:
        print("No existing vae found. Training a new one...")

        # Define train function
        def train_vae(vae, train_loader, num_epochs=150, lr=5e-5,
                      lambda_recon=1.0, lambda_cls=0.5, lambda_center=0.1, lambda_kl=0.05,
                      visualize_every=5, save_dir="./results"):
            """
            Train VAE with improved hyperparameters and training approach

            Args:
                vae: VAE model
                train_loader: DataLoader for training data
                num_epochs: Number of training epochs
                lr: Learning rate
                lambda_recon: Weight for reconstruction loss (increased)
                lambda_cls: Weight for classification loss (reduced)
                lambda_center: Weight for center loss (reduced)
                lambda_kl: Weight for KL divergence loss (reduced)
                visualize_every: Epoch interval for visualization
                save_dir: Directory to save results

            Returns:
                Trained VAE and loss history
            """
            print("Starting enhanced VAE training...")
            import os
            os.makedirs(save_dir, exist_ok=True)
            device = next(vae.parameters()).device

            # Use AdamW optimizer with improved parameters
            optimizer = torch.optim.AdamW(vae.parameters(), lr=lr, betas=(0.9, 0.999),
                                          eps=1e-8, weight_decay=1e-6)

            # Cosine annealing learning rate scheduler
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=10, T_mult=2, eta_min=1e-6
            )

            # Initialize loss history
            loss_history = {'total': [], 'recon': [], 'class': [], 'center': [], 'kl': []}

            # Mixed precision training setup if available
            use_amp = torch.cuda.is_available()
            if use_amp:
                scaler = torch.cuda.amp.GradScaler()

            for epoch in range(num_epochs):
                vae.train()
                epoch_recon_loss = 0
                epoch_class_loss = 0
                epoch_center_loss = 0
                epoch_kl_loss = 0
                epoch_total_loss = 0

                # Cyclical KL annealing
                cycle_size = num_epochs // 4
                cycle_position = epoch % cycle_size
                if cycle_position < cycle_size // 2:
                    # Linear increase in first half of cycle
                    kl_weight = lambda_kl * (0.1 + 0.9 * (cycle_position / (cycle_size // 2)))
                else:
                    # Constant in second half of cycle
                    kl_weight = lambda_kl

                # Training loop
                for batch_idx, (data, labels) in enumerate(train_loader):
                    data = data.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    # Forward pass with mixed precision if available
                    if use_amp:
                        with torch.cuda.amp.autocast():
                            # Forward pass
                            reconstructed, z, mu, logvar, class_features = vae(data)

                            # Calculate losses
                            recon_loss = euclidean_distance_loss(reconstructed, data, spectral_weight=0.2)
                            kl_loss = vae.compute_kl_loss(mu, logvar)
                            class_logits = vae.classify(class_features)
                            class_loss = F.cross_entropy(class_logits, labels)
                            center_loss = vae.compute_center_loss(class_features, labels)

                            # Combined loss with improved weighting
                            total_loss = lambda_recon * recon_loss + kl_weight * kl_loss + \
                                         lambda_cls * class_loss + lambda_center * center_loss

                        # Backward and optimize with mixed precision
                        scaler.scale(total_loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        # Standard precision training
                        # Forward pass
                        reconstructed, z, mu, logvar, class_features = vae(data)

                        # Calculate losses
                        recon_loss = euclidean_distance_loss(reconstructed, data, spectral_weight=0.2)
                        kl_loss = vae.compute_kl_loss(mu, logvar)
                        class_logits = vae.classify(class_features)
                        class_loss = F.cross_entropy(class_logits, labels)
                        center_loss = vae.compute_center_loss(class_features, labels)

                        # Combined loss with improved weighting
                        total_loss = lambda_recon * recon_loss + kl_weight * kl_loss + \
                                     lambda_cls * class_loss + lambda_center * center_loss

                        # Backward and optimize
                        total_loss.backward()
                        torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
                        optimizer.step()

                    # Track losses
                    epoch_recon_loss += recon_loss.item()
                    epoch_class_loss += class_loss.item()
                    epoch_center_loss += center_loss.item()
                    epoch_kl_loss += kl_loss.item()
                    epoch_total_loss += total_loss.item()

                # Update learning rate
                scheduler.step()

                # Calculate average losses
                num_batches = len(train_loader)
                avg_recon_loss = epoch_recon_loss / num_batches
                avg_class_loss = epoch_class_loss / num_batches
                avg_center_loss = epoch_center_loss / num_batches
                avg_kl_loss = epoch_kl_loss / num_batches
                avg_total_loss = epoch_total_loss / num_batches

                # Print progress
                print(f"Epoch {epoch + 1}/{num_epochs}, "
                      f"Total: {avg_total_loss:.6f}, "
                      f"Recon: {avg_recon_loss:.6f}, "
                      f"KL: {avg_kl_loss:.6f} (weight: {kl_weight:.6f}), "
                      f"Class: {avg_class_loss:.6f}, "
                      f"Center: {avg_center_loss:.6f}")

                # Record loss history
                loss_history['recon'].append(avg_recon_loss)
                loss_history['class'].append(avg_class_loss)
                loss_history['center'].append(avg_center_loss)
                loss_history['kl'].append(avg_kl_loss)
                loss_history['total'].append(avg_total_loss)

                # Perform visualization and save checkpoints
                if (epoch + 1) % visualize_every == 0 or epoch == num_epochs - 1:
                    visualize_reconstructions(vae, epoch + 1, save_dir)
                    visualize_latent_space(vae, epoch + 1, save_dir)
                    torch.save(vae.state_dict(), f"{save_dir}/vae_epoch_{epoch + 1}.pt")

            return vae, loss_history

        # Train VAE
        vae, ae_losses = train_vae(
            vae,
            train_loader,  # Add this parameter
            num_epochs=30,
            lr=1e-4,
            lambda_cls=0.2,
            lambda_center=0.5,
            lambda_kl=0.01,
            visualize_every=15,
            save_dir=results_dir
        )

        # Save VAE
        torch.save(vae.state_dict(), vae_path)

        # Plot VAE loss
        plt.figure(figsize=(8, 5))
        # Plot each loss type separately
        plt.figure(figsize=(10, 6))
        plt.plot(ae_losses['total'], label='Total Loss')
        plt.plot(ae_losses['recon'], label='Reconstruction Loss')
        plt.plot(ae_losses['class'], label='Classification Loss')
        plt.plot(ae_losses['center'], label='Center Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Autoencoder Training Losses')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{results_dir}/autoencoder_losses.png")
        plt.close()

    # Create conditional UNet
    conditional_unet = ConditionalUNet(
        in_channels=3,  # 3 channels for RGB
        hidden_dims=[32, 64, 128],
        num_classes=len(class_names)  # 2 classes (cat/dog)
    ).to(device)

    # Initialize weights for UNet if needed
    def init_weights(m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, a=0.2)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    # Check if trained diffusion model exists
    if os.path.exists(diffusion_path):
        print(f"Loading existing diffusion model from {diffusion_path}")
        conditional_unet.load_state_dict(torch.load(diffusion_path, map_location=device))
        diffusion = ConditionalDenoiseDiffusion(conditional_unet, n_steps=1000, device=device)
    else:
        print("No existing diffusion model found. Training a new one...")
        conditional_unet.apply(init_weights)

        # Define train function
        def train_conditional_diffusion(autoencoder, unet, num_epochs=100, lr=1e-3, visualize_every=10,
                                        save_dir="./results"):
            print("Starting Class-Conditional Diffusion Model training...")
            os.makedirs(save_dir, exist_ok=True)

            autoencoder.eval()  # Set autoencoder to evaluation mode

            # Create diffusion model
            diffusion = ConditionalDenoiseDiffusion(unet, n_steps=1000, device=device)
            optimizer = torch.optim.AdamW(unet.parameters(), lr=lr, weight_decay=5e-4)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5
            )

            # Training loop
            loss_history = []

            for epoch in range(num_epochs):
                epoch_loss = 0

                for batch_idx, (data, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")):
                    data = data.to(device)
                    labels = labels.to(device)

                    # Encode images to latent space
                    with torch.no_grad():
                        # Handle the encode method which might return multiple values
                        encode_output = autoencoder.encode(data)

                        # Extract latent representation from tuple
                        if isinstance(encode_output, tuple):
                            latents = encode_output[0]  # Get the first element (latent z)
                        else:
                            latents = encode_output

                        # Reshape latents to spatial form if needed
                        if len(latents.shape) == 2:  # If latents is [B, latent_dim]
                            # Assuming a 256-dim latent vector needs to be reshaped to 3D for diffusion
                            # This is an approximation - adjust according to your diffusion model's expectations
                            latent_dim = latents.shape[1]
                            spatial_size = int((latent_dim / 3) ** 0.5)  # Calculate spatial dimension

                            if spatial_size ** 2 * 3 == latent_dim:  # If perfect square
                                latents = latents.view(-1, 3, spatial_size, spatial_size)
                            else:
                                # If not a perfect fit, use a different approach or fixed size
                                print(f"Warning: Latent dim {latent_dim} can't be reshaped perfectly to 3D")
                                # For example, reshape to fixed 8x8 and use only part of the latent
                                latents = latents[:, :192].view(-1, 3, 8, 8)  # Use first 192 dimensions

                    # Calculate diffusion loss with class conditioning
                    loss = diffusion.loss(latents, labels)

                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)
                    optimizer.step()

                    epoch_loss += loss.item()

                # Calculate average loss
                avg_loss = epoch_loss / len(train_loader)
                loss_history.append(avg_loss)
                print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.6f}")

                # Learning rate scheduling
                scheduler.step(avg_loss)

                # Visualize samples periodically
                if (epoch + 1) % visualize_every == 0 or epoch == num_epochs - 1:
                    # Generate samples for both classes
                    for class_idx in range(len(class_names)):
                        create_diffusion_animation(
                            autoencoder, diffusion, class_idx=class_idx, num_frames=50,
                            save_path=f"{save_dir}/diffusion_animation_class_{class_names[class_idx]}_epoch_{epoch + 1}.gif"
                        )
                        save_path = f"{save_dir}/sample_class_{class_names[class_idx]}_epoch_{epoch + 1}.png"
                        generate_class_samples(
                            autoencoder, diffusion, target_class=class_idx, num_samples=5, save_path=save_path
                        )
                        save_path = f"{save_dir}/denoising_path_{class_names[class_idx]}_epoch_{epoch + 1}.png"
                        visualize_denoising_steps(
                            autoencoder, diffusion, class_idx=class_idx, save_path=save_path
                        )

                    # Save checkpoint
                    torch.save(unet.state_dict(), f"{save_dir}/conditional_diffusion_epoch_{epoch + 1}.pt")

            return unet, diffusion, loss_history

        # Train conditional diffusion model
        conditional_unet, diffusion, diff_losses = train_conditional_diffusion(
            vae, conditional_unet, num_epochs=100, lr=1e-3,
            visualize_every=10,  # Visualize every 5 epochs
            save_dir=results_dir
        )

        # Save diffusion model
        torch.save(conditional_unet.state_dict(), diffusion_path)

        # Plot diffusion loss
        plt.figure(figsize=(8, 5))
        plt.plot(diff_losses)
        plt.title('Diffusion Model Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig(f"{results_dir}/diffusion_loss.png")
        plt.close()

    # Make sure diffusion is defined
    if 'diffusion' not in locals():
        diffusion = ConditionalDenoiseDiffusion(conditional_unet, n_steps=1000, device=device)

    # Generate sample grid for all classes
    print("Generating sample grid for both cat and dog classes...")
    grid_path = generate_samples_grid(vae, diffusion, n_per_class=5, save_dir=results_dir)
    print(f"Sample grid saved to: {grid_path}")

    # Generate denoising visualizations for all classes
    print("Generating denoising visualizations for cat and dog classes...")
    denoising_paths = []
    for class_idx in range(len(class_names)):
        save_path = f"{results_dir}/denoising_path_{class_names[class_idx]}_final.png"
        path = visualize_denoising_steps(vae, diffusion, class_idx, save_path=save_path)
        denoising_paths.append(path)
        print(f"Generated visualization for {class_names[class_idx]}")

    print("\nAll visualizations complete!")
    print(f"Sample grid: {grid_path}")
    print("Denoising visualizations:")
    for i, path in enumerate(denoising_paths):
        print(f"  - {class_names[i]}: {path}")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
