import torch
import torch.nn as nn
import torch.optim as optim
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
batch_size = 256

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
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # Global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


# Updated Center Loss for flat latent space
class CenterLoss(nn.Module):
    def __init__(self, num_classes=2, feat_dim=128, min_distance=1.0, repulsion_strength=1.0):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.min_distance = min_distance
        self.repulsion_strength = repulsion_strength

        # Initialize centers with better separation
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

        # Initialize centers further apart
        with torch.no_grad():
            if num_classes == 2:
                self.centers[0] = -torch.ones(feat_dim) / math.sqrt(feat_dim)
                self.centers[1] = torch.ones(feat_dim) / math.sqrt(feat_dim)
            else:
                for i in range(num_classes):
                    self.centers[i] = torch.randn(feat_dim)
                    self.centers[i] = self.centers[i] / torch.norm(self.centers[i]) * 2.0

    def compute_pairwise_distances(self, x, y):
        """Compute pairwise distances between two sets of vectors without using cdist"""
        n = x.size(0)
        m = y.size(0)

        # Compute squared norms
        x_norm = (x ** 2).sum(1).view(n, 1)
        y_norm = (y ** 2).sum(1).view(1, m)

        # Compute distance matrix using the formula: ||x-y||^2 = ||x||^2 + ||y||^2 - 2*x·y
        distmat = x_norm + y_norm - 2.0 * torch.mm(x, y.transpose(0, 1))

        # Apply sqrt and handle numerical stability
        distmat = torch.clamp(distmat, min=1e-12)
        distmat = torch.sqrt(distmat)

        return distmat

    def forward(self, x, labels):
        batch_size = x.size(0)

        # Calculate distance matrix between samples and centers
        distmat = self.compute_pairwise_distances(x, self.centers)

        # Get class mask
        classes = torch.arange(self.num_classes).to(labels.device)
        labels_expanded = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels_expanded.eq(classes.expand(batch_size, self.num_classes))

        # Apply mask and calculate attraction loss
        attraction_dist = distmat * mask.float()
        attraction_loss = attraction_dist.sum() / batch_size

        # Calculate repulsion between different class centers
        center_distances = self.compute_pairwise_distances(self.centers, self.centers)

        # Create a mask for different centers (all except diagonal)
        diff_mask = 1.0 - torch.eye(self.num_classes, device=x.device)

        # Calculate repulsion loss - penalize centers that are too close
        repulsion_loss = torch.clamp(self.min_distance - center_distances, min=0.0)
        repulsion_loss = (repulsion_loss * diff_mask).sum() / (self.num_classes * (self.num_classes - 1) + 1e-6)

        # Add an intra-class variance term to encourage diversity within each class
        intra_class_variance = 0.0
        for c in range(self.num_classes):
            class_mask = (labels == c)
            if torch.sum(class_mask) > 1:  # Need at least 2 samples for variance
                class_samples = x[class_mask]
                class_center = torch.mean(class_samples, dim=0)
                variance = torch.mean(torch.sum((class_samples - class_center) ** 2, dim=1))
                intra_class_variance += variance

        if self.num_classes > 0:
            intra_class_variance = intra_class_variance / self.num_classes

        # Final loss combines attraction, repulsion, and diversity
        total_loss = attraction_loss + self.repulsion_strength * repulsion_loss - 0.1 * intra_class_variance

        # Store metrics for monitoring (not used in loss calculation)
        with torch.no_grad():
            self.avg_center_dist = torch.sum(center_distances * diff_mask) / (
                        self.num_classes * (self.num_classes - 1) + 1e-6)
            self.avg_sample_dist = torch.mean(distmat)
            self.center_attraction = attraction_loss.item()
            self.center_repulsion = repulsion_loss.item()
            self.intra_variance = intra_class_variance.item() if isinstance(intra_class_variance,
                                                                            torch.Tensor) else intra_class_variance

        return total_loss


# Define Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out


# Enhanced Encoder with skip connections
# Define a simplified attention module that won't conflict with existing code
class SpatialAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SpatialAttention, self).__init__()
        # Global average pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Feature channel downscale and upscale
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


# Improved Encoder with appropriate capacity for CIFAR-10
# Improved Encoder with layer normalization
class Encoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim

        # More appropriate layer sizes for CIFAR-10 with layer norm
        self.conv_layers = nn.Sequential(
            # First block: 32x32x3 -> 16x16x64
            nn.Conv2d(in_channels, 64, 4, stride=2, padding=1),
            nn.GroupNorm(8, 64),  # Group norm with 8 groups
            nn.LeakyReLU(0.2, inplace=True),

            # Second block: 16x16x64 -> 8x8x128
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.GroupNorm(16, 128),  # Group norm with 16 groups
            nn.LeakyReLU(0.2, inplace=True),

            # Third block: 8x8x128 -> 4x4x256
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.GroupNorm(32, 256),  # Group norm with 32 groups
            nn.LeakyReLU(0.2, inplace=True),
        )

        # 256 channels * 4 * 4 = 4096 flattened features
        self.flattened_size = 256 * 4 * 4

        # Fully connected layers for mean and log variance with layer norm
        self.fc_mu = nn.Sequential(
            nn.Linear(self.flattened_size, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, latent_dim)
        )

        self.fc_logvar = nn.Sequential(
            nn.Linear(self.flattened_size, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, latent_dim)
        )

    def forward(self, x):
        # Pass through convolutional layers
        x = self.conv_layers(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Get mean and log variance
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        return mu, logvar


# Improved Decoder with layer normalization
class Decoder(nn.Module):
    def __init__(self, latent_dim=128, out_channels=3):
        super().__init__()
        self.latent_dim = latent_dim

        # Project from latent space to initial feature volume
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256 * 4 * 4),
            nn.LayerNorm(256 * 4 * 4),
            nn.LeakyReLU(0.2)
        )

        # Transposed convolutions for upsampling with group norm
        self.deconv_layers = nn.Sequential(
            # First block: 4x4x256 -> 8x8x128
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.GroupNorm(16, 128),  # Group norm with 16 groups
            nn.LeakyReLU(0.2, inplace=True),

            # Second block: 8x8x128 -> 16x16x64
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.GroupNorm(8, 64),  # Group norm with 8 groups
            nn.LeakyReLU(0.2, inplace=True),

            # Final block: 16x16x64 -> 32x32x3
            nn.ConvTranspose2d(64, out_channels, 4, stride=2, padding=1),
            nn.Sigmoid()  # Output in range [0, 1]
        )

    def forward(self, z):
        # Project and reshape
        x = self.fc(z)
        x = x.view(-1, 256, 4, 4)

        # Decode
        x = self.deconv_layers(x)
        return x


# Define Euclidean distance loss
def euclidean_distance_loss(x, y, reduction='mean'):
    """
    Calculate the Euclidean distance between x and y tensors.

    Args:
        x: First tensor
        y: Second tensor
        reduction: 'mean', 'sum', or 'none'

    Returns:
        Euclidean distance loss
    """
    # Calculate squared differences
    squared_diff = (x - y) ** 2

    # Sum across all dimensions except batch
    squared_dist = squared_diff.view(x.size(0), -1).sum(dim=1)

    # Take square root to get Euclidean distance
    euclidean_dist = torch.sqrt(squared_dist + 1e-8)  # Add small epsilon for numerical stability

    # Apply reduction
    if reduction == 'mean':
        return euclidean_dist.mean()
    elif reduction == 'sum':
        return euclidean_dist.sum()
    else:  # 'none'
        return euclidean_dist


# Improved SimpleAutoencoder (VAE) with appropriate capacity
class SimpleAutoencoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=128, num_classes=2, kl_weight=0.0005):
        super().__init__()
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight

        # Encoder and decoder
        self.encoder = Encoder(in_channels, latent_dim)
        self.decoder = Decoder(latent_dim, in_channels)

        # Classifier with layer norm
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, num_classes)
        )

        # For center loss
        self.register_buffer('class_centers', torch.zeros(num_classes, latent_dim))
        self.register_buffer('center_counts', torch.zeros(num_classes))

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        # Use proper initialization for all layers
        if isinstance(m, nn.Linear):
            # Use Kaiming initialization for all linear layers
            nn.init.kaiming_normal_(m.weight, a=0.2)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick with numerical stability improvements.
        """
        # Clamp logvar to prevent numerical issues
        logvar = torch.clamp(logvar, min=-2.0, max=10.0)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        """Encode input and sample from the latent distribution."""
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return z

    def encode_with_params(self, x):
        """Return distribution parameters for computing KL divergence loss."""
        mu, logvar = self.encoder(x)
        # Add stronger clamping here as well
        logvar = torch.clamp(logvar, min=-2.0, max=10.0)
        return mu, logvar

    def decode(self, z):
        """Decode latent vector to reconstruction."""
        return self.decoder(z)

    def classify(self, z):
        """Classify latent vector."""
        return self.classifier(z)

    def compute_center_loss(self, z, labels):
        """
        Simplified and numerically stable center loss computation.
        """
        centers_batch = self.class_centers[labels]

        # Calculate Euclidean distance for center loss
        squared_diff = (z - centers_batch) ** 2
        squared_dist = squared_diff.sum(dim=1)
        center_loss = torch.sqrt(squared_dist + 1e-8).mean()

        return center_loss

    def update_centers(self, z, labels, momentum=0.9):
        """
        Update class centers with moving average for stability.
        This should be called during training but not included in the loss graph.
        """
        unique_labels = torch.unique(labels)
        for label in unique_labels:
            mask = (labels == label)
            if mask.sum() > 0:
                class_samples = z[mask]
                class_mean = class_samples.mean(0)

                # Update with momentum
                old_center = self.class_centers[label]
                new_center = momentum * old_center + (1 - momentum) * class_mean
                self.class_centers[label] = new_center

    def kl_divergence(self, mu, logvar):
        """
        Compute KL divergence with numerical stability improvements.
        """
        if torch.isnan(mu).any() or torch.isnan(logvar).any():
            print("NaN detected in mu or logvar!")

        # Clamp values for stability
        mu = torch.clamp(mu, min=-10.0, max=10.0)
        logvar = torch.clamp(logvar, min=-2.0, max=10.0)

        # KL divergence formula
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)

        # Further stability: clip extreme values and take mean
        kl_loss = torch.clamp(kl_loss, min=0.0, max=100.0).mean()
        mu_reg = 1e-4 * torch.sum(mu.pow(2))  # L2 regularization on mu
        return kl_loss + mu_reg

    def forward(self, x):
        """Complete forward pass."""
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar, z


# Swish activation function
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


# Time embedding for diffusion model
# Revised TimeEmbedding for flat latent space
class TimeEmbedding(nn.Module):
    def __init__(self, n_channels=128):
        super().__init__()
        self.n_channels = n_channels
        self.lin1 = nn.Linear(self.n_channels, self.n_channels * 2)
        self.act = Swish()
        self.lin2 = nn.Linear(self.n_channels * 2, self.n_channels)

    def forward(self, t):
        # Sinusoidal time embedding similar to positional encoding
        half_dim = self.n_channels // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)

        # Make sure emb has the right dimension
        if emb.shape[1] < self.n_channels:
            padding = torch.zeros(emb.shape[0], self.n_channels - emb.shape[1], device=emb.device)
            emb = torch.cat([emb, padding], dim=1)

        # Process through MLP
        return self.lin2(self.act(self.lin1(emb)))


# Revised ClassEmbedding for flat latent space
class ClassEmbedding(nn.Module):
    def __init__(self, num_classes=2, n_channels=128):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, n_channels)
        self.lin1 = nn.Linear(n_channels, n_channels)
        self.act = Swish()
        self.lin2 = nn.Linear(n_channels, n_channels)

    def forward(self, c):
        # Get class embeddings
        emb = self.embedding(c)
        # Process through MLP
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
        output = self.proj(out) + residual

        return output


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


class ConditionalUNet(nn.Module):
    def __init__(self, latent_dim=128, hidden_dims=[256, 512, 256],
                 time_emb_dim=128, num_classes=2, dropout_rate=0.3):
        super().__init__()
        self.latent_dim = latent_dim
        self.time_emb_dim = time_emb_dim

        # Time and class embeddings (both with same embedding dimension)
        self.time_emb = TimeEmbedding(n_channels=time_emb_dim)
        self.class_emb = ClassEmbedding(num_classes=num_classes, n_channels=time_emb_dim)

        # Initial projection from latent space to first hidden dimension
        self.latent_proj = nn.Linear(latent_dim, hidden_dims[0])

        # Create a list of linear layers to project time (and class) embeddings for each layer.
        # We need one for each layer in hidden_dims.
        self.time_projections = nn.ModuleList([
            nn.Linear(time_emb_dim, dim) for dim in hidden_dims
        ])

        # Build a stack of MLP layers.
        # Each layer is a simple sequential block: a linear layer with dropout and GELU,
        # followed by a projection to the next hidden dimension.
        self.layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            block = nn.Sequential(
                nn.Linear(hidden_dims[i], hidden_dims[i]),
                nn.LayerNorm(hidden_dims[i]),
                nn.Dropout(dropout_rate),
                nn.GELU()
            )
            proj = nn.Linear(hidden_dims[i], hidden_dims[i + 1])
            self.layers.append(nn.ModuleList([block, proj]))

        # One more time projection for the final conditioning
        self.final_time_proj = nn.Linear(time_emb_dim, hidden_dims[-1])
        self.final_class_proj = nn.Linear(time_emb_dim, hidden_dims[-1])
        # Final projection from last hidden dimension back to latent dimension
        self.final = nn.Linear(hidden_dims[-1], latent_dim)

        self.final_norm = nn.LayerNorm(latent_dim)

    def forward(self, x, t, c=None, res_scale=1.0):
        # x: [batch_size, latent_dim] (noisy latent vector)
        # t: diffusion timestep tensor
        # c: optional class labels

        # Save original input for the global residual connection.
        residual = x

        # Base time embedding and (if provided) class embedding.
        t_emb_base = self.time_emb(t)
        c_emb_base = self.class_emb(c) if c is not None else None

        # Project input into hidden dimension.
        h = self.latent_proj(x)

        # Process through each layer.
        for i, (block, downsample) in enumerate(self.layers):
            # Add time conditioning (and class conditioning if available).
            t_emb = self.time_projections[i](t_emb_base)
            h = h + t_emb
            if c_emb_base is not None:
                c_emb = self.time_projections[i](c_emb_base)
                h = h + c_emb

            # Pass through the block and then downsample.
            h = block(h)
            h = downsample(h)

        # Final conditioning before mapping back to latent space.
        t_emb_final = self.final_time_proj(t_emb_base)
        h = h + t_emb_final
        if c_emb_base is not None:
            c_emb_final = self.final_class_proj(c_emb_base)
            h = h + c_emb_final

        # Map back to latent dimension and add the global skip connection.
        out = self.final(h)
        return (out + self.final(residual) * res_scale) / (1 + res_scale)


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

        alpha_bar_t = self.alpha_bar[t].reshape(-1, 1)
        return torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * eps

    def p_sample(self, xt, t, c=None):
        """Single denoising step with optional class conditioning"""
        # Convert time to tensor format expected by model
        if not isinstance(t, torch.Tensor):
            t = torch.tensor([t], device=xt.device)

        # Predict noise (with class conditioning if provided)
        eps_theta = self.eps_model(xt, t, c)

        # Get alpha values
        alpha_t = self.alpha[t].reshape(-1, 1)
        alpha_bar_t = self.alpha_bar[t].reshape(-1, 1)

        # Calculate mean
        mean = (xt - (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t) * eps_theta) / torch.sqrt(alpha_t)

        # Add noise if not the final step
        var = self.beta[t].reshape(-1, 1)

        if t[0] > 0:
            noise = torch.randn_like(xt)
            return mean + torch.sqrt(var) * noise
        else:
            return mean

    def sample(self, shape, device, c=None):
        """Generate samples by denoising from pure noise with optional class conditioning"""
        # Start from pure noise
        x = torch.randn(shape, device=device)

        # Progressively denoise with class conditioning
        for t in tqdm(reversed(range(self.n_steps)), desc="Sampling"):
            x = self.p_sample(x, t, c)

        return x

    def loss(self, x0, labels=None, res_scale=1.0):
        batch_size = x0.shape[0]
        t = torch.randint(0, self.n_steps, (batch_size,), device=x0.device, dtype=torch.long)
        eps = torch.randn_like(x0)
        xt = self.q_sample(x0, t, eps)
        # Pass the residual scaling factor into the eps_model (i.e. your conditional UNet)
        eps_theta = self.eps_model(xt, t, labels, res_scale=res_scale)
        return euclidean_distance_loss(eps, eps_theta)


# Function to generate a grid of samples for all classes
def generate_samples_grid(autoencoder, diffusion, n_per_class=5, save_dir="./results"):
    """Generate a grid of samples with n_per_class samples for each class using VAE decoder"""
    os.makedirs(save_dir, exist_ok=True)
    device = next(autoencoder.parameters()).device

    # Set models to evaluation mode
    autoencoder.eval()
    diffusion.eps_model.eval()

    n_classes = len(class_names)
    # Create figure with extra column for class labels
    fig, axes = plt.subplots(n_classes, n_per_class + 1, figsize=((n_per_class + 1) * 2, n_classes * 2))

    # Add a title to explain what the figure shows
    fig.suptitle(f'CIFAR Cat and Dog Samples Generated by VAE-Diffusion Model',
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

        # Use flat latent shape
        latent_shape = (n_per_class, autoencoder.latent_dim)

        # Sample from the diffusion model with class conditioning
        samples = diffusion.sample(latent_shape, device, class_tensor)

        # Decode samples using VAE decoder
        with torch.no_grad():
            decoded = autoencoder.decode(samples)

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
        "This visualization shows cat and dog images generated by the conditional diffusion model using a VAE decoder.\n"
        "The model creates new, synthetic images based on learned patterns from CIFAR-10.\n"
        "Each row corresponds to a different animal category as labeled."
    )
    plt.figtext(0.5, 0.01, description, ha='center', fontsize=10,
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7))

    plt.tight_layout(rect=[0, 0.03, 1, 0.92])  # Adjust layout to make room for titles
    plt.savefig(f"{save_dir}/vae_samples_grid_all_classes.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Set models back to training mode
    autoencoder.train()
    diffusion.eps_model.train()

    print(f"Generated sample grid for all classes with clearly labeled categories")
    return f"{save_dir}/vae_samples_grid_all_classes.png"


# Visualize latent space denoising process for a specific class
# Modified visualize_denoising_steps for VAE
def visualize_denoising_steps(autoencoder, diffusion, class_idx, save_path=None):
    """
    Visualize both the denoising process and the corresponding path in latent space for VAE.

    Args:
        autoencoder: Trained VAE model
        diffusion: Trained diffusion model
        class_idx: Target class index (0-1 for cat/dog)
        save_path: Path to save the visualization
    """
    device = next(autoencoder.parameters()).device

    # Set models to evaluation mode
    autoencoder.eval()
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
            # For VAE, use the mean vectors for consistency
            mu, logvar = autoencoder.encode_with_params(images)
            all_latents.append(mu.detach().cpu().numpy())
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
    # Use flat latent dimension
    x = torch.randn((n_samples, autoencoder.latent_dim), device=device)

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
            path_latents.append(current_x[0:1].detach().cpu().numpy())

            # Decode to images using VAE decoder
            decoded = autoencoder.decode(current_x)

            # Add to samples
            samples_per_step.append(decoded.cpu())

        # Add final denoised state to path
        path_latents.append(current_x[0:1].detach().cpu().numpy())

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
    ax_denoising.set_title(f"VAE-Diffusion Model Denoising Process for {class_names[class_idx]}",
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

    ax_latent.set_title(f"VAE-Diffusion Path in Latent Space for {class_names[class_idx]}", fontsize=16)
    ax_latent.legend(fontsize=10, loc='best')
    ax_latent.grid(True, linestyle='--', alpha=0.7)

    # Add explanatory text
    plt.figtext(
        0.5, 0.01,
        "This visualization shows the VAE-based denoising process (top) and the corresponding path in latent space (bottom).\n"
        "The first row of images (highlighted in red) corresponds to the red path in the latent space plot below.",
        ha='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

    # Save the figure
    fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0.3, wspace=0.1)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"VAE Denoising visualization for {class_names[class_idx]} saved to {save_path}")

    # Set models back to training mode
    autoencoder.train()
    diffusion.eps_model.train()

    return save_path


# Updated visualize_reconstructions for VAE
def visualize_reconstructions(autoencoder, epoch, save_dir="./results"):
    """Visualize original and reconstructed images at each epoch for VAE"""
    os.makedirs(save_dir, exist_ok=True)
    device = next(autoencoder.parameters()).device

    # Get a batch of test data
    cifar_test = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    test_dataset = create_cat_dog_dataset(cifar_test)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True)

    test_images, test_labels = next(iter(test_loader))
    test_images = test_images.to(device)

    # Generate reconstructions with VAE
    autoencoder.eval()
    with torch.no_grad():
        # Get reconstruction with VAE forward pass
        # Forward pass using the updated autoencoder architecture
        mu, logvar = autoencoder.encode_with_params(test_images)
        z = autoencoder.reparameterize(mu, logvar)
        reconstructed = autoencoder.decode(z)

    # Create visualization
    fig, axes = plt.subplots(2, 8, figsize=(16, 4))

    for i in range(8):
        # Original
        img = test_images[i].cpu().permute(1, 2, 0).numpy()  # Convert from [C,H,W] to [H,W,C]
        axes[0, i].imshow(img)  # No cmap for RGB images
        axes[0, i].set_title(f'Original: {class_names[test_labels[i]]}')
        axes[0, i].axis('off')

        # Reconstruction
        recon_img = reconstructed[i].cpu().permute(1, 2, 0).numpy()  # Convert from [C,H,W] to [H,W,C]
        axes[1, i].imshow(recon_img)  # No cmap for RGB images
        axes[1, i].set_title('Reconstruction')
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.savefig(f"{save_dir}/vae_reconstruction_epoch_{epoch}.png")
    plt.close()
    autoencoder.train()


# Visualize latent space with t-SNE
def visualize_latent_space(autoencoder, epoch, save_dir="./results"):
    """Visualize the latent space of the VAE using t-SNE"""
    os.makedirs(save_dir, exist_ok=True)
    device = next(autoencoder.parameters()).device

    # Get test data
    cifar_test = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    test_dataset = create_cat_dog_dataset(cifar_test)
    test_loader = DataLoader(test_dataset, batch_size=500, shuffle=False)

    # Extract features and labels
    autoencoder.eval()
    all_latents = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            # For VAE, we want the mean vectors for consistency
            # Unpack the three returned values properly
            mu, logvar = autoencoder.encode_with_params(images)
            all_latents.append(mu.cpu().numpy())
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

        plt.title(f"t-SNE Visualization of VAE Latent Space (Epoch {epoch})")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/vae_latent_space_epoch_{epoch}.png")
        plt.close()
    except Exception as e:
        print(f"t-SNE visualization error: {e}")

    autoencoder.train()


# Function to generate samples of a specific class (need this for training)
# Updated generate_class_samples for VAE
def generate_class_samples(autoencoder, diffusion, target_class, num_samples=5, save_path=None):
    """
    Generate samples of a specific target class using the VAE decoder

    Args:
        autoencoder: Trained VAE model
        diffusion: Trained conditional diffusion model
        target_class: Index of the target class (0-1) or class name
        num_samples: Number of samples to generate
        save_path: Path to save the generated samples

    Returns:
        Tensor of generated samples
    """
    device = next(autoencoder.parameters()).device

    # Set models to evaluation mode
    autoencoder.eval()
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
    latent_shape = (num_samples, autoencoder.latent_dim)
    with torch.no_grad():
        # Sample from the diffusion model with class conditioning
        latent_samples = diffusion.sample(latent_shape, device, class_tensor)

        # Decode latents to images using VAE decoder
        samples = autoencoder.decode(latent_samples)

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


# Modified create_diffusion_animation for flat latent space
def create_diffusion_animation(autoencoder, diffusion, class_idx, num_frames=50, seed=42,
                               save_path=None, temp_dir=None, fps=10, reverse=False):
    """
    Create a GIF animation showing the diffusion process.

    Args:
        autoencoder: Trained autoencoder model
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
    device = next(autoencoder.parameters()).device

    # Set models to evaluation mode
    autoencoder.eval()
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
        # Start from pure noise - now using flat latent dimension
        x = torch.randn((1, autoencoder.latent_dim), device=device)

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
                alpha_bar_t = diffusion.alpha_bar[t].reshape(-1, 1)
                current_x = torch.sqrt(alpha_bar_t) * current_x + torch.sqrt(1 - alpha_bar_t) * eps

            # Decode to image
            decoded = autoencoder.decode(current_x)

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


def compute_res_scale(epoch, total_epochs, initial=5.0, final=1.0):
    # Linear decay: at epoch 0, returns initial; at final epoch, returns final.
    if epoch <= 20:
        return 0.1
    return initial + (final - initial) * (epoch / total_epochs)


def cyclical_annealing(epoch, cycle_length, kl_weight_start, kl_weight_end):
    cycle_position = (epoch % cycle_length) / cycle_length

    if cycle_position < 0.5:
        weight = kl_weight_start + (kl_weight_end - kl_weight_start) * (2.0 * cycle_position)
    else:
        weight = kl_weight_end

    return weight

# Main function
def main():
    """Main function to run the entire pipeline non-interactively"""
    print("Starting class-conditional diffusion model for CIFAR cats and dogs")

    # Set device
    device = torch.device("mps" if torch.backends.mps.is_available() else
                         ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    # Create results directory
    results_dir = "./cifar_cat_dog_conditional_v6_3"
    os.makedirs(results_dir, exist_ok=True)

    # Load CIFAR-10 dataset and filter for cats and dogs
    print("Loading and filtering CIFAR-10 dataset for cats and dogs...")
    cifar_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_dataset = create_cat_dog_dataset(cifar_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    # Paths for saved models
    autoencoder_path = f"{results_dir}/cifar_cat_dog_autoencoder.pt"
    diffusion_path = f"{results_dir}/conditional_diffusion_final.pt"

    # Create autoencoder
    autoencoder = SimpleAutoencoder(in_channels=3, latent_dim=128).to(device)  # 3 channels for RGB

    # Check if trained autoencoder exists
    if os.path.exists(autoencoder_path):
        print(f"Loading existing autoencoder from {autoencoder_path}")
        autoencoder.load_state_dict(torch.load(autoencoder_path, map_location=device), strict=False)
        autoencoder.eval()
    else:
        print("No existing autoencoder found. Training a new one...")

        # Updated training function for the VAE
        def train_autoencoder(autoencoder, train_loader, num_epochs=300, lr=1e-4,
                              lambda_cls=0.1, lambda_center=0.05,
                              kl_weight_start=0.0001, kl_weight_end=0.05,
                              visualize_every=10, save_dir="./results"):
            """Improved VAE training function with better stability."""
            print("Starting VAE training with improved stability...")
            os.makedirs(save_dir, exist_ok=True)
            device = next(autoencoder.parameters()).device

            # Adam optimizer with conservative settings
            optimizer = optim.Adam(autoencoder.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=1e-6)

            # Learning rate scheduler for stability
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.7, patience=5, verbose=True, min_lr=1e-6
            )

            loss_history = {'total': [], 'recon': [], 'kl': [], 'class': [], 'center': []}

            # Initialize best loss for model saving
            best_loss = float('inf')

            for epoch in range(num_epochs):
                autoencoder.train()
                epoch_recon_loss = 0
                epoch_kl_loss = 0
                epoch_class_loss = 0
                epoch_center_loss = 0
                epoch_total_loss = 0

                # Apply a simple linear annealing schedule
                kl_weight = min(kl_weight_end, kl_weight_start + (epoch / 100) * (kl_weight_end - kl_weight_start))
                autoencoder.kl_weight = kl_weight

                print(f"Epoch {epoch + 1}/{num_epochs} - KL Weight: {kl_weight:.6f}")

                for batch_idx, (data, labels) in enumerate(tqdm(train_loader, desc=f"Training")):
                    data = data.to(device)
                    labels = labels.to(device)

                    # Forward pass
                    optimizer.zero_grad()
                    recon_x, mu, logvar, z = autoencoder(data)

                    # Multi-phase training strategy
                    if epoch < 20:
                        # Phase 1: Only reconstruction loss - stabilize decoder
                        recon_loss = euclidean_distance_loss(recon_x, data)
                        kl_loss = torch.tensor(0.0, device=device)
                        class_loss = torch.tensor(0.0, device=device)
                        center_loss = torch.tensor(0.0, device=device)
                        total_loss = recon_loss

                    elif epoch < 40:
                        # Phase 2: Add small KL - start learning distribution
                        recon_loss = euclidean_distance_loss(recon_x, data)
                        kl_loss = autoencoder.kl_divergence(mu, logvar)
                        class_loss = torch.tensor(0.0, device=device)
                        center_loss = torch.tensor(0.0, device=device)
                        total_loss = recon_loss + kl_weight * kl_loss

                    elif epoch < 60:
                        # Phase 3: Add classification
                        recon_loss = euclidean_distance_loss(recon_x, data)
                        kl_loss = autoencoder.kl_divergence(mu, logvar)
                        class_logits = autoencoder.classify(z)
                        class_loss = F.cross_entropy(class_logits, labels)
                        center_loss = torch.tensor(0.0, device=device)
                        total_loss = recon_loss + kl_weight * kl_loss + lambda_cls * class_loss

                    else:
                        # Phase 4: Full training with all components
                        recon_loss = euclidean_distance_loss(recon_x, data)
                        kl_loss = autoencoder.kl_divergence(mu, logvar)
                        class_logits = autoencoder.classify(z)
                        class_loss = F.cross_entropy(class_logits, labels)
                        center_loss = autoencoder.compute_center_loss(z, labels)
                        total_loss = recon_loss + kl_weight * kl_loss + lambda_cls * class_loss + lambda_center * center_loss

                    # Skip problematic batches
                    if torch.isnan(total_loss) or torch.isinf(total_loss):
                        print(f"WARNING: Invalid loss detected in batch {batch_idx}. Skipping.")
                        continue

                    # Backward pass with gradient clipping
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), max_norm=1.0)
                    optimizer.step()

                    # Update centers (detached from computational graph)
                    with torch.no_grad():
                        if epoch >= 20:
                            autoencoder.update_centers(z.detach(), labels, momentum=0.9)

                    # Log losses
                    epoch_recon_loss += recon_loss.item()
                    epoch_kl_loss += kl_loss.item() if isinstance(kl_loss, torch.Tensor) else 0
                    epoch_class_loss += class_loss.item() if isinstance(class_loss, torch.Tensor) else 0
                    epoch_center_loss += center_loss.item() if isinstance(center_loss, torch.Tensor) else 0
                    epoch_total_loss += total_loss.item()

                # Calculate average losses
                num_batches = len(train_loader)
                avg_recon_loss = epoch_recon_loss / num_batches
                avg_kl_loss = epoch_kl_loss / num_batches
                avg_class_loss = epoch_class_loss / num_batches
                avg_center_loss = epoch_center_loss / num_batches
                avg_total_loss = epoch_total_loss / num_batches

                # Store losses in history
                loss_history['recon'].append(avg_recon_loss)
                loss_history['kl'].append(avg_kl_loss)
                loss_history['class'].append(avg_class_loss)
                loss_history['center'].append(avg_center_loss)
                loss_history['total'].append(avg_total_loss)

                # Print epoch summary
                print(f"Epoch {epoch + 1}/{num_epochs}, "
                      f"Total Loss: {avg_total_loss:.6f}, "
                      f"Recon Loss: {avg_recon_loss:.6f}, "
                      f"KL Loss: {avg_kl_loss:.6f}, "
                      f"Class Loss: {avg_class_loss:.6f}, "
                      f"Center Loss: {avg_center_loss:.6f}")

                # Update learning rate based on total loss
                scheduler.step(avg_total_loss)

                # Save best model
                if avg_total_loss < best_loss:
                    best_loss = avg_total_loss
                    torch.save(autoencoder.state_dict(), f"{save_dir}/vae_best.pt")
                    print(f"Saved best model with loss: {best_loss:.6f}")

                # Visualizations and regular checkpoints
                if (epoch + 1) % visualize_every == 0 or epoch == num_epochs - 1:
                    visualize_reconstructions(autoencoder, epoch + 1, save_dir)
                    visualize_latent_space(autoencoder, epoch + 1, save_dir)
                    torch.save(autoencoder.state_dict(), f"{save_dir}/vae_epoch_{epoch + 1}.pt")

                # Early stopping check (optional)
                if optimizer.param_groups[0]['lr'] <= 1e-6 and epoch > num_epochs // 2:
                    print(f"Learning rate too small, stopping training at epoch {epoch + 1}")
                    break

            # Save final model
            torch.save(autoencoder.state_dict(), f"{save_dir}/vae_final.pt")
            print(f"Saved final model after {num_epochs} epochs")

            return autoencoder, loss_history

        # Train autoencoder
        autoencoder, ae_losses = train_autoencoder(
            autoencoder,
            train_loader,
            num_epochs=1000,
            lr=1e-4,
            lambda_cls=5.0,
            lambda_center=1.0,
            visualize_every=10,
            save_dir=results_dir
        )

        # Save autoencoder
        torch.save(autoencoder.state_dict(), autoencoder_path)

        # Plot autoencoder loss
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
        hidden_dims=[128, 256, 512, 256, 128],
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

        # Updated train_conditional_diffusion for VAE
        def train_conditional_diffusion(autoencoder, unet, num_epochs=100, lr=1e-3, visualize_every=10,
                                        save_dir="./results"):
            print("Starting Class-Conditional Diffusion Model training with VAE...")
            os.makedirs(save_dir, exist_ok=True)

            autoencoder.eval()  # Set VAE to evaluation mode

            # Create diffusion model
            diffusion = ConditionalDenoiseDiffusion(unet, n_steps=1000, device=device)
            optimizer = torch.optim.AdamW(unet.parameters(), lr=lr, weight_decay=1e-5)
            # T_0: number of iterations (or epochs) before the first restart.
            # T_mult: factor by which the period increases after each restart.
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

            # Training loop
            loss_history = []

            for epoch in range(num_epochs):
                # Compute the current residual scaling factor
                current_res_scale = compute_res_scale(epoch, num_epochs, initial=0.5, final=1.1)
                epoch_loss = 0

                for batch_idx, (data, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")):
                    data = data.to(device)
                    labels = labels.to(device)

                    # Encode images to latent space using VAE
                    with torch.no_grad():
                        mu, logvar = autoencoder.encode_with_params(data)
                        z = autoencoder.reparameterize(mu, logvar)

                    # Calculate diffusion loss with class conditioning
                    loss = diffusion.loss(z, labels, res_scale=current_res_scale)

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
                scheduler.step(int(epoch + batch_idx / len(train_loader)))

                # Visualize samples periodically
                if (epoch + 1) % visualize_every == 0 or epoch == num_epochs - 1:
                    # Generate samples for both classes
                    for class_idx in range(len(class_names)):
                        create_diffusion_animation(autoencoder, diffusion, class_idx=class_idx, num_frames=50,
                                                   save_path=f"{save_dir}/diffusion_animation_class_{class_names[class_idx]}_epoch_{epoch + 1}.gif")
                        save_path = f"{save_dir}/sample_class_{class_names[class_idx]}_epoch_{epoch + 1}.png"
                        generate_class_samples(autoencoder, diffusion, target_class=class_idx, num_samples=5,
                                               save_path=save_path)
                        save_path = f"{save_dir}/denoising_path_{class_names[class_idx]}_epoch_{epoch + 1}.png"
                        visualize_denoising_steps(autoencoder, diffusion, class_idx=class_idx, save_path=save_path)

                    # Save checkpoint
                    torch.save(unet.state_dict(), f"{save_dir}/conditional_diffusion_epoch_{epoch + 1}.pt")

            return unet, diffusion, loss_history

        # Train conditional diffusion model
        conditional_unet, diffusion, diff_losses = train_conditional_diffusion(
            autoencoder, conditional_unet, num_epochs=500, lr=1e-3,
            visualize_every=10,  # Visualize every 10 epochs
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
    grid_path = generate_samples_grid(autoencoder, diffusion, n_per_class=5, save_dir=results_dir)
    print(f"Sample grid saved to: {grid_path}")

    # Generate denoising visualizations for all classes
    print("Generating denoising visualizations for cat and dog classes...")
    denoising_paths = []
    for class_idx in range(len(class_names)):
        save_path = f"{results_dir}/denoising_path_{class_names[class_idx]}_final.png"
        path = visualize_denoising_steps(autoencoder, diffusion, class_idx, save_path=save_path)
        denoising_paths.append(path)
        print(f"Generated visualization for {class_names[class_idx]}")

    print("\nAll visualizations complete!")
    print(f"Sample grid: {grid_path}")
    print("Denoising visualizations:")
    for i, path in enumerate(denoising_paths):
        print(f"  - {class_names[i]}: {path}")


if __name__ == "__main__":
    main()
