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
from torch.optim.lr_scheduler import OneCycleLR
from torch.nn.utils import spectral_norm

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Set image size for CIFAR (32x32 RGB images)
img_size = 32

# Data preprocessing with additional augmentation
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
])

# Batch size for training
batch_size = 128

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


# Improved Euclidean distance loss with robust normalization
def euclidean_distance_loss(x, y, reduction='mean'):
    """
    Calculate the Euclidean distance between x and y tensors with improved stability.

    Args:
        x: First tensor
        y: Second tensor
        reduction: 'mean', 'sum', or 'none'

    Returns:
        Euclidean distance loss
    """
    # Normalize inputs to prevent numerical instability
    x_norm = F.normalize(x, p=2, dim=-1, eps=1e-12)
    y_norm = F.normalize(y, p=2, dim=-1, eps=1e-12)

    # Calculate squared differences
    squared_diff = (x_norm - y_norm) ** 2

    # Sum across all dimensions except batch
    squared_dist = squared_diff.view(x.size(0), -1).sum(dim=1)

    # Take square root to get Euclidean distance
    euclidean_dist = torch.sqrt(squared_dist + 1e-12)  # Increased epsilon for stability

    # Apply reduction
    if reduction == 'mean':
        return euclidean_dist.mean()
    elif reduction == 'sum':
        return euclidean_dist.sum()
    else:  # 'none'
        return euclidean_dist


# Swish activation function
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


# SiLU is PyTorch's built-in version of Swish
silu = nn.SiLU()


# Channel Attention Layer with improved stability
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # Global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            spectral_norm(nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias)),
            nn.SiLU(inplace=True),
            spectral_norm(nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias)),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


# Improved Convolutional Attention Block with gating
class CAB(nn.Module):
    def __init__(self, n_feat, reduction=16, bias=False):
        super(CAB, self).__init__()
        self.body = nn.Sequential(
            spectral_norm(nn.Conv2d(n_feat, n_feat, 3, padding=1, bias=bias)),
            nn.GroupNorm(min(8, n_feat), n_feat),  # More stable than BatchNorm
            nn.SiLU(inplace=True),
            spectral_norm(nn.Conv2d(n_feat, n_feat, 3, padding=1, bias=bias)),
        )
        self.ca = CALayer(n_feat, reduction, bias=bias)
        self.gate = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, 1, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        res = self.body(x)
        res = self.ca(res)
        gate = self.gate(x)
        return x + res * gate  # Gated residual connection


# Enhanced Encoder network with spectral normalization for stability
class Encoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=256):
        super().__init__()
        self.latent_dim = latent_dim

        # Improved backbone with spectral normalization and group norm
        self.backbone = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, 16, 3, stride=1, padding=1)),
            nn.GroupNorm(4, 16),
            CAB(16, reduction=4),
            nn.SiLU(inplace=True),

            spectral_norm(nn.Conv2d(16, 32, 4, stride=2, padding=1)),  # 32x32 -> 16x16
            nn.GroupNorm(8, 32),

            spectral_norm(nn.Conv2d(32, 64, 3, stride=1, padding=1)),
            nn.GroupNorm(8, 64),
            CAB(64, reduction=8),
            nn.SiLU(inplace=True),

            spectral_norm(nn.Conv2d(64, 128, 4, stride=2, padding=1)),  # 16x16 -> 8x8
            nn.GroupNorm(8, 128),

            spectral_norm(nn.Conv2d(128, 64, 3, stride=1, padding=1)),
            nn.GroupNorm(8, 64),
            nn.SiLU(inplace=True),
        )

        # Flattened size: 64 * 8 * 8 = 4096
        self.flattened_size = 64 * 8 * 8

        # Improved projection layers with dropout for better regularization
        self.fc_mu = nn.Sequential(
            spectral_norm(nn.Linear(self.flattened_size, 512)),
            nn.LayerNorm(512),  # LayerNorm instead of BatchNorm for stability
            nn.SiLU(inplace=True),
            nn.Dropout(0.1),  # Small dropout helps regularization
            nn.Linear(512, latent_dim)
        )

        self.fc_logvar = nn.Sequential(
            spectral_norm(nn.Linear(self.flattened_size, 512)),
            nn.LayerNorm(512),
            nn.SiLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, latent_dim)
        )

    def forward(self, x):
        features = self.backbone(x)
        flattened = features.view(features.size(0), -1)

        # Return both mean and log variance
        mu = self.fc_mu(flattened)
        logvar = self.fc_logvar(flattened)

        # Constrain the range of logvar for stability
        logvar = torch.clamp(logvar, min=-20.0, max=2.0)

        return mu, logvar


# Enhanced Decoder with more complex architecture for better reconstruction
class Decoder(nn.Module):
    def __init__(self, latent_dim=256, out_channels=3):
        super().__init__()
        self.latent_dim = latent_dim

        # Initial fully connected layer from latent space to spatial features
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.LayerNorm(512),
            nn.SiLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, 64 * 8 * 8)
        )

        # Enhanced decoder network with skip connections and residual blocks
        self.initial_conv = nn.Sequential(
            spectral_norm(nn.Conv2d(64, 128, 3, stride=1, padding=1)),
            nn.GroupNorm(8, 128),
            nn.SiLU(inplace=True)
        )

        # Residual block 1
        self.res_block1 = nn.Sequential(
            CAB(128, reduction=8),
            nn.SiLU(inplace=True),
            CAB(128, reduction=8),
            nn.SiLU(inplace=True)
        )

        # First upsampling: 8x8 -> 16x16
        self.upsample1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(inplace=True)
        )

        # Residual block 2
        self.res_block2 = nn.Sequential(
            CAB(64, reduction=8),
            nn.SiLU(inplace=True),
            CAB(64, reduction=8),
            nn.SiLU(inplace=True)
        )

        # Second upsampling: 16x16 -> 32x32
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(inplace=True)
        )

        # Final refinement layers
        self.final_block = nn.Sequential(
            CAB(32, reduction=4),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.GroupNorm(4, 16),
            nn.SiLU(inplace=True),
            nn.Conv2d(16, out_channels, 3, padding=1),
            nn.Sigmoid()  # Output activation for [0,1] range
        )

    def forward(self, z):
        # z is [batch_size, latent_dim]
        features = self.fc(z)
        features = features.view(-1, 64, 8, 8)  # Reshape to [batch_size, 64, 8, 8]

        # Apply decoder layers with residual connections
        x = self.initial_conv(features)
        x = x + self.res_block1(x)  # Residual connection

        x = self.upsample1(x)
        x = x + self.res_block2(x)  # Residual connection

        x = self.upsample2(x)
        x = self.final_block(x)

        return x


# Improved Center Loss for better class separation in latent space
class CenterLoss(nn.Module):
    def __init__(self, num_classes=2, feat_dim=256, device='mps'):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device

        # Initialize centers with Xavier/Glorot initialization
        self.centers = nn.Parameter(
            torch.randn(num_classes, feat_dim).to(device) * 0.02
        )

        # Register a buffer for center updates
        self.register_buffer('classes_batch_count',
                             torch.zeros(num_classes).to(device))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim)
            labels: ground truth labels with shape (batch_size)
        """
        batch_size = x.size(0)

        # Apply L2 normalization to features and centers
        x_norm = F.normalize(x, p=2, dim=1, eps=1e-12)
        centers_norm = F.normalize(self.centers, p=2, dim=1, eps=1e-12)

        # Compute distances more efficiently and numerically stable
        # 1 - cosine similarity as distance
        distmat = 1.0 - torch.mm(x_norm, centers_norm.t())

        # Get class mask using one-hot encoding
        mask = torch.zeros(batch_size, self.num_classes, device=x.device)
        mask.scatter_(1, labels.unsqueeze(1), 1)

        # Apply mask and calculate loss
        dist = distmat * mask.float()
        loss = dist.sum() / batch_size

        return loss

    def update_centers(self, features, labels, alpha=0.1):
        """
        Update centers according to the batch features (optional)
        Args:
            features: feature matrix with shape (batch_size, feat_dim)
            labels: ground truth labels with shape (batch_size)
            alpha: learning rate for centers
        """
        # For each class in the batch, accumulate and update centers
        batch_size = features.size(0)

        for i in range(batch_size):
            self.classes_batch_count[labels[i]] += 1
            delta_c = features[i] - self.centers[labels[i]]
            alpha_c = alpha / (1.0 + self.classes_batch_count[labels[i]])
            self.centers[labels[i]] += alpha_c * delta_c


# Modified SimpleAutoencoder to implement VAE functionality
class SimpleAutoencoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=256, num_classes=2, kl_weight=0.001, device='mps'):
        super().__init__()
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight
        self.device = device

        # Updated encoder and decoder with enhanced architecture
        self.encoder = Encoder(in_channels, latent_dim)
        self.decoder = Decoder(latent_dim, in_channels)

        # Improved classifier for latent space with more regularization
        self.classifier = nn.Sequential(
            spectral_norm(nn.Linear(latent_dim, 256)),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(0.3),

            spectral_norm(nn.Linear(256, 128)),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(0.2),

            nn.Linear(128, num_classes)
        )

        # Center loss with updated feature dimension
        self.center_loss = CenterLoss(num_classes=num_classes, feat_dim=latent_dim, device=device)

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from N(0,1).
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        """
        Encode input and sample from the latent distribution.
        Returns the sampled latent vector z.
        """
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return z

    def encode_with_params(self, x):
        """
        Encode input and return distribution parameters.
        This is useful for computing the KL divergence loss.
        """
        return self.encoder(x)

    def decode(self, z):
        """
        Decode latent vector to reconstruction.
        """
        return self.decoder(z)

    def classify(self, z):
        """
        Classify latent vector.
        """
        return self.classifier(z)

    def compute_center_loss(self, z, labels):
        """
        Compute center loss for the latent vectors.
        """
        return self.center_loss(z, labels)

    def kl_divergence(self, mu, logvar):
        """
        Compute KL divergence between N(mu, var) and N(0, 1).
        With improved numerical stability
        """
        # Clamp values for stability
        logvar = torch.clamp(logvar, min=-20.0, max=20.0)

        # Compute KL divergence term by term for better stability
        var = torch.exp(logvar)
        kl_per_element = mu.pow(2) + var - 1.0 - logvar

        # Sum over dimensions, mean over batch
        return 0.5 * torch.sum(kl_per_element, dim=1).mean()

    def forward(self, x):
        """
        Complete forward pass.
        """
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar, z


# Improved Time embedding for diffusion model
class TimeEmbedding(nn.Module):
    def __init__(self, n_channels=256):
        super().__init__()
        self.n_channels = n_channels

        # More complex time embedding network
        self.lin1 = spectral_norm(nn.Linear(self.n_channels, self.n_channels * 2))
        self.norm1 = nn.LayerNorm(self.n_channels * 2)
        self.act = nn.SiLU()
        self.drop1 = nn.Dropout(0.1)

        self.lin2 = spectral_norm(nn.Linear(self.n_channels * 2, self.n_channels))
        self.norm2 = nn.LayerNorm(self.n_channels)
        self.drop2 = nn.Dropout(0.1)

    def forward(self, t):
        # Improved sinusoidal time embedding
        half_dim = self.n_channels // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=1)

        # Make sure emb has the right dimension
        if emb.shape[1] < self.n_channels:
            padding = torch.zeros(emb.shape[0], self.n_channels - emb.shape[1], device=emb.device)
            emb = torch.cat([emb, padding], dim=1)

        # Process through enhanced MLP
        emb = self.lin1(emb)
        emb = self.norm1(emb)
        emb = self.act(emb)
        emb = self.drop1(emb)

        emb = self.lin2(emb)
        emb = self.norm2(emb)
        emb = self.act(emb)
        emb = self.drop2(emb)

        return emb


# Enhanced Class Embedding
class ClassEmbedding(nn.Module):
    def __init__(self, num_classes=2, n_channels=256):
        super().__init__()
        # Fixed embedding dimension for stability
        self.emb_dim = n_channels

        # Improved embedding with proper initialization
        self.embedding = nn.Embedding(num_classes, self.emb_dim)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

        # More expressive transformation
        self.lin1 = spectral_norm(nn.Linear(self.emb_dim, self.emb_dim * 2))
        self.norm1 = nn.LayerNorm(self.emb_dim * 2)
        self.act = nn.SiLU()
        self.drop1 = nn.Dropout(0.1)

        self.lin2 = spectral_norm(nn.Linear(self.emb_dim * 2, n_channels))
        self.norm2 = nn.LayerNorm(n_channels)
        self.drop2 = nn.Dropout(0.1)

    def forward(self, c):
        # Get class embeddings
        emb = self.embedding(c)

        # Process through enhanced MLP
        emb = self.lin1(emb)
        emb = self.norm1(emb)
        emb = self.act(emb)
        emb = self.drop1(emb)

        emb = self.lin2(emb)
        emb = self.norm2(emb)
        emb = self.act(emb)
        emb = self.drop2(emb)

        return emb


# Improved Attention block with multi-head attention
class UNetAttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads

        assert channels % num_heads == 0, "channels must be divisible by num_heads"

        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5

        self.norm = nn.GroupNorm(1, channels)
        self.qkv = spectral_norm(nn.Conv2d(channels, channels * 3, 1))
        self.proj = spectral_norm(nn.Conv2d(channels, channels, 1))

        # Add output normalization and dropout for stability
        self.out_norm = nn.GroupNorm(1, channels)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        b, c, h, w = x.shape
        residual = x

        # Normalize input
        x = self.norm(x)

        # QKV projection
        qkv = self.qkv(x)
        qkv = qkv.reshape(b, 3, self.num_heads, c // self.num_heads, h * w)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        # Reshape for attention computation
        q = q.permute(0, 1, 3, 2)  # [b, heads, h*w, c//heads]
        k = k.permute(0, 1, 2, 3)  # [b, heads, c//heads, h*w]
        v = v.permute(0, 1, 3, 2)  # [b, heads, h*w, c//heads]

        # Compute attention with better numerical stability
        attn = torch.matmul(q, k) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = F.dropout(attn, p=0.1, training=self.training)

        # Apply attention
        out = torch.matmul(attn, v)  # [b, heads, h*w, c//heads]
        out = out.permute(0, 3, 1, 2)  # [b, c//heads, heads, h*w]
        out = out.reshape(b, c, h, w)

        # Project, normalize, add dropout, and add residual
        out = self.proj(out)
        out = self.out_norm(out)
        out = self.dropout(out)

        return out + residual


# Enhanced Residual block for UNet with class conditioning
class UNetResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, d_time=256, num_groups=8, dropout_rate=0.2):
        super().__init__()

        # Feature normalization and convolution
        self.norm1 = nn.GroupNorm(min(num_groups, in_channels), in_channels)
        self.conv1 = spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))

        # Time and class embedding projections
        self.time_emb = nn.Sequential(
            spectral_norm(nn.Linear(d_time, out_channels)),
            nn.SiLU()
        )

        self.class_emb = nn.Sequential(
            spectral_norm(nn.Linear(d_time, out_channels)),
            nn.SiLU()
        )

        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout_rate)

        # Second convolution
        self.norm2 = nn.GroupNorm(min(num_groups, out_channels), out_channels)
        self.conv2 = spectral_norm(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))

        # Gating mechanism for improved gradient flow
        self.gate = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

        # Residual connection handling
        if in_channels != out_channels:
            self.residual = spectral_norm(nn.Conv2d(in_channels, out_channels, 1))
        else:
            self.residual = nn.Identity()

    def forward(self, x, t, c=None):
        # First part
        h = self.act(self.norm1(x))
        h = self.conv1(h)

        # Add time embedding
        t_emb = self.time_emb(t)
        h = h + t_emb.view(-1, t_emb.shape[1], 1, 1)

        # Add class embedding if provided
        if c is not None:
            c_emb = self.class_emb(c)
            h = h + c_emb.view(-1, c_emb.shape[1], 1, 1)

        # Second part
        h = self.act(self.norm2(h))
        h = self.dropout(h)
        h = self.conv2(h)

        # Apply gating
        g = self.gate(h)
        h = h * g

        # Residual connection
        return h + self.residual(x)


# Modified Sequential for handling time and class embeddings
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


# Enhanced Class-Conditional UNet with improved architecture
class ConditionalUNet(nn.Module):
    def __init__(self, latent_dim=256, hidden_dims=[256, 512, 1024, 512, 256],
                 time_emb_dim=256, num_classes=2, dropout_rate=0.3, device='mps'):
        super().__init__()
        self.latent_dim = latent_dim
        self.device = device

        # Use consistent embedding dimension for time and class
        self.time_emb_dim = time_emb_dim

        # Time embedding with fixed dimension
        self.time_emb = TimeEmbedding(n_channels=time_emb_dim)

        # Class embedding with same dimension
        self.class_emb = ClassEmbedding(num_classes=num_classes, n_channels=time_emb_dim)

        # Initial normalization for better gradient flow
        self.initial_norm = nn.LayerNorm(latent_dim)

        # Initial projection from latent space to first hidden layer
        self.latent_proj = spectral_norm(nn.Linear(latent_dim, hidden_dims[0]))

        # Improved attention mechanism in hidden layers
        self.layers = nn.ModuleList()
        self.attentions = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        # Build hidden layers
        for i in range(len(hidden_dims) - 1):
            # Layer norm and dropout for current dimension
            self.norms.append(nn.LayerNorm(hidden_dims[i]))
            self.dropouts.append(nn.Dropout(dropout_rate))

            # Attention for feature mixing at each level
            self.attentions.append(nn.MultiheadAttention(
                embed_dim=hidden_dims[i],
                num_heads=8,
                dropout=0.1,
                batch_first=True
            ))

            # Time/class conditioned MLP layer
            nn.ModuleList([
                # First MLP layer
                nn.Sequential(
                    spectral_norm(nn.Linear(hidden_dims[i], hidden_dims[i] * 4)),
                    nn.LayerNorm(hidden_dims[i] * 4),
                    nn.SiLU(),
                    nn.Dropout(dropout_rate),
                    spectral_norm(nn.Linear(hidden_dims[i] * 4, hidden_dims[i]))
                ),
                # Projection to next dimension
                nn.Sequential(
                    spectral_norm(nn.Linear(hidden_dims[i], hidden_dims[i + 1])),
                    nn.LayerNorm(hidden_dims[i + 1])
                )
            ])

        # Final layer norm and dropout
        self.final_norm = nn.LayerNorm(hidden_dims[-1])
        self.final_dropout = nn.Dropout(dropout_rate)

        # Final layer projecting back to latent dimension with skip connection
        self.final = nn.Sequential(
            spectral_norm(nn.Linear(hidden_dims[-1], hidden_dims[-1] * 2)),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dims[-1] * 2, latent_dim)
        )

    def forward(self, x, t, c=None):
        original_x = x

        # Initial normalization
        x = self.initial_norm(x)

        # Time embedding
        t_emb = self.time_emb(t)

        # Class embedding (if provided)
        c_emb = None
        if c is not None:
            c_emb = self.class_emb(c)

        # Initial projection
        h = self.latent_proj(x)

        # Process through layers with conditioning
        for i in range(len(self.layers)):
            # Normalize and apply attention
            h_norm = self.norms[i](h)
            h_attn, _ = self.attentions[i](h_norm, h_norm, h_norm)
            h = h + h_attn  # Residual connection
            h = self.dropouts[i](h)

            # Current MLP layer
            residual_mlp, project = self.layers[i]

            # Create conditional inputs by concatenating embeddings
            h_cond = h

            # Add time conditioning (add as residual)
            if t_emb is not None:
                h_cond = h_cond + t_emb

            # Add class conditioning if provided
            if c_emb is not None:
                h_cond = h_cond + c_emb

            # Apply MLP with residual connection
            h = h + residual_mlp(h_cond)

            # Apply projection to next dimension
            h = project(h)

        # Final normalization and MLP
        h = self.final_norm(h)
        h = self.final_dropout(h)

        # Apply final projection with skip connection
        final_output = self.final(h)

        # Residual connection to input
        return final_output + original_x

# Improved beta schedule for diffusion process
def cosine_beta_schedule(timesteps, s=0.008):
    """Cosine schedule as proposed in https://arxiv.org/abs/2102.09672"""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

# Linear schedule for compatibility with older checkpoints
def create_linear_schedule(n_steps=1000, device=None):
    """Create the original linear beta schedule for loading checkpoints"""
    beta = torch.linspace(0.0001, 0.02, n_steps).to(device)
    return beta

# Blend two schedules for smooth transitions
def blend_schedules(schedule1, schedule2, ratio=0.5):
    """Blend two beta schedules with the given ratio"""
    return schedule1 * (1 - ratio) + schedule2 * ratio

# Enhanced Class-conditional diffusion model
class ConditionalDenoiseDiffusion:
    def __init__(self, eps_model, n_steps=1000, device=None, use_linear=False):
        self.eps_model = eps_model
        self.device = device
        self.n_steps = n_steps

        # Choose schedule based on parameter
        if use_linear:
            # Use original linear schedule when loading from checkpoint
            self.beta = create_linear_schedule(n_steps, device)
        else:
            # Use cosine schedule for new models (generally better)
            self.beta = cosine_beta_schedule(n_steps).to(device)

        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

        # Precompute values for efficiency
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bar)

    def q_sample(self, x0, t, eps=None):
        """Forward diffusion process: add noise to data"""
        if eps is None:
            eps = torch.randn_like(x0)

        # Extract alpha_bar values for the specific timesteps
        sqrt_alpha_bar_t = self.sqrt_alpha_bar[t].reshape(-1, 1)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bar[t].reshape(-1, 1)

        # Apply diffusion formula more efficiently
        return sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * eps

    def p_sample(self, xt, t, c=None, guidance_scale=3.0):
        """Single denoising step with classifier-free guidance"""
        # Convert time to tensor format expected by model
        if not isinstance(t, torch.Tensor):
            t = torch.tensor([t], device=xt.device)

        # Classifier-free guidance implementation
        if c is not None and guidance_scale > 1.0:
            # Get unconditional prediction (no class information)
            eps_uncond = self.eps_model(xt, t, None)
            # Get conditional prediction (with class information)
            eps_cond = self.eps_model(xt, t, c)
            # Combine with guidance scale
            eps_theta = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
        else:
            # Standard prediction (original code path)
            eps_theta = self.eps_model(xt, t, c)

        # Get alpha values for the current timestep
        alpha_t = self.alpha[t].reshape(-1, 1)
        alpha_bar_t = self.alpha_bar[t].reshape(-1, 1)

        # Calculate mean with numerical stability improvements
        eps_coef = (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t + 1e-8)
        mean = (xt - eps_coef * eps_theta) / torch.sqrt(alpha_t + 1e-8)

        # Add noise if not the final step
        var = self.beta[t].reshape(-1, 1)

        if t[0] > 0:
            noise = torch.randn_like(xt)
            return mean + torch.sqrt(var) * noise
        else:
            return mean

    def sample(self, shape, device, c=None, guidance_scale=3.0, callback=None):
        """Generate samples with classifier-free guidance and optional progress callback"""
        # Start from pure noise
        x = torch.randn(shape, device=device)

        # Progressively denoise with guidance
        for t in reversed(range(self.n_steps)):
            # Create batch of same timestep
            timestep = torch.full((shape[0],), t, device=device, dtype=torch.long)

            # Denoise one step
            x = self.p_sample(x, timestep, c, guidance_scale=guidance_scale)

            # Call the callback if provided (for visualization/monitoring)
            if callback is not None and t % 100 == 0:
                callback(x, t)

        return x

    def loss(self, x0, labels=None, timestep_weights=None):
        """
        Calculate noise prediction loss with optional class conditioning and timestep weighting

        Args:
            x0: Initial clean data
            labels: Optional class labels for conditioning
            timestep_weights: Optional weights for different timesteps
        """
        batch_size = x0.shape[0]

        # Random timestep for each sample
        t = torch.randint(0, self.n_steps, (batch_size,), device=x0.device, dtype=torch.long)

        # Add noise
        eps = torch.randn_like(x0)
        xt = self.q_sample(x0, t, eps)

        # Predict noise (with class conditioning if labels provided)
        eps_theta = self.eps_model(xt, t, labels)

        # Compute MSE loss
        loss = (eps - eps_theta).square().mean(dim=list(range(1, len(eps.shape))))

        # Apply timestep weights if provided
        if timestep_weights is not None:
            loss = loss * timestep_weights[t]

        return loss.mean()

# Function to generate a grid of samples for all classes
def generate_samples_grid(autoencoder, diffusion, n_per_class=5, save_dir="./results", guidance_scale=3.0):
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
    fig.suptitle(f'CIFAR Cat and Dog Samples Generated by VAE-Diffusion Model (Guidance: {guidance_scale})',
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

        # Sample from the diffusion model with class conditioning and guidance
        samples = diffusion.sample(latent_shape, device, class_tensor, guidance_scale=guidance_scale)

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
        "Each row corresponds to a different animal category as labeled.\n"
        f"Classifier-free guidance scale: {guidance_scale} (higher values = stronger class conditioning)"
    )
    plt.figtext(0.5, 0.01, description, ha='center', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7))

    plt.tight_layout(rect=[0, 0.03, 1, 0.92])  # Adjust layout to make room for titles
    plt.savefig(f"{save_dir}/vae_samples_grid_all_classes_guidance_{guidance_scale:.1f}.png", dpi=150,
                bbox_inches='tight')
    plt.close()

    # Set models back to training mode
    autoencoder.train()
    diffusion.eps_model.train()

    print(f"Generated sample grid for all classes with clearly labeled categories (guidance: {guidance_scale})")
    return f"{save_dir}/vae_samples_grid_all_classes_guidance_{guidance_scale:.1f}.png"

# Visualize latent space denoising process for a specific class
def visualize_denoising_steps(autoencoder, diffusion, class_idx, save_path=None, guidance_scale=3.0):
    """
    Visualize both the denoising process and the corresponding path in latent space for VAE.

    Args:
        autoencoder: Trained VAE model
        diffusion: Trained diffusion model
        class_idx: Target class index (0-1 for cat/dog)
        save_path: Path to save the visualization
        guidance_scale: Scale for classifier-free guidance (higher = stronger conditioning)
    """
    device = next(autoencoder.parameters()).device

    # Set models to evaluation mode
    autoencoder.eval()
    diffusion.eps_model.eval()

    # ===== PART 1: Setup dimensionality reduction for latent space =====
    print(f"Generating latent space projection for class {class_names[class_idx]}...")

    # Load CIFAR-10 test data
    cifar_test = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)
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
            mu, _ = autoencoder.encoder(images)
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

            # Denoise from current step to t=0 with class conditioning and guidance
            for time_step in range(t, -1, -1):
                current_x = diffusion.p_sample(current_x, torch.tensor([time_step], device=device),
                                               class_tensor, guidance_scale=guidance_scale)

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
    ax_denoising.set_title(
        f"VAE-Diffusion Model Denoising Process for {class_names[class_idx]} (Guidance: {guidance_scale})",
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
    ax_latent.scatter(path_2d[0, 0], path_2d[0, 1], c='black', s=100, marker='x', label="Start (Noise)",
                      zorder=11)
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

    ax_latent.set_title(
        f"VAE-Diffusion Path in Latent Space for {class_names[class_idx]} (Guidance: {guidance_scale})",
        fontsize=16)
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
    cifar_test = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)
    test_dataset = create_cat_dog_dataset(cifar_test)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True)

    test_images, test_labels = next(iter(test_loader))
    test_images = test_images.to(device)

    # Generate reconstructions with VAE
    autoencoder.eval()
    with torch.no_grad():
        # Get reconstruction with VAE forward pass
        reconstructed, mu, logvar, _ = autoencoder(test_images)

    # Create visualization
    fig, axes = plt.subplots(2, 8, figsize=(16, 4))

    for i in range(8):
        # Original
        img = test_images[i].cpu().permute(1, 2, 0).numpy()
        axes[0, i].imshow(img)
        axes[0, i].set_title(f'Original: {class_names[test_labels[i]]}')
        axes[0, i].axis('off')

        # Reconstruction
        recon_img = reconstructed[i].cpu().permute(1, 2, 0).numpy()
        axes[1, i].imshow(recon_img)
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
    cifar_test = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)
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
            mu, _ = autoencoder.encode_with_params(images)
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

# Function to generate samples of a specific class
def generate_class_samples(autoencoder, diffusion, target_class, num_samples=5, save_path=None,
                           guidance_scale=3.0):
    """
    Generate samples of a specific target class using the VAE decoder with guidance

    Args:
        autoencoder: Trained VAE model
        diffusion: Trained conditional diffusion model
        target_class: Index of the target class (0-1) or class name
        num_samples: Number of samples to generate
        save_path: Path to save the generated samples
        guidance_scale: Scale for classifier-free guidance (higher = stronger conditioning)

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
        # Sample from the diffusion model with class conditioning and guidance
        latent_samples = diffusion.sample(latent_shape, device, class_tensor, guidance_scale=guidance_scale)

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

        plt.suptitle(f"Generated {class_names[target_class]} Samples (Guidance: {guidance_scale})")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    return samples

# Create diffusion animation for visualization
def create_diffusion_animation(autoencoder, diffusion, class_idx, num_frames=50, seed=42,
                               save_path=None, temp_dir=None, fps=10, reverse=False, guidance_scale=3.0):
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
        guidance_scale: Scale for classifier-free guidance (higher = stronger conditioning)

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

    print(
        f"Creating diffusion animation for class '{class_names[class_idx]}' with guidance scale {guidance_scale}...")
    frame_paths = []

    with torch.no_grad():
        # First, generate a proper clean image at t=0 by denoising from pure noise
        print("Generating initial clean image...")
        # Start from pure noise - now using flat latent dimension
        x = torch.randn((1, autoencoder.latent_dim), device=device)

        # Denoise completely to get clean image at t=0 with guidance
        for time_step in tqdm(range(total_steps - 1, -1, -1), desc="Denoising"):
            x = diffusion.p_sample(x, torch.tensor([time_step], device=device),
                                   class_tensor, guidance_scale=guidance_scale)

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
            title = f'Class: {class_names[class_idx]} (t={t}, {progress:.1f}% noise, guidance={guidance_scale})'
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

# Updated function to compute VAE loss (reconstruction + KL divergence)
def vae_loss(x_recon, x, mu, logvar, kl_weight=0.001, max_kl=50.0, reduction='mean'):
    # Reconstruction loss (using Euclidean distance)
    squared_diff = (x_recon - x) ** 2
    squared_dist = squared_diff.view(x.size(0), -1).sum(dim=1)
    recon_loss = torch.sqrt(squared_dist + 1e-8)  # Add small epsilon to avoid numerical instability

    # KL divergence
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)

    kl_div = torch.clamp(kl_div, max=max_kl)

    # Total loss
    if reduction == 'mean':
        total_loss = recon_loss.mean() + kl_weight * kl_div.mean()
        recon_loss = recon_loss.mean()
        kl_div = kl_div.mean()
    elif reduction == 'sum':
        total_loss = recon_loss.sum() + kl_weight * kl_div.sum()
        recon_loss = recon_loss.sum()
        kl_div = kl_div.sum()
    else:  # 'none'
        total_loss = recon_loss + kl_weight * kl_div

    return total_loss, recon_loss, kl_div

# Main function
# Main function
def main(checkpoint_path=None, total_epochs=10000):
    """Main function to run the entire pipeline non-interactively"""
    print("Starting class-conditional diffusion model for CIFAR cats and dogs")

    # Set device
    device = torch.device("mps" if torch.backends.mps.is_available() else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    # Create results directory
    results_dir = "./cifar_cat_dog_conditional_v9"
    os.makedirs(results_dir, exist_ok=True)

    # Load CIFAR-10 dataset and filter for cats and dogs
    print("Loading and filtering CIFAR-10 dataset for cats and dogs...")
    cifar_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    train_dataset = create_cat_dog_dataset(cifar_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2,
                              pin_memory=True)

    # Paths for saved models
    autoencoder_path = f"{results_dir}/cifar_cat_dog_autoencoder.pt"
    diffusion_path = f"{results_dir}/conditional_diffusion_final.pt"

    # Create autoencoder
    autoencoder = SimpleAutoencoder(in_channels=3, latent_dim=256).to(
        device)  # 3 channels for RGB

    # Define the training function OUTSIDE the conditional block
    def train_autoencoder(autoencoder, train_loader, num_epochs=100, lr=1e-4, lambda_cls=5.0,
                          lambda_center=2.0, kl_weight=0.03, visualize_every=100, save_dir="./results"):
        """Train VAE with separate optimization paths"""
        print("Starting VAE training with class separation...")
        os.makedirs(save_dir, exist_ok=True)
        device = next(autoencoder.parameters()).device

        # Create optimizers - don't share parameters between optimizers
        recon_optimizer = optim.AdamW(
            list(autoencoder.encoder.parameters()) +
            list(autoencoder.decoder.parameters()),
            lr=lr,
            weight_decay=1e-5  # Add weight decay for regularization
        )

        class_optimizer = optim.AdamW(
            list(autoencoder.classifier.parameters()) +
            list(autoencoder.center_loss.parameters()),
            lr=lr * 5,  # Keep the higher learning rate for classifier
            weight_decay=1e-5  # Same weight decay
        )

        # Create learning rate schedulers with warmup
        total_steps = len(train_loader) * num_epochs

        # OneCycleLR provides both warmup and annealing
        recon_scheduler = OneCycleLR(
            recon_optimizer,
            max_lr=lr,
            total_steps=total_steps,
            pct_start=0.05,  # 5% of training for warmup
            anneal_strategy='cos',  # Cosine annealing after warmup
            div_factor=25.0,  # Initial lr = max_lr/25
            final_div_factor=10000.0  # Final lr = initial_lr/10000
        )

        class_scheduler = OneCycleLR(
            class_optimizer,
            max_lr=lr * 5,  # Keep the 5x multiplier for classifier
            total_steps=total_steps,
            pct_start=0.05,
            anneal_strategy='cos',
            div_factor=25.0,
            final_div_factor=10000.0
        )

        loss_history = {'total': [], 'recon': [], 'kl': [], 'class': [], 'center': []}

        cycle_length = num_epochs // 4
        R = 0.7

        for epoch in range(num_epochs):
            autoencoder.train()
            epoch_recon_loss = 0
            epoch_kl_loss = 0
            epoch_class_loss = 0
            epoch_center_loss = 0
            epoch_total_loss = 0

            if epoch % cycle_length == 0 and epoch > 0:
                print(f"Epoch {epoch + 1}: Starting new annealing cycle, resetting optimizer momentum...")
                for param_group in recon_optimizer.param_groups:
                    for p in param_group['params']:
                        if p in recon_optimizer.state:
                            param_state = recon_optimizer.state[p]
                            if 'momentum_buffer' in param_state:
                                param_state['momentum_buffer'].zero_()
                            if 'exp_avg' in param_state:  # Adam优化器
                                param_state['exp_avg'].zero_()
                                param_state['exp_avg_sq'].zero_()

                for param_group in class_optimizer.param_groups:
                    for p in param_group['params']:
                        if p in class_optimizer.state:
                            param_state = class_optimizer.state[p]
                            if 'momentum_buffer' in param_state:
                                param_state['momentum_buffer'].zero_()
                            if 'exp_avg' in param_state:  # Adam优化器
                                param_state['exp_avg'].zero_()
                                param_state['exp_avg_sq'].zero_()

            cycle_position = (epoch % cycle_length) / cycle_length

            if cycle_position < R:
                current_kl_weight = kl_weight * (cycle_position / R)
            else:
                current_kl_weight = kl_weight

            print(f"Epoch {epoch + 1}: Using KL weight = {current_kl_weight:.6f}")

            for batch_idx, (data, labels) in enumerate(
                    tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")):
                data = data.to(device)
                labels = labels.to(device)

                # Step 1: Reconstruction optimization (VAE path)
                recon_optimizer.zero_grad()

                # Get mu and logvar from encoder
                mu, logvar = autoencoder.encode_with_params(data)

                # Sample latent vector using reparameterization trick
                z = autoencoder.reparameterize(mu, logvar)

                # Decode to get reconstruction
                reconstructed = autoencoder.decode(z)

                # Compute VAE loss (reconstruction + KL divergence)
                total_loss, recon_loss, kl_loss = vae_loss(
                    reconstructed, data, mu, logvar, kl_weight=current_kl_weight
                )

                # Backward pass for reconstruction path
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(autoencoder.encoder.parameters()) + list(autoencoder.decoder.parameters()),
                    max_norm=0.1
                )
                recon_optimizer.step()
                recon_scheduler.step()  # Step the scheduler after each batch

                # Step 2: Classification optimization
                class_optimizer.zero_grad()

                # Detach encoder output to prevent gradients flowing back into encoder
                with torch.no_grad():
                    mu, logvar = autoencoder.encode_with_params(data)
                    z = autoencoder.reparameterize(mu, logvar)

                # Forward through classification branch only
                class_logits = autoencoder.classify(z)
                class_loss = F.cross_entropy(class_logits, labels)
                center_loss = autoencoder.compute_center_loss(z, labels)

                # Combined classification loss
                total_class_loss = lambda_cls * class_loss + lambda_center * center_loss
                total_class_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(autoencoder.classifier.parameters()) + list(autoencoder.center_loss.parameters()),
                    max_norm=0.5
                )

                class_optimizer.step()
                class_scheduler.step()  # Step the scheduler after each batch

                # Record losses
                epoch_recon_loss += recon_loss.item()
                epoch_kl_loss += kl_loss.item()
                epoch_class_loss += class_loss.item()
                epoch_center_loss += center_loss.item()
                epoch_total_loss += total_loss.item()

            # Calculate average losses
            num_batches = len(train_loader)
            avg_recon_loss = epoch_recon_loss / num_batches
            avg_kl_loss = epoch_kl_loss / num_batches
            avg_class_loss = epoch_class_loss / num_batches
            avg_center_loss = epoch_center_loss / num_batches
            avg_total_loss = epoch_total_loss / num_batches

            # Store losses for plotting
            loss_history['recon'].append(avg_recon_loss)
            loss_history['kl'].append(avg_kl_loss)
            loss_history['class'].append(avg_class_loss)
            loss_history['center'].append(avg_center_loss)
            loss_history['total'].append(avg_total_loss)

            # Print losses
            print(f"Epoch {epoch + 1}/{num_epochs}, "
                  f"Total Loss: {avg_total_loss:.6f}, "
                  f"Recon Loss: {avg_recon_loss:.6f}, "
                  f"KL Loss: {avg_kl_loss:.6f}, "
                  f"Class Loss: {avg_class_loss:.6f}, "
                  f"Center Loss: {avg_center_loss:.6f}")

            # Visualizations
            if (epoch + 1) % visualize_every == 0 or epoch == num_epochs - 1:
                visualize_reconstructions(autoencoder, epoch + 1, save_dir)
                visualize_latent_space(autoencoder, epoch + 1, save_dir)

                # Save checkpoint
                torch.save(autoencoder.state_dict(), f"{save_dir}/vae_epoch_{epoch + 1}.pt")

        return autoencoder, loss_history

    # Check if trained autoencoder exists
    if os.path.exists(autoencoder_path):
        print(f"Loading existing autoencoder from {autoencoder_path}")
        autoencoder.load_state_dict(torch.load(autoencoder_path, map_location=device))
        autoencoder.eval()
        # Create dummy loss history for consistency
        ae_losses = {'total': [], 'recon': [], 'kl': [], 'class': [], 'center': []}
    else:
        print("No existing autoencoder found. Training a new one...")
        # Train autoencoder
        autoencoder, ae_losses = train_autoencoder(
            autoencoder,
            train_loader,
            num_epochs=1000,
            lr=1e-4,
            lambda_cls=5.0,
            lambda_center=2.3,
            kl_weight=0.035,
            visualize_every=100,
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

    # Create conditional UNet with increased dimensions
    conditional_unet = ConditionalUNet(
        latent_dim=256,
        hidden_dims=[256, 512, 1024, 512, 256],  # Doubled each dimension
        time_emb_dim=256,
        num_classes=len(class_names)
    ).to(device)

    # Initialize weights for UNet if needed
    def init_weights(m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, a=0.2)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    # Extract epoch number from checkpoint path if provided
    start_epoch = 0
    # First check the provided checkpoint path, if any
    if checkpoint_path and os.path.exists(checkpoint_path):
        # Parse epoch number from filename (assumes format "conditional_diffusion_epoch_X.pt")
        try:
            filename = os.path.basename(checkpoint_path)
            # Extract number between "epoch_" and ".pt"
            epoch_str = filename.split("epoch_")[1].split(".pt")[0]
            start_epoch = int(epoch_str)
            print(f"Continuing training from epoch {start_epoch}")

            # Load the model
            conditional_unet.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)
            diffusion = ConditionalDenoiseDiffusion(conditional_unet, n_steps=1000, device=device)
        except (IndexError, ValueError) as e:
            print(f"Could not extract epoch number from checkpoint filename: {e}")
            print("Starting from epoch 0")
            start_epoch = 0
    # Then check if standard diffusion path exists
    elif os.path.exists(diffusion_path):
        print(f"Loading existing diffusion model from {diffusion_path}")
        conditional_unet.load_state_dict(torch.load(diffusion_path, map_location=device))
        diffusion = ConditionalDenoiseDiffusion(conditional_unet, n_steps=1000, device=device)
    # If no checkpoint found, we'll start from scratch
    else:
        print("No existing diffusion model found. Training a new one...")
        conditional_unet.apply(init_weights)
        # diffusion variable will be initialized during training

    # Define train_conditional_diffusion function
    def train_conditional_diffusion(autoencoder, unet, num_epochs=100, lr=1e-3, visualize_every=10,
                                   save_dir="./results", start_epoch=0, transition_epochs=[10000, 10001]):
        """
        Train the conditional diffusion model with schedule transitions.

        Args:
            autoencoder: Trained VAE model
            unet: UNet denoising model
            num_epochs: Number of epochs to train
            lr: Learning rate
            visualize_every: Epochs between visualizations
            save_dir: Directory to save results
            start_epoch: Starting epoch number (if continuing from checkpoint)
            transition_epochs: When to transition to cosine schedule [blend_point, full_cosine_point]
        """
        print(f"Starting Class-Conditional Diffusion Model training with VAE from epoch {start_epoch + 1}...")
        os.makedirs(save_dir, exist_ok=True)

        autoencoder.eval()  # Set VAE to evaluation mode

        # Create diffusion model with initial linear schedule for continuing from checkpoint
        diffusion = ConditionalDenoiseDiffusion(unet, n_steps=1000, device=device, use_linear=True)

        # Create optimizer and warm restart scheduler
        optimizer = torch.optim.AdamW(unet.parameters(), lr=lr, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=20,  # Restart every 20 epochs
            T_mult=2,  # Double period after each restart
            eta_min=1e-6
        )

        # Step scheduler to match start_epoch
        if start_epoch > 0:
            for _ in range(start_epoch):
                scheduler.step()

        # Create cosine schedule for later transition
        cosine_beta = cosine_beta_schedule(diffusion.n_steps).to(device)

        # Create timestep weights to focus more on important timesteps
        timestep_weights = torch.ones(diffusion.n_steps, device=device)
        mid_point = diffusion.n_steps // 2
        # Add more weight to middle timesteps (bell curve)
        for t in range(diffusion.n_steps):
            # Create a bell curve centered at mid_point
            timestep_weights[t] = 1.0 + 0.5 * torch.exp(torch.tensor(-0.5 * ((t - mid_point) / (diffusion.n_steps / 6)) ** 2, device=device))

        # Training loop
        loss_history = []

        for epoch in range(start_epoch, start_epoch + num_epochs):
            epoch_loss = 0

            # Schedule transition logic
            current_epoch_in_continuation = epoch - start_epoch

            # First transition - blend linear and cosine schedules
            if transition_epochs[0] is not None and current_epoch_in_continuation == transition_epochs[0]:
                print(f"Epoch {epoch + 1}: Transitioning to blended beta schedule (50% cosine)...")
                linear_beta = diffusion.beta.clone()
                diffusion.beta = blend_schedules(linear_beta, cosine_beta, ratio=0.5)
                diffusion.alpha = 1 - diffusion.beta
                diffusion.alpha_bar = torch.cumprod(diffusion.alpha, dim=0)
                # Update precomputed values
                diffusion.sqrt_alpha_bar = torch.sqrt(diffusion.alpha_bar)
                diffusion.sqrt_one_minus_alpha_bar = torch.sqrt(1 - diffusion.alpha_bar)

            # Second transition - full cosine schedule
            elif transition_epochs[1] is not None and current_epoch_in_continuation == transition_epochs[1]:
                print(f"Epoch {epoch + 1}: Completing transition to full cosine beta schedule...")
                diffusion.beta = cosine_beta.clone()
                diffusion.alpha = 1 - diffusion.beta
                diffusion.alpha_bar = torch.cumprod(diffusion.alpha, dim=0)
                # Update precomputed values
                diffusion.sqrt_alpha_bar = torch.sqrt(diffusion.alpha_bar)
                diffusion.sqrt_one_minus_alpha_bar = torch.sqrt(1 - diffusion.alpha_bar)

            # Normal training loop
            for batch_idx, (data, labels) in enumerate(
                    tqdm(train_loader, desc=f"Epoch {epoch + 1}/{start_epoch + num_epochs}")):
                data = data.to(device)
                labels = labels.to(device)

                # Encode images to latent space using VAE
                with torch.no_grad():
                    latents = autoencoder.encode(data)

                # Calculate diffusion loss with class conditioning and timestep weighting
                loss = diffusion.loss(latents, labels, timestep_weights)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()

            # Calculate average loss
            avg_loss = epoch_loss / len(train_loader)
            loss_history.append(avg_loss)
            print(f"Epoch {epoch + 1}/{start_epoch + num_epochs}, Average Loss: {avg_loss:.6f}")

            # Learning rate scheduling
            scheduler.step()

            # Visualize samples periodically
            if (epoch + 1) % visualize_every == 0 or epoch == start_epoch + num_epochs - 1:
                # Generate samples for both classes
                for class_idx in range(len(class_names)):
                    create_diffusion_animation(autoencoder, diffusion, class_idx=class_idx, num_frames=50,
                                               save_path=f"{save_dir}/diffusion_animation_class_{class_names[class_idx]}_epoch_{epoch + 1}.gif",
                                               guidance_scale=3.0)

                    save_path = f"{save_dir}/sample_class_{class_names[class_idx]}_epoch_{epoch + 1}.png"
                    generate_class_samples(autoencoder, diffusion, target_class=class_idx, num_samples=5,
                                           save_path=save_path, guidance_scale=3.0)

                    save_path = f"{save_dir}/denoising_path_{class_names[class_idx]}_epoch_{epoch + 1}.png"
                    visualize_denoising_steps(autoencoder, diffusion, class_idx=class_idx, save_path=save_path,
                                              guidance_scale=3.0)

                # Save current schedule information
                if epoch + 1 == start_epoch + transition_epochs[0] or epoch + 1 == start_epoch + \
                        transition_epochs[1]:
                    schedule_type = "50_percent_cosine" if epoch + 1 == start_epoch + transition_epochs[
                        0] else "full_cosine"
                    torch.save({
                        'beta': diffusion.beta,
                        'alpha': diffusion.alpha,
                        'alpha_bar': diffusion.alpha_bar,
                        'schedule_type': schedule_type
                    }, f"{save_dir}/diffusion_schedule_{schedule_type}_epoch_{epoch + 1}.pt")

                # Save checkpoint
                torch.save(unet.state_dict(), f"{save_dir}/conditional_diffusion_epoch_{epoch + 1}.pt")

        return unet, diffusion, loss_history

    remaining_epochs = total_epochs - start_epoch

    # If no diffusion model is loaded, train a new one
    if 'diffusion' not in locals():
        # Train conditional diffusion model
        conditional_unet, diffusion, diff_losses = train_conditional_diffusion(
            autoencoder, conditional_unet,
            num_epochs=remaining_epochs,
            lr=1e-3,
            visualize_every=100,
            save_dir=results_dir,
            start_epoch=start_epoch
        )

        # Save diffusion model
        torch.save(conditional_unet.state_dict(), diffusion_path)

        # Plot diffusion loss
        plt.figure(figsize=(8, 5))
        plt.plot(range(start_epoch + 1, start_epoch + len(diff_losses) + 1),
                 diff_losses)  # Adjust x-axis for plot
        plt.title('Diffusion Model Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig(f"{results_dir}/diffusion_loss.png")
        plt.close()
    # If we loaded a model but want to continue training
    elif start_epoch > 0:
        # Continue training from the loaded checkpoint
        conditional_unet, diffusion, diff_losses = train_conditional_diffusion(
            autoencoder, conditional_unet, num_epochs=remaining_epochs, lr=1e-3,
            visualize_every=100,  # Visualize every 100 epochs
            save_dir=results_dir,
            start_epoch=start_epoch  # Pass the start epoch
        )

        # Save diffusion model
        torch.save(conditional_unet.state_dict(), diffusion_path)

        # Plot diffusion loss
        plt.figure(figsize=(8, 5))
        plt.plot(range(start_epoch + 1, start_epoch + len(diff_losses) + 1),
                 diff_losses)  # Adjust x-axis for plot
        plt.title('Diffusion Model Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig(f"{results_dir}/diffusion_loss_continued.png")
        plt.close()

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
    main(total_epochs=10000)
