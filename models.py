import torch
import torch.nn as nn
import torch.nn.functional as F

class cEncoder(nn.Module):
    def __init__(self, z_dim, num_classes):
        super(cEncoder, self).__init__()
        self.encoder_backbone = nn.Sequential(
            # 1x20x20 -> 16x10x10
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            # 16x10x10 -> 32x5x5
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            # 32x5x5 -> 64x3x3
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            # 64x3x3 -> 256x1x1
            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=0), nn.ReLU(),
            nn.Flatten()
        )
        # Combine image features with class condition -> mu and logvar
        self.encoder_head = nn.Linear(256+num_classes, 2*z_dim)
         
    def forward(self, x, cond):
        x = self.encoder_backbone(x)
        x = torch.cat([x, cond], dim=1)
        x = self.encoder_head(x)
        mu, logvar = torch.chunk(x, 2, dim=1)
        return mu, logvar

class cDecoder(nn.Module):
    def __init__(self, z_dim, num_classes):
        super(cDecoder, self).__init__()
        # Combine latent vector with class condition
        self.decoder_input = nn.Sequential(
            nn.Linear(z_dim + num_classes, 256), nn.ReLU()
        )
        self.decoder_backbone = nn.Sequential(
            nn.Unflatten(1, (256, 1, 1)),
            # 256x1x1 -> 64x3x3
            nn.ConvTranspose2d(256, 64, kernel_size=3, stride=1, padding=0), nn.ReLU(),
            # 64x3x3 -> 32x5x5
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=0), nn.ReLU(),
            # 32x5x5 -> 16x10x10
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            # 16x10x10 -> 1x20x20
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, z, cond):
        z = torch.cat([z, cond], dim=1)
        z = self.decoder_input(z)
        x_hat = self.decoder_backbone(z)
        return x_hat

# Define the conditional variational autoencoder
class cVAE(nn.Module):
    def __init__(self, z_dim, num_classes):
        super(cVAE, self).__init__()
        self.encoder = cEncoder(z_dim, num_classes)
        self.decoder = cDecoder(z_dim, num_classes)
        self.z_dim = z_dim
        self.num_classes = num_classes
    
    # Encode an image and class condition into latent distribution parameters
    def encode(self, x, cond):
        return self.encoder(x, cond)

    # Decode a latent representation with the class condition
    def decode(self, z, cond):
        return self.decoder(z, cond)
    
    # Sample from the latent distribution and reconstruct the input image
    def forward(self, x, cond):
        mu, logvar = self.encoder(x, cond)
        sigma = torch.exp(0.5 * logvar)
        eps = torch.randn_like(sigma)
        z = mu + eps * sigma
        x_recon = self.decoder(z, cond)
        return x_recon, mu, logvar

def compute_vae_loss(x, x_hat, mu, logvar, beta=0.1):
    """
    Compute the cVAE loss:
    reconstruction loss + beta-weighted KL divergence.
    """
    # Reconstruction loss: measures the difference between
    # the original image and the reconstructed image
    recon_loss = 100 * F.mse_loss(x_hat, x, reduction='mean')

    # KL divergence: regularises the latent distribution
    # When beta = 0, the model behaves like a standard autoencoder
    if beta == 0:
        kl_loss = 0.0
    else:
        kl_loss = torch.mean(
            -0.5 * torch.sum(
                1 + logvar - mu.pow(2) - logvar.exp(),
                dim=1
            )
        )
    # Total cVAE loss
    total_loss = recon_loss + beta * kl_loss

    return total_loss, recon_loss, kl_loss

# Define the conditional discriminator for real and reconstructed images
class cDiscriminator(nn.Module):
    def __init__(self, num_classes):
        super(cDiscriminator, self).__init__()
        self.num_classes = num_classes

        # Extract image features using convolutional layers
        self.discriminator_backbone = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0), nn.LeakyReLU(0.2),
            nn.Flatten()
        )

        # Combine image features with the class condition and output a discriminator logit
        self.discriminator_head = nn.Sequential(
            nn.Linear(128 + num_classes, 1)
        )

    def forward(self, x, cond):
        x = self.discriminator_backbone(x)
        x = torch.cat([x, cond], dim=1)
        return self.discriminator_head(x)

class SimpleClassifier(nn.Module):
    def __init__(self):
        super(SimpleClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 5 * 5, 128), nn.ReLU()
        )
        self.head = nn.Linear(128, 10)
        
    def forward(self, x):
        return self.head(self.features(x))
        
    def extract_features(self, x):
        return self.features(x)








