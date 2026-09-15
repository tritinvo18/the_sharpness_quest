import torch
import torch.nn as nn
import io
import os
import zipfile

class cEncoder(nn.Module):
    def __init__(self, z_dim, num_classes):
        super(cEncoder, self).__init__()
        self.encoder_backbone = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=0), nn.ReLU(),
            nn.Flatten()
        )
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
        self.decoder_input = nn.Sequential(
            nn.Linear(z_dim + num_classes, 256), nn.ReLU()
        )
        self.decoder_backbone = nn.Sequential(
            nn.Unflatten(1, (256, 1, 1)),
            nn.ConvTranspose2d(256, 64, kernel_size=3, stride=1, padding=0), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=0), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid() 
        )

    def forward(self, z, cond):
        z = torch.cat([z, cond], dim=1)
        z = self.decoder_input(z)
        x_hat = self.decoder_backbone(z)
        return x_hat

class cVAE(nn.Module):
    def __init__(self, z_dim, num_classes):
        super(cVAE, self).__init__()
        self.encoder = cEncoder(z_dim, num_classes)
        self.decoder = cDecoder(z_dim, num_classes)
    
    def decode(self, z, cond):
        return self.decoder(z, cond)
        
    def forward(self, x, cond):
        mu, logvar = self.encoder(x, cond)
        sigma = torch.exp(0.5 * logvar)
        eps = torch.randn_like(sigma)
        z = mu + eps * sigma
        x_recon = self.decoder(z, cond)
        return x_recon, mu, logvar

class cDiscriminator(nn.Module):
    def __init__(self, num_classes):
        super(cDiscriminator, self).__init__()
        self.features = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)), nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)), nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)), nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0)), nn.LeakyReLU(0.2),
            nn.Flatten()
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 + num_classes, 64), nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x, cond, return_features=False):
        f = self.features(x)
        out = torch.cat([f, cond], dim=1)
        pred = self.classifier(out)
        if return_features:
            return pred, f
        return pred

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

def compute_vae_loss(x, x_hat, mu, logvar, beta=0.1):
    recon_loss = 100 * torch.nn.functional.mse_loss(x_hat, x, reduction='mean')
    kl_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    return recon_loss + beta * kl_loss, recon_loss, kl_loss

def load_dataset(dir_path):
    '''Dynamically packages an unzipped PyTorch saved tensor directory into a bytes buffer and loads it'''
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(dir_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, os.path.dirname(dir_path))
                zinfo = zipfile.ZipInfo(arcname, date_time=(1980, 1, 1, 0, 0, 0))
                with open(file_path, 'rb') as fp:
                    zf.writestr(zinfo, fp.read())
    buffer.seek(0)
    return torch.load(buffer, weights_only=False)
