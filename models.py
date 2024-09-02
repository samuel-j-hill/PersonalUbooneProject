import torch
import torch.nn as nn

class ToyAutoencoder(nn.Module):
    def __init__(self, input_dim, z_dim):
        super(ToyAutoencoder, self).__init__()

        # Encoder layers
        self.en1 = nn.Linear(input_dim, 2000)
        self.en2 = nn.Linear(2000, 1500)
        self.en3 = nn.Linear(1500, 1000)
        self.en4 = nn.Linear(1000, z_dim)

        # Decoder layers
        self.de1 = nn.Linear(z_dim, 1000)
        self.de2 = nn.Linear(1000, 1500)
        self.de3 = nn.Linear(1500, 2000)
        self.de4 = nn.Linear(2000, input_dim)

        self.relu = nn.LeakyReLU()

    def encoder(self, x):
        out = self.en1(x)
        out = self.relu(out)
        out = self.en2(out)
        out = self.relu(out)
        out = self.en3(out)
        out = self.relu(out)
        out = self.en4(out)

        return out

    def decoder(self, z):
        out = self.de1(z)
        out = self.relu(out)
        out = self.de2(out)
        out = self.relu(out)
        out = self.de3(out)
        out = self.relu(out)
        out = self.de4(out)

        return out

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded
    
    
class ConvAutoencoder(nn.Module):
    def __init__(self, z_dim):
        super(ConvAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(512 * 17 * 38, z_dim),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 512 * 17 * 38),
            nn.Unflatten(1, (512, 17, 38)),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=(1, 0)),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=(0, 1)),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=(1,1)),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=(0,1)),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=5, stride=2, padding=2, output_padding=(1,1)),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded