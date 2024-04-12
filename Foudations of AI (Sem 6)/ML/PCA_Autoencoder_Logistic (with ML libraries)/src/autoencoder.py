import torch
from tqdm.auto import tqdm
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

"""
Implementation of Autoencoder
"""
class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, encoding_dim: int) -> None:
        """
        Modify the model architecture here for comparison
        """
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim),
            nn.Linear(encoding_dim, encoding_dim//2),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim//2, encoding_dim),
            nn.Linear(encoding_dim, input_dim),
        )
    
    def forward(self, x):
        #TODO: 5%
        #input to output
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def fit(self, X, epochs=10, batch_size=32):

        #TODO: 5%
        data_loader = torch.utils.data.DataLoader(dataset = torch.tensor(X, dtype=torch.float32), batch_size= batch_size, shuffle = True)
        
        criterion = nn.MSELoss()
        #learning rate = 0.001
        optimizer = optim.Adam(self.parameters(), lr = 0.001)

        for i in range(epochs):
            for j in data_loader:
                input = j
                output = self.forward(input)
                #do MSE
                loss = criterion(output, input)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


    def transform(self, X):
        #TODO: 2%
        encoded = self.encoder(torch.tensor(X, dtype=torch.float32))
        return encoded.detach().numpy()

    def reconstruct(self, X):
        #TODO: 2%
        #X is already a tensor
        decoded = self.forward(X)
        return decoded.detach().numpy()


"""
Implementation of DenoisingAutoencoder
"""
class DenoisingAutoencoder(Autoencoder):
    def __init__(self, input_dim, encoding_dim, noise_factor=0.2):
        super(DenoisingAutoencoder, self).__init__(input_dim,encoding_dim)
        self.noise_factor = noise_factor
    
    def add_noise(self, x):
        #TODO: 3%
        noise = torch.randn(x.size()) * self.noise_factor
        noisy = x + noise
        return noisy
    
    def fit(self, X, epochs=10, batch_size=32):
        #TODO: 4%
        data_loader = torch.utils.data.DataLoader(dataset = torch.tensor(X, dtype=torch.float32), batch_size= batch_size, shuffle = True)
        
        criterion = nn.MSELoss()
        #learning rate = 0.001
        optimizer = optim.Adam(self.parameters(), lr = 0.001)

        for i in range(epochs):
            for j in data_loader:
                input = j   
                output_noise = self.add_noise(input)
                output = self.forward(output_noise)

                #do MSE
                loss = criterion(output, input)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()



