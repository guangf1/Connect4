import torch as th
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):  # Increased feature dimensions
        super().__init__(observation_space, features_dim)
        
        n_input_channels = 1
        n_rows = 6
        n_cols = 7
        
        # Enhanced CNN architecture with deeper network
        self.cnn = nn.Sequential(
            # First conv block
            nn.Conv2d(n_input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # Second conv block
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # Third conv block - especially good for detecting connect-4 patterns
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # Additional conv to better learn game patterns
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Flatten(),
        )
        
        # Compute output shape
        with th.no_grad():
            sample = th.zeros(1, n_input_channels, n_rows, n_cols)
            n_flatten = self.cnn(sample).shape[1]
        
        # Deeper MLP head
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, 512),
            nn.ReLU(),
            nn.Linear(512, features_dim),
            nn.ReLU(),
        )
    
    def forward(self, observations):
        batch_size = observations.shape[0]
        reshaped = observations.view(batch_size, 1, 6, 7)
        return self.linear(self.cnn(reshaped))

'''
policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=256),  # Larger feature space
)'
'''