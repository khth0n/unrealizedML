import torch
import torch.nn as nn

import torch.nn.functional as F

from PIL import Image

import numpy as np
import matplotlib.pyplot as plt

#Import Helper Layers
from SepConv2D import SepConv2d
from NonSepConv2D import NonSepConv2d

rng = np.random.default_rng(0)

class NCAPerception2d(nn.Module):
    
    def __init__(self, in_channels: int, learned_filters: int, device):
        super().__init__()
        
        # Small optimization to use separable convolution where possible
        # to decrease training and inference complexity somewhat.
        
        # NOTE: separable and nonseparable filters are static,
        #       learned filters are updated over training with backprop.
        self.separable, separable_count = self.create_separable_filters(device)
        self.nonseparable, nonseparable_count = self.create_nonseparable_filters(device)
        
        self.learned = self.create_learned_filters(in_channels, learned_filters, device)
        
        # Calculating the number of output channels based on filters defined 
        # in helper functions (in case additional pre-processing filters are used).
        # Specifically useful for defining how to format subsequent layers dynamically.
        self.OUT_CHANNELS = in_channels * (separable_count + nonseparable_count) + learned_filters
        
    def create_separable_filters(self, device):
        
        a = torch.tensor([
            [0, 1, 0],
            [1, 0, -1],
            [1, 2, 1]
        ], dtype=torch.float32, device=device)
        
        b = torch.tensor([
            [0, 1, 0],
            [1, 2, 1],
            [1, 0, -1]
        ], dtype=torch.float32, device=device)
        
        filter_count = len(a)
        
        return (SepConv2d(a, b), filter_count)
    
    def create_nonseparable_filters(self, device):
        
        filters = torch.tensor([
            [
                [0, -1, 0],
                [-1, 4, -1],
                [0, -1, 0]
            ]
        ], dtype=torch.float32, device=device)
        
        filter_count = len(filters)
        
        return (NonSepConv2d(filters), filter_count)
    
    def create_learned_filters(self, in_channels: int, learned_filters: int, device):
        
        return nn.Conv2d(in_channels, learned_filters, 3, device=device)
    
    def forward(self, x: torch.Tensor):
        
        padded_x = F.pad(x, (1, 1, 1, 1), mode='replicate')
    
        #Create feature maps using all of the filters on the input data
        separable_features = self.separable(padded_x)
        nonseparable_features = self.nonseparable(padded_x)
        learned_features = self.learned(padded_x)
        
        features = torch.cat([ separable_features, nonseparable_features, learned_features ], 1)
        
        return features
    
class NCALearner2d(nn.Module):
    
    def __init__(self, perception: NCAPerception2d, fc1: int, fc2: int, device):
        super().__init__()
        
        # Main processing layers responsible for choosing next
        # update that allows the input to converge to the target.
        self.layers = nn.Sequential(
            nn.Conv2d(perception.OUT_CHANNELS, fc1, 1),
            nn.ReLU(),
            nn.Conv2d(fc1, fc2, 1)
        )
        
        self.DEVICE = device
        self.layers.to(device)
    
    def forward(self, x: torch.Tensor):
        
        # Stochastic update to reduce overfitting during training.
        # Using a random binary mask for the update helps learn
        # parameters that result in more effective NCA rules.
        with torch.no_grad():
            mask_shape = (x.shape[0], 1, x.shape[2], x.shape[3])
            rolls = torch.rand(mask_shape, device=self.DEVICE)
            mask = (rolls > 0.5).float() 
            
            return self.layers(x) * mask
        
        return self.layers(x)
    
class NCA2d(nn.Module):
    """
    My implementation of Neural Cellular Automata (NCA)
    
    #### Notes
    - Assumes the following model input structure: `(batch_size, in_channels, grid_size, grid_size)`
    
    #### References
    - [Growing Cellular Automata](https://distill.pub/2020/growing-ca/)
    - [DyNCA](https://dynca.github.io/)
    - [Parameter efficient diffusion with NCA](https://www.nature.com/articles/s44335-025-00026-4)
    """
    
    def __init__(self, in_channels: int, learned_filters: int, hidden_neurons: int, device: str):
        """
        Creates 2D NCA with provided parameters
        #### Args
        `in_channels`
        - Input channels of the input, second value of the assumed input structure.
        #
        `learned_filters`
        - Number of learned filters.
        #
        `hidden_neurons`
        - Number of neurons in convolution-based fully-connected layer
        #
        `device`
        - Current device to move tensor values to.
        """
        super().__init__()
        
        self.to(device)
        
        perception = NCAPerception2d(in_channels, learned_filters, device)
        
        self.layers = nn.Sequential(
            perception,
            NCALearner2d(perception, hidden_neurons, in_channels, device)
        )
    
    def forward(self, x: torch.Tensor):
        
        return x + self.layers(x)
    
    def train(self, in_channels: int, grid_size: int, display_every: int, device: str):
        """
        NCA training function
        
        # [test](output/live.png)
        #### Args
        `in_channels`
        - Input channels of the input, second value of the assumed input structure.
        #### grid_size
        #### display_every
        ####
        *test*
        """
        
        
        def show_sample(state, path='sample.png'):
            img = state.detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy()
            
            img = np.flipud(img)
            
            fig, ax = plt.subplots(figsize=(1, 1), dpi=grid_size)
            
            axis_lim = grid_size - 1
            plt.xlim(0, axis_lim)
            plt.ylim(0, axis_lim)

            plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
            ax.set_axis_off()
            
            plt.imshow(img)
            plt.savefig(path)
            plt.close(fig)
        
        def initialize(batch_size, seed, target):
            
            initial_shape = (batch_size, in_channels, seed.shape[-2], seed.shape[-1])
            seed_batch = torch.zeros(initial_shape).to(device)
            seed_batch[:, :3] = seed.expand(batch_size, -1, -1, -1)
            target_batch = target.unsqueeze(0).expand(batch_size, -1, -1, -1)
            return (seed_batch, target_batch)       
        
        def routine(batch_size, iterations, seeds, targets, max_steps):
            optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
            
            for i in range(iterations):
                
                index = rng.integers(0, len(seeds))
                #index = 1
                seed = seeds[index]
                target = targets[index]
                
                x, target = initialize(batch_size, seed, target)

                # Using an RL approach to training penalizing 
                #Logic: Record absolute difference between
                #       input and output image channels.
                #       Our loss will be the sum of these values.  
                score = torch.zeros((x.shape[2], x.shape[3]), device=device)
                for _ in range(max_steps):  # Evolve solution up to max steps
                    x = self.forward(x)

                    score += torch.sum(F.l1_loss(x[:, :3], target), 0)
                
                loss = torch.log1p(torch.sum(score))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if i % display_every == 0:
                    
                    print(f"[{i}] Loss: {loss.item():.4f}")
                    show_sample(target[0], f'output/target.png')
                    show_sample(x[0, :3], f'output/live.png')
                    
        return routine