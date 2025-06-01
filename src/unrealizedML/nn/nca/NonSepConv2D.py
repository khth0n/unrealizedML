import torch
import torch.nn as nn

class NonSepConv2d(nn.Module):
    
    def __init__(self, filters: torch.Tensor):
        super().__init__()
        
        assert len(filters.shape) == 3, 'Create filters using shape of (filter_count, filter_width, filter_height)'
        
        self.filters = filters
        
    def freeze(self):
        
        for param in self.parameters():
            
            param.requires_grad = False
        
    def forward(self, x: torch.Tensor):
        
        in_channels = x.shape[1]
        
        filter_width = self.filters.shape[-2]
        filter_height = self.filters.shape[-1]
        
        filter_count = len(self.filters)

        filter_weights = self.filters.reshape(filter_count, 1, filter_width, filter_height)
        filter_weights = filter_weights.repeat(in_channels, 1, 1, 1)

        return torch.conv2d(x, filter_weights, groups=in_channels)