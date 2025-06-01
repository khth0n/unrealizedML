import torch
import torch.nn as nn

class SepConv2d(nn.Module):
    
    def __init__(self, a: torch.Tensor, b: torch.Tensor):
        super().__init__()
        
        assert a.shape == b.shape and len(a.shape) == len(b.shape) == 2, 'separable convolution requires stripes of 1D tensors to create kernel!'
        
        self.a = a
        self.b = b
        
    def freeze(self):
        
        for param in self.parameters():
            
            param.requires_grad = False
        
    def forward(self, x: torch.Tensor):
        
        in_channels = x.shape[1]
        
        filter_count = len(self.a)
        filter_size = len(self.a[0])
        
        a_weights = self.a.reshape(filter_count, 1, 1, filter_size).repeat(in_channels, 1, 1, 1)
        b_weights = self.b.reshape(filter_count, 1, filter_size, 1).repeat(in_channels, 1, 1, 1)
        
        intermediate = torch.conv2d(x, a_weights, groups=in_channels)
        
        return torch.conv2d(intermediate, b_weights, groups=in_channels*filter_count)

if __name__ == "__main__":
    
    pass