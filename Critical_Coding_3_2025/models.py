from torch import nn


# Conv2D is a custom convolutional layer module. It consists of a 2D convolution, 
# followed by batch normalization and an activation function (LeakyReLU by default).
class StrideConv2D(nn.Module):
  def __init__(self, in_channel, out_channel, transpose = False, stride = 2, padding = 1, activation = True, batch_norm = True) -> None:
    super().__init__()
    self.in_channel = in_channel
    self.out_channel = out_channel

    modules = []
    if transpose:
        modules.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size=4, stride=stride, padding=padding, bias=False))
    else:
        modules.append(nn.Conv2d(in_channel, out_channel, kernel_size=4, stride=stride, padding=padding, bias=False))
        
    modules.append(nn.LeakyReLU() if activation else nn.Identity())
    modules.append(nn.BatchNorm2d(out_channel) if batch_norm else nn.Identity())
    
    self.convolutions = nn.ModuleList(modules)

  def forward(self, x):
    assert x.shape[1] == self.in_channel, f'input shape {x.shape} at index 1 doesnt match expected {self.in_channel}'

    for layer in self.convolutions:
      x = layer(x)
    
    return x

class Generator(nn.Module):
    def __init__(self, resolution, z_dim, c_base, colour_channels):
        super(Generator, self).__init__()
        
        self.resolution      = resolution
        self.z_dim           = z_dim
        self.colour_channels = colour_channels
        
        self.main = nn.Sequential(
            # [Batch, z_dim, 1, 1] -> [Batch, c_base*8, 4, 4]
            StrideConv2D(z_dim,    c_base*8, transpose = True, stride = 1, padding = 0),
            
            # [Batch, c_base*8, 4, 4] -> [Batch, c_base*4, 8, 8]
            StrideConv2D(c_base*8, c_base*4, transpose = True, stride = 2, padding = 1),
            
            # [Batch, c_base*8, 4, 4] -> [Batch, c_base*4, 8, 8]
            StrideConv2D(c_base*4, c_base*4, transpose = True, stride = 2, padding = 1),
            
            # [Batch, c_base*4, 8, 8] -> [Batch, c_base*2, 16, 16]
            StrideConv2D(c_base*4, c_base*2, transpose = True, stride = 2, padding = 1),
            
            # [Batch, c_base*2, 16, 16] -> [Batch, c_base, 32, 32]
            StrideConv2D(c_base*2, c_base*1, transpose = True, stride = 2, padding = 1),
            
            # [Batch, z_dim, 32, 32] -> [Batch, colour_channels, 32, 32]
            nn.Conv2d(c_base*1, colour_channels, kernel_size = 3, stride = 1, padding = 1),
            
            nn.Tanh()
        )

    def forward(self, x):
        
        assert x.dim() == 2, f'input dimension should be [Batch, z_dim], got {x.shape}'
        assert x.shape[1] == self.z_dim, f'input shape {x.shape} at index 1 doesnt match expected {self.z_dim}'
        
        x = x.unsqueeze(2).unsqueeze(3)
        return self.main(x)
    
class Discriminator(nn.Module):
    def __init__(self, resolution, c_base, colour_channels):
        super(Discriminator, self).__init__()
        self.resolution = resolution
        self.colour_channels = colour_channels
        
        self.main = nn.Sequential(
            
            # [Batch, colour_c, 32, 32] -> [Batch, c_base, 32, 32]
            nn.Conv2d(colour_channels, c_base, kernel_size = 3, stride = 1, padding = 1),
            
            # [Batch, c_base, 32, 32] -> [Batch, c_base*2, 16, 16]
            StrideConv2D(c_base, c_base*2, stride = 2, padding = 1, batch_norm = False),
            
            # [Batch, c_base*2, 16, 16] -> [Batch, c_base*4, 8, 8]
            StrideConv2D(c_base*2, c_base*4, stride = 2, padding = 1),
            
            # [Batch, c_base*2, 16, 16] -> [Batch, c_base*4, 8, 8]
            StrideConv2D(c_base*4, c_base*4, stride = 2, padding = 1),
            
            # [Batch, c_base*4, 8, 8] -> [Batch, c_base*8, 4, 4]
            StrideConv2D(c_base*4, c_base*8, stride = 2, padding = 1),
            
            # [Batch, c_base*8, 4, 4] -> [Batch, 1, 1, 1]
            StrideConv2D(c_base*8, 1, stride = 1, padding = 0, activation = False, batch_norm = False),
            
            nn.Sigmoid()
        )

    def forward(self, x):
        assert x.shape[1] == self.colour_channels, f'input shape {x.shape} at index 1 doesnt match expected {self.colour_channels}'
        
        return self.main(x).squeeze()