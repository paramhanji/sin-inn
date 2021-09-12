import math
from typing import List, Tuple, Union
import torch as torch
from torch.functional import Tensor
import torch.nn as nn

class Sine(nn.Module):
    def __init__(self, omega_0 = 30):
        nn.Module.__init__(self)
        self.omega_0 = omega_0

    def forward(self, x: Tensor) -> Tensor:
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(self.omega_0 * x)

class Layer(nn.Module):
    def __init__(self, in_channels, out_channels, activation='relu', is_first=False):
        """
        One layer of Perceptron
        in_channels: the number of channels for input
        out_channels: the number of channels for output
        relu: if relu is used at the end
        """
        nn.Module.__init__(self)
        self.linear = nn.Linear(in_channels, out_channels)
        nn.init.constant_(self.linear.bias, 0)
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
            nn.init.xavier_uniform_(self.linear.weight)
        elif activation == 'siren':
            omega_0 = 30.0
            self.activation = Sine(omega_0)
            if is_first:
                r = 1 / in_channels
            else:
                r = math.sqrt(6.0 / in_channels) / omega_0
            nn.init.uniform_(self.linear.weight, -r, r)
        else:
            self.activation = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear(x)
        return self.activation(x)

class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim, hidden_layers, activation):
        """
        Multi-layer perceptron
        in_channels: the number of channels for input
        out_channels: the number of channels for output
        hidden_dim: the number of channels for hidden_layers
        hidden_layers: how many hidden layers in total
        activation: relu or siren
        """
        nn.Module.__init__(self)
        self.layers = nn.Sequential()
        self.layers.add_module('input', Layer(in_channels, hidden_dim, activation, is_first=True))
        for i in range(1, hidden_layers):
            self.layers.add_module(f'hidden{i+1}', Layer(hidden_dim, hidden_dim, activation))
        self.layers.add_module('output', Layer(hidden_dim, out_channels, activation=None))
    
    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)

class PositionalEncoder(nn.Module):
    def __init__(self, size: Tuple[int, int]):
        """
        the positional encoder with sin(2^Lp) and cos(2^Lp)
        size: tuple, the actual dimension of the encoder
        """
        nn.Module.__init__(self)
        self.dims = [self.get_num_frequencies_nyquist(s) for s in size]
        self.mask_dims = [dim if not const else 1 
            for dim in self.dims for const in [True, False, False]]
        self.out_channels = sum(dim*2+1 for dim in self.dims)

    def forward(self, poses) -> Tensor:
        outputs = []
        for dim, pos in zip(self.dims, poses.unbind(-1)):
            pos = pos.unsqueeze(-1)
            code = 2 * math.pi * torch.pow(2, torch.arange(0, dim).to(pos)) * pos
            outputs.append(pos)
            outputs.append(code.sin())
            outputs.append(code.cos())
        return torch.cat(outputs, -1)

    def get_num_frequencies_nyquist(self, size: int) -> int:
        """get the nyquist frequency given a size"""
        nyquist_rate = size / 4
        return int(math.floor(math.log(nyquist_rate, 2)))

class GaussianEncoder(nn.Module):
    def __init__(self, size: Tuple[int, int], sigma=1):
        nn.Module.__init__(self)
        self.dims = [s for s in size for _ in 'xy']
        self.B = nn.Linear(2, sum(self.dims), bias=False)
        self.out_channels = self.B.out_features * 2
        self.mask_dims = [self.out_channels]
        nn.init.normal_(self.B.weight, 0.0, sum(size)*0.5)

    def forward(self, poses: Tensor) -> Tensor:
        code = 2 * math.pi * self.B(poses)
        return torch.cat((torch.sin(code), torch.cos(code)), -1)
    

class ProgressiveMask(nn.Module):
    def __init__(self, dims: List[int], T: int = 1, d: int = 1):
        """
        progressive encoding
        dims: tuple, the actual dimension of the encoding
        T: the total number of iteration for optimisation
        """
        nn.Module.__init__(self)
        self.dims = dims
        self.T = T
        self.d = d

    def forward(self, t: int) -> Union[Tensor, float]:
        if not self.training:
            return 1
        masks = []
        for dim in self.dims:
            masks.append(self.alpha(dim, t)) # cos terms
        return torch.cat(masks).view(1, -1)

    def alpha(self, n: int, t: int) -> Tensor:
        """
        Equation 4 of the paper
        """
        I = torch.arange(0, n).float()
        alpha = (t*self.T/(2*n)+self.d-I).clamp(0, 1)
        alpha[self.d:] = 1.0
        return alpha

class BaseNet(nn.Module):
    def __init__(self, size=(256, 256), out_channels=3, hidden_dim=256, hidden_layers=3):
        nn.Module.__init__(self)
        in_channels = 2
        self.layers = MLP(
            in_channels,
            out_channels,
            hidden_dim,
            hidden_layers,
            activation='relu')

    def forward(self, poses):
        return self.layers(poses)

class SIREN(nn.Module):
    def __init__(self, size=(256, 256), out_channels=3, hidden_dim=256, hidden_layers=3):
        """
        size: the dimension of the image
        out_channels: the number of channels for output, should be 1 if grayscale or 3 if rgb
        hidden_dim: the number of channels for hidden_layers
        hidden_layers: how many hidden layers in total
        """
        nn.Module.__init__(self)
        self.size = size
        in_channels = 2 # coordinates
        self.layers = MLP(
            in_channels,
            out_channels,
            hidden_dim,
            hidden_layers,
            activation='siren')
    
    def forward(self, poses: Tensor) -> Tensor:
        return self.layers(poses)
