"""
This script defines the NeRF architecture
"""

import numpy as np
import torch
from torch import nn

def sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-np.sqrt(6 / num_input), np.sqrt(6 / num_input))

def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-1 / num_input, 1 / num_input)

class Siren(nn.Module):
    """
    Siren layer
    """

    def __init__(self, w0=1.0):
        super().__init__()
        self.w0 = w0

    def forward(self, input):
        return torch.sin(self.w0 * input)


class Mapping(nn.Module):
    def __init__(self, mapping_size, in_size, logscale=True):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        in_channels: number of input channels (3 for both xyz and direction)
        """
        super().__init__()
        self.N_freqs = mapping_size
        self.in_channels = in_size
        self.funcs = [torch.sin, torch.cos]
        self.out_channels = self.in_channels*(len(self.funcs)*self.N_freqs+1)

        if logscale:
            self.freq_bands = 2**torch.linspace(0, self.N_freqs-1, self.N_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2**(self.N_freqs-1), self.N_freqs)

    def forward(self, x):
        """
        Embeds x to (x, sin(2^k x), cos(2^k x), ...)
        Different from the paper, "x" is also in the output
        See https://github.com/bmild/nerf/issues/12
        Inputs:
            x: (B, self.in_channels)
        Outputs:
            out: (B, self.out_channels)
        """
        #out = [x]
        out = []
        for freq in self.freq_bands:
            for func in self.funcs:
                out += [func(freq*x)]

        return torch.cat(out, -1)

# Concatenation done after the application of the layer in the NeRF framework. That's wy the skip layer is 5 instead of 4 as suggested in the paper
class NeRF(nn.Module):
    def __init__(self,
                 layers=8, feat=100,
                 input_sizes=[3, 3],
                 skips=[4], siren=False,
                 mapping=True,
                 mapping_sizes=[10, 4],
                 variant="nerf",
                 t_embedding_dims=16):
        """
        layers: integer, number of layers for density (sigma) encoder
        feat: integer, number of hidden units in each layer
        input_sizes: tuple [a, b] where a is the number of input channels for xyz (3*10*2=60 by default)
                                        b is the number of input channels for dir (3*4*2=24 by default)
        skips: list of layer indices, e.g. [i] means add skip connection in the i-th layer
        siren: boolean, use Siren where possible instead of ReLU if True
        mapping: boolean, use positional encoding if True
        mapping_sizes: tuple [a, b] where a and b are the number of freqs for the positional encoding of xyz and dir
        """
        super(NeRF, self).__init__()
        self.layers = layers
        self.skips = skips
        self.mapping = mapping
        self.input_sizes = input_sizes
        self.rgb_padding = 0.001
        self.outputs_per_variant = {
            "nerf": 4,  # r, g, b (3) + sigma (1)
            "s-nerf": 8,  # r, g, b (3) + sigma (1) + sun visibility (1) + r, g, b from sky color (3)
            "s-nerf-w": 11,  # r, g, b (3) + sigma (1) + sun visibility (1) + rgb a (3) + rgb b (3)
        }

        # activation function
        nl = Siren() if siren else torch.nn.ReLU()

        # use positional encoding if specified
        in_size = input_sizes.copy()
        if mapping:
            self.mapping = [Mapping(map_sz, in_sz) for map_sz, in_sz in zip(mapping_sizes, input_sizes)]
            in_size = [2 * map_sz * in_sz for map_sz, in_sz in zip(mapping_sizes, input_sizes)]
        else:
            self.mapping = [nn.Identity(), nn.Identity()]

        # define the main network of fully connected layers, i.e. FC_NET
        fc_layers = []
        fc_layers.append(torch.nn.Linear(in_size[0], feat))
        fc_layers.append(Siren(w0=30.0) if siren else nl)
        for i in range(1, layers):
            if i in skips:
                fc_layers.append(torch.nn.Linear(feat + in_size[0], feat))
            else:
                fc_layers.append(torch.nn.Linear(feat, feat))
            fc_layers.append(nl)
        self.fc_net = torch.nn.Sequential(*fc_layers)  # shared 8-layer structure that takes the encoded xyz vector

        # FC_NET output 1: volume density
        self.sigma_from_xyz = torch.nn.Sequential(torch.nn.Linear(feat, 1), nn.Softplus())

        # FC_NET output 2: vector of features from the spatial coordinates
        self.feats_from_xyz = torch.nn.Linear(feat, feat) # No non-linearity here in the original paper

        # the FC_NET output 2 is concatenated to the encoded viewing direction input
        # and the resulting vector of features is used to predict the rgb color
        self.rgb_from_xyzdir = torch.nn.Sequential(torch.nn.Linear(feat + in_size[1], feat // 2), nl,
                                                   torch.nn.Linear(feat // 2, 3), torch.nn.Sigmoid())

        # Shadow-NeRF (s-nerf) additional layers
        self.variant = variant
        if self.variant in ["s-nerf", "s-nerf-w"]:
            sun_dir_in_size = 3
            sun_v_layers = []
            sun_v_layers.append(torch.nn.Linear(feat + sun_dir_in_size, feat // 2))
            sun_v_layers.append(Siren() if siren else nl)
            for i in range(1, 3):
                sun_v_layers.append(torch.nn.Linear(feat // 2, feat // 2))
                sun_v_layers.append(nl)
            sun_v_layers.append(torch.nn.Linear(feat // 2, 1))
            sun_v_layers.append(torch.nn.Sigmoid())
            self.sun_v_net = torch.nn.Sequential(*sun_v_layers)

        if self.variant == "s-nerf":
            self.sky_color = torch.nn.Sequential(
                torch.nn.Linear(sun_dir_in_size, feat // 2),
                torch.nn.ReLU(),
                torch.nn.Linear(feat // 2, 3),
                torch.nn.Sigmoid(),
            )

        if self.variant == "s-nerf-w":
            self.ambient_encoding = torch.nn.Sequential(
                torch.nn.Linear(t_embedding_dims, feat // 4), nl,
                torch.nn.Linear(feat // 4, feat // 4), nl
            )
            self.ambientA = nn.Sequential(nn.Linear(feat // 4, 3), nn.Sigmoid())
            self.ambientB = nn.Sequential(nn.Linear(feat // 4, 3), nn.Sigmoid())

        if siren:
            self.fc_net.apply(sine_init)
            self.fc_net[0].apply(first_layer_sine_init)
            if self.variant in ["s-nerf", "s-nerf-w"]:
                self.sun_v_net.apply(sine_init)
                self.sun_v_net[0].apply(first_layer_sine_init)


    def forward(self, input_xyz, input_dir=None, input_sun_dir=None, input_t=None, sigma_only=False):
        """
        Predicts the values rgb, sigma from a batch of input rays
        the input rays are represented as a set of 3d points xyz

        Args:
            input_xyz: (B, 3) input tensor, with the 3d spatial coordinates, B is batch size
            sigma_only: boolean, infer sigma only if True, otherwise infer both sigma and color

        Returns:
            if sigma_ony:
                sigma: (B, 1) volume density
            else:
                out: (B, 4) first 3 columns are rgb color, last column is volume density
        """

        # compute shared features
        input_xyz = self.mapping[0](input_xyz)
        xyz_ = input_xyz
        for i in range(self.layers):
            if i in self.skips:
                xyz_ = torch.cat([input_xyz, xyz_], -1)
            xyz_ = self.fc_net[2*i](xyz_)
            xyz_ = self.fc_net[2*i + 1](xyz_)
        shared_features = xyz_

        # compute volume density
        sigma = self.sigma_from_xyz(shared_features)
        if sigma_only:
            return sigma

        # compute color
        xyz_features = self.feats_from_xyz(shared_features)
        if self.input_sizes[1] > 0:
            input_xyzdir = torch.cat([xyz_features, self.mapping[1](input_dir)], -1)
        else:
            input_xyzdir = xyz_features
        rgb = self.rgb_from_xyzdir(input_xyzdir)
        # improvement suggested by Jon Barron to help stability (same paper as soft+ suggestion)
        rgb = rgb * (1 + 2 * self.rgb_padding) - self.rgb_padding

        out = torch.cat([rgb, sigma], 1) # (B, 4)

        if self.variant == "s-nerf":
            # shadow nerf outputs
            input_sun_v_net = torch.cat([xyz_features, input_sun_dir], -1)
            sun_v = self.sun_v_net(input_sun_v_net)
            sky_color = self.sky_color(input_sun_dir)
            out = torch.cat([out, sun_v, sky_color], 1) # (B, 8)

        if self.variant == "s-nerf-w":
            # sat-nerf outputs
            input_sun_v_net = torch.cat([xyz_features, input_sun_dir], -1)
            sun_v = self.sun_v_net(input_sun_v_net)
            if input_t is None:
                a = torch.ones(sun_v.shape[0], 3).cuda()
                b = torch.zeros(sun_v.shape[0], 3).cuda()
            else:
                ambient_features = self.ambient_encoding(input_t)
                a = self.ambientA(ambient_features)
                b = self.ambientB(ambient_features)

            out = torch.cat([out, sun_v, a, b], 1)  # (B, 11)

        return out