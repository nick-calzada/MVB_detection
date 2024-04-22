import torch
import os
import json
from unet import UNet, ConvPass

class Model(torch.nn.Module):

    def __init__(
            self,
            in_channels=1,
            num_fmaps=12,
            fmap_inc_factor=5,
            downsample_factors=((1,2,2),(1,2,2),(1,3,3)),
            kernel_size_down = (
                ((3, 3, 3), (3, 3, 3)),
                ((1, 3, 3), (1, 3, 3)),
                ((1, 3, 3), (1, 3, 3)),
                ((3, 3, 3), (3, 3, 3))
            ),
            kernel_size_up = (
                ((1, 3, 3), (1, 3, 3)),
                ((1, 3, 3), (1, 3, 3)),
                ((3, 3, 3), (3, 3, 3))
            ),
            padding="valid"):

        super().__init__()

        self.unet = UNet(
                in_channels=in_channels,
                num_fmaps=num_fmaps,
                fmap_inc_factor=fmap_inc_factor,
                downsample_factors=downsample_factors,
                kernel_size_down=kernel_size_down,
                kernel_size_up=kernel_size_up,
                constant_upsample=True,
                padding=padding)

        self.mask_head = ConvPass(num_fmaps, 1, [[1, 1, 1]], activation='Tanh')

    def forward(self, raw):

        z = self.unet(raw)
        mask = self.mask_head(z)

        return mask


class WeightedMSELoss(torch.nn.Module):

    def __init__(self):
        super(WeightedMSELoss, self).__init__()

    def forward(self, pred, target, weights):

        scale = (weights * (pred - target) ** 2)

        if len(torch.nonzero(scale)) != 0:

            mask = torch.masked_select(scale, torch.gt(weights, 0))
            loss = torch.mean(mask)

        else:

            loss = torch.mean(scale)

        return loss
