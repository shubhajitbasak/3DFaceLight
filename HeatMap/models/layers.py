import torch.nn as nn


def make_conv_layers(feat_dims, kernel=3, stride=1, padding=1, bnrelu_final=True):
    layers = []
    for i in range(len(feat_dims) - 1):
        layers.append(
            nn.Conv2d(
                in_channels=feat_dims[i],
                out_channels=feat_dims[i + 1],
                kernel_size=kernel,
                stride=stride,
                padding=padding
            ))
        # Do not use BN and ReLU for final estimation
        if i < len(feat_dims) - 2 or (i == len(feat_dims) - 2 and bnrelu_final):
            layers.append(nn.BatchNorm2d(feat_dims[i + 1]))
            layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)


def make_dwconv_layers(feat_dims, kernel=3, stride=1, padding=1, bnrelu_final=True):
    layers = []
    for i in range(len(feat_dims) - 1):
        layers.append(
            nn.Conv2d(
                kernel_size=kernel,
                in_channels=feat_dims[i],
                out_channels=feat_dims[i],
                stride=stride,
                padding=padding,
                groups=feat_dims[i]))
        layers.append(nn.BatchNorm2d(feat_dims[i]))
        layers.append(nn.ReLU(inplace=True))
        # pointwise
        layers.append(nn.Conv2d(feat_dims[i], feat_dims[i + 1], kernel_size=1, stride=1, padding=0))

        # Do not use BN and ReLU for final estimation
        if i < len(feat_dims) - 2 or (i == len(feat_dims) - 2 and bnrelu_final):
            layers.append(nn.BatchNorm2d(feat_dims[i + 1]))
            layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)