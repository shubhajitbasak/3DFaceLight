# https://github.com/chaofengc/Face-SPARNet
# Learning Spatial Attention for Face Super-Resolution - https://arxiv.org/pdf/2012.01211.pdf
from LightExp.Model.sparnetBlocks import *
import torch
from torch import nn
import numpy as np
from torchsummary import summary


class SPARNet(nn.Module):
    """Deep residual network with spatial attention for face SR.
    - gmacs: 9.135652864 , gflops: 4.567826432, params: 7.400013M
    # Arguments:
        - n_ch: base convolution channels
        - down_steps: how many times to downsample in the encoder
        - res_depth: depth of residual layers in the main body
        - up_res_depth: depth of residual layers in each upsample block

    """

    def __init__(
            self,
            num_classes=1000,
            min_ch=32,
            max_ch=320,
            in_size=256,
            out_size=256,
            min_feat_size=8,
            res_depth=5,
            relu_type='leakyrelu',
            norm_type='bn',
            att_name='spar',
            bottleneck_size=4,
    ):
        super(SPARNet, self).__init__()
        nrargs = {'norm_type': norm_type, 'relu_type': relu_type}

        ch_clip = lambda x: max(min_ch, min(x, max_ch))

        down_steps = int(np.log2(in_size // min_feat_size))
        up_steps = int(np.log2(out_size // min_feat_size))
        n_ch = ch_clip(max_ch // int(np.log2(in_size // min_feat_size) + 1))

        # ------------ define encoder --------------------
        self.encoder = []
        self.encoder.append(ConvLayer(3, n_ch, 3, 1))
        hg_depth = int(np.log2(64 / bottleneck_size))
        for i in range(down_steps):
            cin, cout = ch_clip(n_ch), ch_clip(n_ch * 2)
            self.encoder.append(ResidualBlock(cin, cout, scale='down', hg_depth=hg_depth, att_name=att_name, **nrargs))

            n_ch = n_ch * 2
            hg_depth = hg_depth - 1
        hg_depth = hg_depth + 1
        self.encoder = nn.Sequential(*self.encoder)

        # ------------ define residual layers --------------------
        self.res_layers = []
        for i in range(res_depth + 3 - down_steps):
            channels = ch_clip(n_ch)
            self.res_layers.append(ResidualBlock(channels, channels, hg_depth=hg_depth, att_name=att_name, **nrargs))
        self.res_layers = nn.Sequential(*self.res_layers)

        # ------------ define decoder --------------------
        # self.decoder = []
        # for i in range(up_steps):
        #     hg_depth = hg_depth + 1
        #     cin, cout = ch_clip(n_ch), ch_clip(n_ch // 2)
        #     self.decoder.append(ResidualBlock(cin, cout, scale='up', hg_depth=hg_depth, att_name=att_name, **nrargs))
        #     n_ch = n_ch // 2
        #
        # self.decoder = nn.Sequential(*self.decoder)
        # self.classifier_head = nn.Sequential(ConvLayer(ch_clip(n_ch), 3, 3, 1),
        #                                      conv_1x1_bn(max_ch, 1280),
        #                                      nn.AdaptiveAvgPool2d((1, 1)),
        #                                      nn.Linear(1280, num_classes))
        # self.out_conv = ConvLayer(ch_clip(n_ch), 3, 3, 1)
        self.conv = conv_1x1_bn(max_ch, 1280)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(1280, num_classes)

    def forward(self, input_img):
        out = self.encoder(input_img)  # BX320X8X8
        out = self.res_layers(out)  # BX320X8X8
        out = self.conv(out)  # BX1280X8X8
        out = self.avgpool(out)  # BX1280X1X1
        out = out.view(out.size(0), -1)  # BX1280
        out = self.classifier(out)  # BX1000
        return out


if __name__ == '__main__':
    net = SPARNet()
    x = torch.randn(1, 3, 256, 256)
    print(net(x).shape)
    summary(net.cuda(),(3, 256, 256))

