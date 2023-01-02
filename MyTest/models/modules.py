import torch
import torch.nn as nn
from MyTest.utils.util import soft_argmax_3d
from MyTest.models.layers import make_conv_layers, make_dwconv_layers


class PositionNet(nn.Module):
    def __init__(self, cfg, in_channel):
        super(PositionNet, self).__init__()
        self.joint_num = cfg.keypoints
        if cfg.model_position_use_dw:
            self.conv = make_dwconv_layers([in_channel, self.joint_num * self.cfg.output_hm_shape[0]], kernel=3,
                                           stride=1, padding=1, bnrelu_final=False)
        self.output_hm_shape = cfg.output_hm_shape
        self.conv = make_conv_layers([in_channel,
                                      self.joint_num * self.output_hm_shape[0]],
                                     kernel=1, stride=1, padding=0, bnrelu_final=False)

    def forward(self, img_feat):
        joint_hm = self.conv(img_feat).view(-1,
                                            self.joint_num,
                                            self.output_hm_shape[0],
                                            self.output_hm_shape[1],
                                            self.output_hm_shape[2])
        joint_coord = soft_argmax_3d(joint_hm)
        return joint_coord


class PositionMultiscaleNet(nn.Module):
    def __init__(self, cfg, channels):
        super(PositionMultiscaleNet, self).__init__()
        self.cfg = cfg
        in_channel = channels[-1]
        in_channel2 = channels[-2]
        self.joint_num = cfg.keypoints
        kernel_size = cfg.kernel_size
        if kernel_size == 1:
            padding = 0
        elif kernel_size == 3:
            padding = 1

        self.model_position_use_dw = cfg.model_position_use_dw

        if self.model_position_use_dw:
            self.conv = make_dwconv_layers([in_channel, self.joint_num * self.cfg.output_hm_shape[0]], kernel=3,
                                           stride=1, padding=1, bnrelu_final=True)
            self.convh = make_dwconv_layers([in_channel2, self.joint_num * self.cfg.output_hm_shape[0]], kernel=3,
                                            stride=1, padding=1, bnrelu_final=True)
        else:
            self.conv = make_conv_layers([in_channel, self.joint_num * self.cfg.output_hm_shape[0]], kernel=kernel_size,
                                         stride=1, padding=padding, bnrelu_final=True)
            self.convh = make_conv_layers([in_channel2, self.joint_num * self.cfg.output_hm_shape[0]],
                                          kernel=kernel_size, stride=1, padding=padding, bnrelu_final=True)
        self.pool = nn.AdaptiveAvgPool2d((self.cfg.output_hm_shape[1], self.cfg.output_hm_shape[2]))
        final_conv_kernel_size = cfg.final_conv_kernel_size
        if final_conv_kernel_size == 1:
            final_conv_padding = 0
        elif final_conv_kernel_size == 3:
            final_conv_padding = 1

        self.final_conv = make_conv_layers(
            [2 * self.joint_num * self.cfg.output_hm_shape[0], self.joint_num * self.cfg.output_hm_shape[0]],
            kernel=final_conv_kernel_size, stride=1, padding=final_conv_padding, bnrelu_final=False)

    def forward(self, img_feats):
        img_feat = img_feats[-1]
        joint_hm = self.conv(img_feat)
        t = self.convh(img_feats[-2])
        joint_hm2 = self.pool(t)

        joints = torch.cat((joint_hm, joint_hm2), dim=1)
        joint_hm = self.final_conv(joints).view(-1, self.joint_num, self.cfg.output_hm_shape[0],
                                                self.cfg.output_hm_shape[1], self.cfg.output_hm_shape[2])

        joint_coord = soft_argmax_3d(joint_hm)
        return joint_coord

