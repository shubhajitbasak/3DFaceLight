import torch.nn as nn
from utils.util import soft_argmax_3d
from .layers import make_conv_layers


class PositionNet(nn.Module):
    def __init__(self, keypoint_num, in_channel):
        super(PositionNet, self).__init__()
        self.joint_num = keypoint_num
        # if getBooleanFromCfg(cfg, 'model_position_use_dw'):
        #     self.conv = make_dwconv_layers([in_channel,self.joint_num*self.cfg.output_hm_shape[0]], kernel=3, stride=1, padding=1, bnrelu_final=False)
        self.output_hm_shape = (8, 8, 8)
        self.conv = make_conv_layers([in_channel,
                                      self.joint_num*self.output_hm_shape[0]],
                                     kernel=1, stride=1, padding=0, bnrelu_final=False)

    def forward(self, img_feat):
        joint_hm = self.conv(img_feat).view(-1,
                                            self.joint_num,
                                            self.output_hm_shape[0],
                                            self.output_hm_shape[1],
                                            self.output_hm_shape[2])
        joint_coord = soft_argmax_3d(joint_hm)
        return joint_coord
