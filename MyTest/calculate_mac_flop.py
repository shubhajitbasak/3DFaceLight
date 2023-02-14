import torch
from thop import profile
from utils.util import get_config
from torchsummary import summary
import timm
from LightExp.Model import sparnet

config_file = 'configs/config.py'
cfg = get_config(config_file)

from models.network import get_network
# net = get_network(cfg).cuda()
# net = timm.create_model('mobilenetv2_100', num_classes=1500).cuda()
# net = timm.create_model('mobilenetv2_100', pretrained=True, num_classes=cfg.keypoints * 3).cuda()
net = sparnet()
input_profile = torch.randn(1, 3, 256, 256).cuda()

print(summary(net, (3, 256, 256)))

macs, params = profile(net, inputs=(input_profile, ))

print('gmacs: {0} , gflops: {1}, params: {2}M'.format(macs/1000000000.0, macs/2000000000.0, params/1000000))


