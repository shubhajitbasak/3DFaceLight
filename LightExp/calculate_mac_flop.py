import torch
from thop import profile
from torchsummary import summary
from Model.sparnet import SPARNet
from Model.hnct import HNCT
from Model.HiFuse import HiFuse_Tiny

net = SPARNet().cuda()
# net = HNCT().cuda()
# net = HiFuse_Tiny(1560).cuda()  # 224
input_profile = torch.randn(1, 3, 256, 256).cuda()

# print(summary(net, (3, 256, 256)))

macs, params = profile(net, inputs=(input_profile, ))

print('gmacs: {0} , gflops: {1}, params: {2}M'.format(macs/1000000000.0, macs/2000000000.0, params/1000000))


