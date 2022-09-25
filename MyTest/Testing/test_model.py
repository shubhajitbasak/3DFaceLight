import timm
import torch
from torchsummary import summary

model = timm.create_model('mobilenetv2_100', pretrained=True, num_classes=1500)

# x = torch.randn(1, 3, 256, 256)
#
# x1 = model(x)


print(summary(model.cuda(), (3, 256, 256)))

# print(x1.shape)
