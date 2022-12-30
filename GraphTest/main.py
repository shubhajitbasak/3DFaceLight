import timm
import torch

# print(timm.list_models('mobilenetv3*',pretrained=True))



m = timm.create_model('mobilenetv3_small_050', pretrained=True, num_classes=0)

o = m(torch.randn(1, 3, 256, 256))
print(f'Original shape: {o.shape}')
o = m.forward_features(torch.randn(1, 3, 256, 256))
print(f'Unpooled shape: {o.shape}')

