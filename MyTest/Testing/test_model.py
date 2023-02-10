import timm
import torch
from torchsummary import summary
# from autoencoder import DepthEncoder, DepthDecoder
# from sklearn.metrics import f1_score


model = timm.create_model('mobilenetv2_100', pretrained=True, num_classes=1500)
x = torch.randn(1, 3, 256, 256)
x1 = model(x)
# print(summary(model.cuda(), (3, 256, 256)))
# print(x1.shape)
print(model.classifier)
print(model.feature_info.channels())

# depth_num_layers = 50  # 18
# depth_pretrained_path = '../checkpoints/resnet_pretrained/resnet50-19c8e357.pth'
# DepthEncoder = DepthEncoder(depth_num_layers, depth_pretrained_path)
# DepthDecoder = DepthDecoder(DepthEncoder.num_ch_enc)
#
# x = torch.randn(1, 3, 256, 256)
# x1 = DepthEncoder(x)
#
# print(x1[0].shape)
# print(x1[1].shape)
# print(x1[2].shape)
# print(x1[3].shape)
# print(x1[4].shape)

# import numpy
# def wer(r, h):
#     """
#     Source: https://martin-thoma.com/word-error-rate-calculation/
#     Calculation of WER with Levenshtein distance.
#     Works only for iterables up to 254 elements (uint8).
#     O(nm) time ans space complexity.
#     Parameters
#     ----------
#     r : list
#     h : list
#     Returns
#     -------
#     int
#     Examples
#     --------
#     >>> wer("who is there".split(), "is there".split())
#     1
#     >>> wer("who is there".split(), "".split())
#     3
#     >>> wer("".split(), "who is there".split())
#     3
#     """
#     # initialisation
#     d = numpy.zeros((len(r)+1)*(len(h)+1), dtype=numpy.uint8)
#     d = d.reshape((len(r)+1, len(h)+1))
#     for i in range(len(r)+1):
#         for j in range(len(h)+1):
#             if i == 0:
#                 d[0][j] = j
#             elif j == 0:
#                 d[i][0] = i
#
#     # computation
#     for i in range(1, len(r)+1):
#         for j in range(1, len(h)+1):
#             if r[i-1] == h[j-1]:
#                 d[i][j] = d[i-1][j-1]
#             else:
#                 substitution = d[i-1][j-1] + 1
#                 insertion    = d[i][j-1] + 1
#                 deletion     = d[i-1][j] + 1
#                 d[i][j] = min(substitution, insertion, deletion)
#
#     return d[len(r)][len(h)]
#
# def wer_sentence(r, h):
#     return wer(r.split(), h.split())

# print(wer_sentence('lay blue in x six please', 'lay blue at o five please'))

