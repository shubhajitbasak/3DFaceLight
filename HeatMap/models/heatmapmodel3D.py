import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

sys.path.insert(0, '..')
from models.mobilenet import mobilenetv2


def heatmap2coord(heatmap, topk=7):
    N, C, H, W = heatmap.shape
    score, index = heatmap.view(N, C, 1, -1).topk(topk, dim=-1)
    # coord = torch.cat([index % W, index // W], dim=2)
    coord = torch.cat([index % W, torch.div(index, W, rounding_mode='floor')], dim=2)
    return (coord * F.softmax(score, dim=-1)).sum(-1)


class BinaryHeadBlock(nn.Module):
    """BinaryHeadBlock
    """

    def __init__(self, in_channels, proj_channels, out_channels, **kwargs):
        super(BinaryHeadBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, proj_channels, 1, bias=False),
            nn.BatchNorm2d(proj_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(proj_channels, out_channels * 2, 1, bias=False),
        )

    def forward(self, input):
        N, C, H, W = input.shape
        binary_heats = self.layers(input).view(N, 2, -1, H, W)

        return binary_heats


class BinaryHeatmap2Coordinate(nn.Module):
    """BinaryHeatmap2Coordinate
    """

    def __init__(self, stride=4.0, topk=5, **kwargs):
        super(BinaryHeatmap2Coordinate, self).__init__()
        self.topk = topk
        self.stride = stride

    def forward(self, input):
        return self.stride * heatmap2coord(input[:, 1, ...], self.topk)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'topk={}, '.format(self.topk)
        format_string += 'stride={}'.format(self.stride)
        format_string += ')'
        return format_string


class HeatmapHead(nn.Module):
    """HeatmapHead
    """

    def __init__(self, kp_num):
        super(HeatmapHead, self).__init__()

        self.decoder = BinaryHeatmap2Coordinate(topk=18, stride=4)

        self.head = BinaryHeadBlock(in_channels=152, proj_channels=152, out_channels=kp_num)

    def forward(self, input):
        binary_heats = self.head(input)
        lmks = self.decoder(binary_heats)

        return binary_heats, lmks


class HeatMapLandmarker(nn.Module):
    def __init__(self, kp_num=520, pretrained=False, model_url=None):
        super(HeatMapLandmarker, self).__init__()
        self.backbone = mobilenetv2(pretrained=pretrained, model_url=model_url)
        self.heatmap_head = HeatmapHead(kp_num)
        # self.transform = transforms.Compose([
        #     transforms.Resize((256, 256)),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # ])

    def forward(self, x):
        heatmaps, landmark = self.heatmap_head(self.backbone(x))

        # Note that the 0 channel indicate background
        return heatmaps[:, 1, ...], landmark


if __name__ == "__main__":
    import time

    torch.manual_seed(0)

    # Inference model
    x = torch.rand((16, 3, 256, 256))
    model = HeatMapLandmarker(pretrained=False)
    heatmaps, lmks = model(x)
    print(f'\nheat size :{heatmaps.shape}.\n\nlmks shape :{lmks.shape}')
