import torch
from torch import nn
from HeatMap.models.modules import PositionNet, PositionMultiscaleNet


class TimmBackbone(nn.Module):
    def __init__(self, name, need_multiscale=True):
        super(TimmBackbone, self).__init__()
        import timm

        self.need_multiscale = need_multiscale

        self.model = timm.create_model(name, features_only=True, pretrained=True)
        self.last_channel = self.model.feature_info[-2]['num_chs']

        if self.need_multiscale:
            self.channels = self.model.feature_info.channels()

    def init_weights(self):
        pass

    def forward(self, x):
        out = self.model(x)
        if not self.need_multiscale:
            return out[-2]
        return out


def init_weights(m):
    if type(m) == nn.ConvTranspose2d:
        nn.init.normal_(m.weight, std=0.001)
    elif type(m) == nn.Conv2d:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif type(m) == nn.BatchNorm2d:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)
        nn.init.constant_(m.bias, 0)


class Model3DHeatmap(nn.Module):
    def __init__(self, cfg):
        super(Model3DHeatmap, self).__init__()
        self.cfg = cfg
        self.timmModel = TimmBackbone(cfg.model_name, cfg.multiscale)
        self.backbone = self.timmModel

        # backbone = timm.create_model('mobilenetv3_small_050', pretrained=True, num_classes=0)
        if cfg.multiscale:
            self.position_net = PositionMultiscaleNet(self.cfg, self.backbone.channels)
        else:
            self.position_net = PositionNet(self.cfg, self.backbone.last_channel)  # backbone.last_channel  backbone.num_features
        # self.imageNetNorm = self.cfg.imageNetNorm
        if self.cfg.pretrained:
            self.backbone.init_weights()
            self.position_net.apply(init_weights)

    def forward(self, image):
        # if self.imageNetNorm:
        #     mean = (0.485, 0.456, 0.406)
        #     std = (0.229, 0.224, 0.225)
        #     # image = (image - mean) / std
        #     image = TVF.normalize(image, mean, std, False)

        img_feat = self.backbone(image)
        heatmap, keypoints = self.position_net(img_feat)

        return heatmap, keypoints


def main():
    from HeatMap.conf import conf
    cfg = conf.config
    net = Model3DHeatmap(cfg)
    hm, kp = net(torch.randn(1, 3, 256, 256))
    print(hm.shape, kp.shape)


if __name__ == '__main__':
    main()