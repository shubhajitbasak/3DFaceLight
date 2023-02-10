import torch
from torch import nn
from MyTest.models.modules import PositionNet, PositionMultiscaleNet

def kaiming_leaky_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')


class CustomMappingNetwork(nn.Module):
    def __init__(self, z_dim, map_hidden_dim, map_output_dim):
        super().__init__()

        self.network = nn.Sequential(nn.Linear(z_dim, map_hidden_dim),
                                     nn.LeakyReLU(0.2, inplace=True),

                                     nn.Linear(map_hidden_dim, map_hidden_dim),
                                     nn.LeakyReLU(0.2, inplace=True),

                                     nn.Linear(map_hidden_dim, map_hidden_dim),
                                     nn.LeakyReLU(0.2, inplace=True),

                                     nn.Linear(map_hidden_dim, map_output_dim))

        self.network.apply(kaiming_leaky_init)
        with torch.no_grad():
            self.network[-1].weight *= 0.25

    def forward(self, z):
        frequencies_offsets = self.network(z)
        frequencies = frequencies_offsets[..., :frequencies_offsets.shape[-1] // 2]
        phase_shifts = frequencies_offsets[..., frequencies_offsets.shape[-1] // 2:]

        return frequencies, phase_shifts


class FiLMLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.layer = nn.Linear(input_dim, hidden_dim)

    def forward(self, x, freq, phase_shift):
        x = self.layer(x)
        return torch.sin(freq * x + phase_shift)


class InstanceNorm(nn.Module):
    def __init__(self, epsilon=1e-8):
        """
            @notice: avoid in-place ops.
            https://discuss.pytorch.org/t/encounter-the-runtimeerror-one-of-the-variables-needed-for-gradient-computation-has-been-modified-by-an-inplace-operation/836/3
        """
        super(InstanceNorm, self).__init__()
        self.epsilon = epsilon

    def forward(self, x):
        x = x - torch.mean(x, (2, 3), True)
        tmp = torch.mul(x, x)  # or x ** 2
        tmp = torch.rsqrt(torch.mean(tmp, (2, 3), True) + self.epsilon)
        return x * tmp


class ApplyStyle(nn.Module):
    def __init__(self, latent_size, channels):
        super(ApplyStyle, self).__init__()
        self.linear = nn.Linear(latent_size, channels * 2)

    def forward(self, x, latent):
        style = self.linear(latent).unsqueeze(2).unsqueeze(3)  # B, 2*c, 1, 1
        gamma, beta = style.chunk(2, 1)
        x = gamma * x + beta
        return x


class ResnetBlock_Adain(nn.Module):
    def __init__(self, dim, latent_size, padding_type='reflect', activation=nn.ReLU(True)):
        super(ResnetBlock_Adain, self).__init__()

        p = 0
        conv1 = []
        if padding_type == 'reflect':
            conv1 += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv1 += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv1 += [nn.Conv2d(dim, dim, kernel_size=3, padding=p), InstanceNorm()]
        self.conv1 = nn.Sequential(*conv1)
        self.style1 = ApplyStyle(latent_size, dim)
        self.act1 = activation

        p = 0
        conv2 = []
        if padding_type == 'reflect':
            conv2 += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv2 += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv2 += [nn.Conv2d(dim, dim, kernel_size=3, padding=p), InstanceNorm()]
        self.conv2 = nn.Sequential(*conv2)
        self.style2 = ApplyStyle(latent_size, dim)

    def forward(self, x, dlatents_in_slice):
        y = self.conv1(x)
        y = self.style1(y, dlatents_in_slice)
        y = self.act1(y)
        y = self.conv2(y)
        y = self.style2(y, dlatents_in_slice)
        out = x + y
        return out


class OneNetwork(nn.Module):
    def __init__(self, cfg):
        super(OneNetwork, self).__init__()
        self.num_verts = cfg.num_verts  # 500
        self.input_size = cfg.img_size  # 256
        kwargs = {}
        num_classes = self.num_verts * 3

        if cfg.network.startswith('resnet'):
            kwargs['base_width'] = int(64 * cfg.width_mult)
        p_num_classes = num_classes

        if cfg.network == 'resnet_jmlr':
            from .resnet import resnet_jmlr
            self.net = resnet_jmlr(num_classes=p_num_classes, **kwargs)
        # else:
        #     self.net = timm.create_model(cfg.network, num_classes = p_num_classes, **kwargs)

    def forward(self, x):
        pred = self.net(x)
        return pred


def get_network(cfg):
    if cfg.use_onenetwork:
        net = OneNetwork(cfg)
    # else:
    #     net = timm.create_model(cfg.network, num_classes = 1220*5)
    return net


class TimmBackbone(nn.Module):
    def __init__(self, name, need_multiscale=True):
        super(TimmBackbone, self).__init__()
        import timm

        self.need_multiscale = need_multiscale

        self.model = timm.create_model(name, features_only=True, pretrained=True)
        self.last_channel = self.model.feature_info[-1]['num_chs']

        if self.need_multiscale:
            self.channels = self.model.feature_info.channels()

    def init_weights(self):
        pass

    def forward(self, x):
        out = self.model(x)
        if not self.need_multiscale:
            return out[-1]
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
    from MyTest.configs import config
    cfg = config.config
    net = Model3DHeatmap(cfg)
    hm, kp = net(torch.randn(1, 3, 256, 256))
    print(hm.shape, kp.shape)

if __name__ == '__main__':
    main()

