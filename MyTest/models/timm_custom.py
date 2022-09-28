from torch import nn
import timm
from collections import OrderedDict


class CustomTimmModel(nn.Module):
    def __init__(self, cfg, pretrained=True):
        super().__init__()
        self.N_CLASS = cfg.N_CLASS*4
        self.cnn = timm.create_model(cfg.model_name, pretrained=pretrained, num_classes=self.N_CLASS)
        n_inputs = self.cnn.fc.in_features
        self.outVariance = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(n_inputs, self.N_CLASS)),
            ('relu', nn.ReLU)
        ]))

    def forward(self, x):
        preds = self.cnn(x)
        vars = self.outVariance()
        return preds


if __name__ == '__main__':
    from MyTest.utils.util import get_config
    from MyTest.configs import config
    cfg = config.config
    model = CustomTimmModel(cfg)
    print(model)