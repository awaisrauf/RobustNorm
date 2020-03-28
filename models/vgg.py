# Modified from https://github.com/pytorch/vision.git
import math
import torch.nn as nn

__all__ = [
    'vgg'
]


class VGGNetwork(nn.Module):
    # VGG model
    def __init__(self, features, num_classes=10):
        super(VGGNetwork, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, num_classes),
        )
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def make_layers(cfg, batch_norm=False, norm_type=None, norm_power=0.2, in_channels=3):
    layers = []

    from .norm import select_norm
    normlayer = select_norm(norm_type, norm_power=norm_power)

    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, normlayer(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


# ToDo: Check if we can change number of channels for it to use CIFAR
cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
          512, 512, 512, 512, 'M'],
}


def vgg(depth, norm_type=None, num_classes=10, norm_power=0.2, in_channels=3):
    """
    selects vgg network for based depth
    :param depth: number of layers in vgg, allowed depths 11, 16, 19
    :type depth: int
    :param norm_type: name of normalization you want to use
    :type norm_type: str
    :param num_classes: number of classes of data
    :type num_classes: int
    :param norm_power: hyper-parameter meant only for robust norm
    :type norm_power: float
    :param in_channels: number of channels of input image
    :type in_channels: int
    :return: vgg network as function
    :rtype: python function
    """
    if depth == 11:
        return VGGNetwork(make_layers(cfg['A'], batch_norm=True, norm_type=norm_type, norm_power=norm_power, in_channels=in_channels), num_classes=num_classes)
    elif depth == 16:
        # VGG 13-layer model (configuration "B") with batch normalization
        return VGGNetwork(make_layers(cfg['D'], batch_norm=True, norm_type=norm_type, norm_power=norm_power, in_channels=in_channels), num_classes=num_classes)

    elif depth == 19:
        # VGG 16-layer model (configuration "D") with batch normalization
        return VGGNetwork(make_layers(cfg['D'], batch_norm=True, norm_type=norm_type, norm_power=norm_power, in_channels=in_channels), num_classes=num_classes)

    # elif depth == 19 and config == 'E':
    #     # VGG 19-layer model (configuration 'E') with batch normalization"""
    #     return VGG(make_layers(cfg['E'], batch_norm=True))
    else:
        raise Exception('Depth can only be 11, 16 or 19')

