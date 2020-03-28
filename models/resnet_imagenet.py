import torch.nn as nn
from torchvision import models
try:
    from norm import select_norm
except:
    from .norm import select_norm

def change_norm(model, norm_type="BN", norm_power=0.2):
    """
    given a resnet model from torchvision.models, it changes its norm according to norm_type
    :param model: torchvision model
    :type model: pytorch class
    :param norm_type: name of the norm
    :type norm_type: str
    :param norm_power: hyper-parameter meant only for robust norm
    :type norm_power: float
    :return: model with orignal norm replaced with norm_type
    :rtype: pytorch model
    """

    # select norm to be used
    normlayer = select_norm(norm_type, norm_power=norm_power)
    # find total number of childern
    model_len = 0
    for n, child in enumerate(model.children()):
        model_len = n

    # for layer 0 which is outside
    conv_shape = model.conv1.out_channels
    model.bn1 = normlayer(conv_shape)
    # replace in all other layers
    for n, child in enumerate(model.children()):
        if 4 <= n <= model_len - 2:
            for i in range(len(child)):
                conv_shape = child[i].conv1.out_channels
                child[i].bn1 = normlayer(conv_shape)
                conv_shape = child[i].conv2.out_channels
                child[i].bn2 = normlayer(conv_shape)

                # if model have bn3 as well
                try:
                    conv_shape = child[i].conv3.out_channels
                    child[i].bn3 = normlayer(conv_shape)
                except:
                    pass
                try:
                    conv_shape = child[i].downsample[0].out_channels
                    child[i].downsample[1] = normlayer(conv_shape)
                    print("downsample")
                except:
                    print("no downsample")
    return model


def remove_tracking(model, norm_type, norm_power=0.2):
    """
    Given a pretrained resnet with tracked normalization, it changes normalization to non-tracked i.e. BN --> BNwoT
    :param model: model to be used
    :type model: pytorch class
    :param norm_power: hyper-parameter meant only for robust norm
    :type norm_power: float
    :return: pytorch model with tracking removed
    :rtype: pytorch class
    """
    normlayer = select_norm(norm_type, norm_power=norm_power)
    # find total number of childern
    model_len = 0
    for n, child in enumerate(model.children()):
        model_len = n

    # for layer 0 which is outside
    conv_shape = model.conv1.out_channels
    w = model.bn1.weight
    b = model.bn1.bias
    model.bn1 = normlayer(conv_shape)
    model.bn1.weight = w
    model.bn1.bias = b

    # replace in all other layers
    for n, child in enumerate(model.children()):
        if 4 <= n <= model_len - 2:
            for i in range(len(child)):
                conv_shape = child[i].conv1.out_channels
                w = child[i].bn1.weight
                b = child[i].bn1.bias
                child[i].bn1 = normlayer(conv_shape)
                child[i].bn1.weight = w
                child[i].bn1.bias = b

                conv_shape = child[i].conv2.out_channels
                w = child[i].bn2.weight
                b = child[i].bn2.bias
                child[i].bn2 = normlayer(conv_shape)
                child[i].bn2.weight = w
                child[i].bn2.bias = b
                # if model have bn3 as well
                try:
                    conv_shape = child[i].conv3.out_channels
                    w = child[i].bn3.weight
                    b = child[i].bn3.bias
                    child[i].bn3 = normlayer(conv_shape)
                    child[i].bn3.weight = w
                    child[i].bn3.bias = b
                except:
                    pass
                try:
                    conv_shape = child[i].downsample[0].out_channels
                    w = child[i].downsample[1].weight
                    b = child[i].downsample[1].bias
                    child[i].downsample[1] = normlayer(conv_shape)
                    child[i].downsample[1].weight = w
                    child[i].downsample[1].bias = b
                    print("downsample")
                except:
                    print("no downsample")

    return model


def resnet_imagenet(model_name, norm_type, tracking=True, norm_power=0.2, num_classes=1000, pretrained=False):
    """
    gives back resnet model with required normalization
    :param model_name: name of the model, resnet18, 34, 50 are allowed only
    :type model_name: str
    :param norm_type: name of the normalization to be used
    :type norm_type: str
    :param tracking: if tracking required
    :type tracking: bool
    :param norm_power: hyper-parameter meant only for robust norm
    :type norm_power: float
    :param num_classes: number of classes of the data
    :type num_classes: int
    :param pretrained: if use pretrained weights
    :type pretrained: bool
    :return: pytorch model
    :rtype: pytorch class
    """
    if model_name == "resnet18":
        model = models.resnet18(pretrained=pretrained)
    elif model_name == "resnet34":
        model = models.resnet34(pretrained=pretrained)
    elif model_name == "resnet50":
        model = models.resnet50(pretrained=pretrained)
    else:
        raise Exception("Only resnet18, 34 and 50 are allowed!")

    # if tracking is False:
    if norm_type == "BN":
        pass
    elif norm_type == "BNwoT":
        model = remove_tracking(model, norm_type=norm_type, norm_power=norm_power)
    else:
        if tracking is False:
            model = remove_tracking(model, norm_type=norm_type, norm_power=norm_power)
        else:
            model = change_norm(model, norm_type=norm_type, norm_power=norm_power)

    return model


if __name__ == '__main__':
    import torch
    a = resnet_imagenet("resnet50", "GN", tracking=True, pretrained=False)
    print(a)
    a = resnet_imagenet("resnet50", "RN", tracking=False, pretrained=False)
    img = torch.randn(2,3,224,224)
    print(a)
    print(a(img))