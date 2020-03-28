

def create_model_cifar(args, num_classes):
    """
    creates a deep neural network for cifar data
    :param args: args
    :type args:
    :param num_classes:
    :type num_classes:
    :return:
    :rtype:
    """
    # Create Model
    if args.model == "resnet":
        from .resnet import ResNet
        model = ResNet(num_classes=num_classes, depth=args.depth, norm_type=args.norm, basicblock=args.basicblock,
                       norm_power=args.norm_power)
    elif args.model == "vgg":
        from models.vgg import vgg

        model = vgg(args.depth, norm_type=args.norm, num_classes=num_classes, norm_power=args.norm_power)
    else:
        print("Model name is wrong")
        model = None

    return model





