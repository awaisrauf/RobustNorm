
import argparse


def args_for_train_cifar(argv=None):
    """
    get all the input arguemnts from commandline
    :param argv:
    :return: args that have all the arguemtns
    """
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')

    # Datasets
    parser.add_argument('-d', '--dataset', default='cifar10', type=str,
                        help='name of dataset {cifar10, cifar100}')
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers')
    # Architecture
    parser.add_argument('--depth', default=20, type=int,
                        help='model depth')
    parser.add_argument('--norm', default='BN', type=str,
                        help='normalization type {BN, RN, BNwoT}')
    parser.add_argument('--norm_power', type=float, default=0.2,
                        help='Hyperparameter of AvgNorm')
    parser.add_argument('--model', default='resnet', type=str,
                        help='model type {resnet, vgg}')
    parser.add_argument('--basicblock', action='store_true', default=False,
                        help='force to use basicblock')
    # Optimization options
    parser.add_argument('--epochs', default=164, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                        help='train batchsize')
    parser.add_argument('--test-batch', default=128, type=int, metavar='N',
                        help='test batchsize')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR',
                        help='initial learning rate')
    parser.add_argument('--schedule', type=int, nargs='+', default=[80, 120],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='LR is multiplied by gamma on schedule.')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W',
                        help='weight decay')
    parser.add_argument('--bin-lr', default=10.0, type=float, metavar='M',
                        help='lr mutiplier for BIN gates')
    # Checkpoints
    parser.add_argument('-c', '--checkpoint', default='checkpoints', type=str, metavar='PATH',
                        help='path to save checkpoint')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint')
    # Misc
    parser.add_argument('--train_mode', type=str,default='Normal',
                        help='manual seed')
    # Device options
    parser.add_argument('-g', '--gpu-id', default='0', type=str,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--adver_path', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint')
    args = parser.parse_args()
    return args


def args_for_test_cifar(argv=None):
    """
    get all the input arguemnts from commandline
    :param argv:
    :return: args that have all the arguemtns
    """
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')

    # Datasets
    parser.add_argument('-d', '--dataset', default='cifar10', type=str,
                        help='name of dataset {cifar10, cifar100}')
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers')
    # Architecture
    parser.add_argument('--depth', default=20, type=int,
                        help='model depth')
    parser.add_argument('--norm', default='bn', type=str,
                        help='normalization type {bn, in, bin}')
    parser.add_argument('--norm_power', type=float, default=0.2,
                        help='Hyperparameter of AvgNorm')
    parser.add_argument('--model', default='resnet', type=str,
                        help='model type {resnet, vgg}')
    parser.add_argument('--basicblock', action='store_true', default=False,
                        help='force to use basicblock')
    # Optimization options
    parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                        help='train batchsize')
    parser.add_argument('--test-batch', default=128, type=int, metavar='N',
                        help='test batchsize')
    # Checkpoints
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint')
    parser.add_argument('--output_file_path', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint')
    # Miscs
    parser.add_argument('--epsilons', type=float, nargs='+', default=[0.004, 0.008],
                        help='')
    parser.add_argument('--targeted', dest='targeted', action='store_true')
    parser.add_argument('--not-targeted', dest='targeted', action='store_false')
    # Device options
    parser.add_argument('-g', '--gpu-id', default='0', type=str,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--adver_path', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint')
    args = parser.parse_args()
    return args


def args_for_train_imegenet(argv=None):
    """
    get all the input arguemnts from commandline
    :param argv:
    :return: args that have all the arguments
    """
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

    # Datasets
    parser.add_argument('-d', '--dataset', default='imagenet', type=str,
                        help='name of dataset {imagenet}')
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers')
    # Architecture
    parser.add_argument('--depth', default=18, type=int,
                        help='model depth')
    parser.add_argument('--norm', default='bn', type=str,
                        help='normalization type {bn, in, bin}')
    parser.add_argument('--norm_power', type=float, default=0.2,
                        help='Hyperparameter of AvgNorm')
    parser.add_argument('--model', default='resnet', type=str,
                        help='model type {resnet, vgg}')
    parser.add_argument('--basicblock', action='store_true', default=False,
                        help='force to use basicblock')
    # Optimization options
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                        help='train batchsize')
    parser.add_argument('--test-batch', default=128, type=int, metavar='N',
                        help='test batchsize')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR',
                        help='initial learning rate')
    parser.add_argument('--schedule', type=int, nargs='+', default=[30,60,90],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='LR is multiplied by gamma on schedule.')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W',
                        help='weight decay')
    parser.add_argument('--bin-lr', default=10.0, type=float, metavar='M',
                        help='lr mutiplier for BIN gates')
    # Checkpoints
    parser.add_argument('-c', '--checkpoint', default='checkpoints', type=str, metavar='PATH',
                        help='path to save checkpoint')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint')
    # Miscs
    parser.add_argument('--train_mode', type=str,default='Normal',
                        help='manual seed')
    # Device options
    parser.add_argument('-g', '--gpu-id', default='0', type=str,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--adver_path', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint')
    args = parser.parse_args()
    return args


def args_for_test_imagenet(argv=None):
    """
    get all the input arguemnts from commandline
    :param argv:
    :return: args that have all the arguemtns
    """
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')

    # Datasets
    parser.add_argument('-d', '--dataset', default='cifar10', type=str,
                        help='name of dataset {cifar10, cifar100}')
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers')
    # Architecture
    parser.add_argument('--depth', default=20, type=int,
                        help='model depth')
    parser.add_argument('--norm', default='bn', type=str,
                        help='normalization type {bn, in, bin}')
    parser.add_argument('--norm_power', type=float, default=0.2,
                        help='Hyperparameter of AvgNorm')
    parser.add_argument('--model', default='resnet', type=str,
                        help='model type {resnet, vgg}')
    parser.add_argument('--basicblock', action='store_true', default=False,
                        help='force to use basicblock')
    # Optimization options
    parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                        help='train batchsize')
    parser.add_argument('--test-batch', default=128, type=int, metavar='N',
                        help='test batchsize')
    # Checkpoints
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint')
    parser.add_argument('--output_file_path', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint')
    # Miscs
    parser.add_argument('--epsilons', type=float, nargs='+', default=[0.004, 0.008],
                        help='')
    parser.add_argument('--targeted', dest='targeted', action='store_true')
    parser.add_argument('--not-targeted', dest='targeted', action='store_false')
    parser.add_argument('--tracking', type=str, default='True',
                        help='')
    # Device options
    parser.add_argument('-g', '--gpu-id', default='0', type=str,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--adver_path', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint')
    args = parser.parse_args()
    return args



def args_for_train_mnist(argv=None):
    """
    get all the input arguemnts from commandline
    :param argv:
    :return: args that have all the arguemtns
    """
    parser = argparse.ArgumentParser(description='PyTorch mnist/fmnist Training')

    # Datasets
    parser.add_argument('-d', '--dataset', default='mnist', type=str,
                        help='name of dataset {mnist, fmnist}')
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers')
    # Architecture
    parser.add_argument('--model', default='resnet', type=str,
                        help='model')
    parser.add_argument('--depth', default=8, type=int,
                        help='model depth')
    parser.add_argument('--norm', default='bn', type=str,
                        help='normalization type {bn, in, bin}')
    parser.add_argument('--norm_power', type=float, default=0.2,
                        help='Hyperparameter of AvgNorm')
    # Optimization options
    parser.add_argument('--epochs', default=50, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                        help='train batchsize')
    parser.add_argument('--test-batch', default=128, type=int, metavar='N',
                        help='test batchsize')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR',
                        help='initial learning rate')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='LR is multiplied by gamma on schedule.')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W',
                        help='weight decay')
    parser.add_argument('--bin-lr', default=10.0, type=float, metavar='M',
                        help='lr mutiplier for BIN gates')
    # Checkpoints
    parser.add_argument('-c', '--checkpoint', default='checkpoints', type=str, metavar='PATH',
                        help='path to save checkpoint')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint')
    # Miscs
    parser.add_argument('--train_mode', type=str,default='Normal',
                        help='manual seed')
    # Device options
    parser.add_argument('-g', '--gpu-id', default='0', type=str,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--adver_path', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint')
    args = parser.parse_args()
    return args
