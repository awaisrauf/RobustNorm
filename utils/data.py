
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os


# loads cifar dataset
def load_cifar(dataset,train_batch, test_batch=100, workers=1):
    """
    loads cifar10 and cifar100 from pytorch datasets along with preprocessing
    :param dataset: cifar10 or cifar100
    :param train_batch: batchsize of trainsets
    :param test_batch: batchsize of testset
    :param workers:  number of workers for pytorch dataloading
    :return: train and test set loader as well as number of classes
    """
    assert dataset == 'cifar10' or dataset == 'cifar100', 'Dataset can only be cifar10 or cifar100'

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    # choose dataset
    if dataset == 'cifar10':
        dataloader = datasets.CIFAR10
        num_classes = 10
    elif dataset == 'cifar100':
        dataloader = datasets.CIFAR100
        num_classes = 100

    trainset = dataloader(root='./data', train=True, download=True, transform=transform_train)
    trainloader = data.DataLoader(trainset, batch_size=train_batch, shuffle=True, num_workers=workers)

    testset = dataloader(root='./data', train=False, download=False, transform=transform_test)
    testloader = data.DataLoader(testset, batch_size=test_batch, shuffle=False, num_workers=workers)

    return trainloader, testloader, num_classes



def load_imagenet(folder, train_batch, test_batch=128, workers=2):
    """
    loads imagnet dataset with all the preprocessing done
    :param folder: path where imagenet dataset is located
    :param train_batch: batch size of training set
    :param test_batch: batch size of test set
    :param workers: number of workers for pytorch data loading
    :return: trainloader,testloader and number of classes of dataset
    """
    # Data loading code
    traindir = os.path.join(folder, 'train')
    valdir = os.path.join(folder, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_data = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    val_data = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))
    image_datasets = {"train": train_data, "val": val_data}
    batch_sizes = {"train": train_batch, "val": test_batch}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_sizes[x],
                                                  shuffle=True, num_workers=workers, pin_memory=True) for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    num_classes = 1000
    trainloader = dataloaders["train"]
    testloader = dataloaders["val"]

    return trainloader, testloader, num_classes


    """
    load transfer leanring datasets such as birds or flowers dataset with all the preprocessing done
    :param dataset: which transfer learning dataset? birds or flowers
    :param folder: path where imagenet dataset is located
    :param train_batch: batch size of training set
    :param test_batch: batch size of test set
    :param workers: number of workers for pytorch data loading
    :return: trainloader,testloader and number of classes of dataset
    """
    # Data loading code
    traindir = os.path.join(folder, 'train')
    valdir = os.path.join(folder, 'test')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_data = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    val_data = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))
    image_datasets = {"train": train_data, "val": val_data}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=train_batch,
                                                  shuffle=True, num_workers=workers) for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    if dataset == "flowers":
        num_classes = 102
    elif dataset == "birds":
        num_classes = 200
    else:
        print("dataset not recognized")
    trainloader = dataloaders["train"]
    testloader = dataloaders["val"]

    return trainloader, testloader, num_classes

def load_mnist(dataset,train_batch, test_batch=128, workers=1):
    """
    loads mnist from pytorch datasets along with preprocessing
    :param dataset: mnist
    :param train_batch: batchsize of trainsets
    :param test_batch: batchsize of testset
    :param workers:  number of workers for pytorch dataloading
    :return: train and test set loader as well as number of classes
    """
    assert dataset == 'mnist' or dataset == 'fmnist', 'Dataset can only be cifar10 or cifar100'

    transform_train = transforms.Compose([
        transforms.RandomCrop(28),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.1307], std=[0.3081]),
    ])

    transform_test = transforms.Compose([
        transforms.RandomCrop(28),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.1307], std=[0.3081]),
    ])
    # choose dataset
    if dataset == 'mnist':
        dataloader = datasets.MNIST
        num_classes = 10
    elif dataset == 'fmnist':
        dataloader = datasets.FashionMNIST
        num_classes = 10

    trainset = dataloader(root='./data', train=True, download=True, transform=transform_train)
    trainloader = data.DataLoader(trainset, batch_size=train_batch, shuffle=True, num_workers=workers)

    testset = dataloader(root='./data', train=False, download=True, transform=transform_test)
    testloader = data.DataLoader(testset, batch_size=test_batch, shuffle=False, num_workers=workers)

    return trainloader, testloader, num_classes




def select_dataset(args):
    if args.dataset == "cifar10" or args.dataset == "cifar100":
        from utils.data import load_cifar
        trainloader, testloader, num_classes = load_cifar(args.dataset, args.train_batch, test_batch=args.test_batch)

    elif args.dataset == "imagenet":
        folder = "/data/imagenet"
        from utils.data import load_imagenet
        trainloader, testloader, num_classes = load_imagenet(folder, args.train_batch, test_batch=args.test_batch)

    elif args.dataset == "mnist" or args.dataset == "fmnist":
        from utils.data import load_mnist
        trainloader, testloader, num_classes = load_mnist(args.dataset, args.train_batch, test_batch=args.test_batch)

    else:
        print("select right dataset", args.dataset, " is not included.")

    return trainloader, testloader, num_classes