# Towards an Adversarially Robust Normalization Approach
This is code of the paper "Towards an adversarially Robust Normalization Approach".
## Prerequisites
- [PyTorch 0.4.0+](https://pytorch.org/)
- Python 3.5+
- Cuda 8.0+

### File structure 
    .
    ├── models                  # containt different neural networks
    │    ├── create_model.py    # Select model
    │    ├── norm.py            # Different normalizations are defineded in it
    │    ├── resnet.py          # resnet for cifar is defined
    │    ├── resnet_imagenet.py # resnet for imagenet, pytorch torch.vision resnet is modidfied
    │    ├── vgg.py             # vgg for cifar 
    ├── utils                   # functions of many different types that are not related to main paper are defined here
    │    ├── data.py            # functions for dataloder for diffeerent datasets 
    ├── attacks.py              # function to select adversarial attacks from advertorch package 
    ├── functions.py            # functions to use in training and testing 
    ├── test_cifar.py
    ├── test_imagenet.py
    ├── train_cifar.py
    ├── train_imagenet.py
    ├── run.sh
    └── README.md

## Training Examples

Training ResNet-50 on CIFAR-100 using RobustNorm
```
python train_cifar.py --dataset cifar100 --depth 20 --norm RNT --checkpoint checkpoints/cifar100-resnet50-RNT
```
Testing 
```
python test_cifar.py --dataset cifar10 --model resnet --depth 20 --norm BN --resume checkpoints/Adverserial/cifar10-resnet20-BN/model_best.pth.tar --lr 0.1 --train-batch 128 --gpu 0
```

Train ResNet18 on ImageNet using RobustNorm
```
python3 train_imagenet.py --dataset imagenet --model resnet18 --norm RNT --gpu 0,1,2,3 --checkpoint checkpoints/imagenet/imagenet-resnet18-RN
```

### References
Some parts of this code are based on this [repository](https://github.com/hyeonseobnam/Batch-Instance-Normalization)  
