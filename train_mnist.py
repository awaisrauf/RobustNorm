import os
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from utils import Logger, mkdir_p
from functions import train_one_epoch, train_adversary_one_epoch, test, set_optimizer, save_checkpoint,\
    adjust_learning_rate
from utils.utils import args_for_train_mnist
from models.cnn_mnist import ResNet_MNIST
from utils.data import load_mnist


# ............. Basics .....................
# get all the input variables
args = args_for_train_mnist()
state = {k: v for k, v in args._get_kwargs()}
# Validate dataset name
assert args.dataset == 'mnist' or args.dataset == 'fmnist', 'Dataset can only be mnist or fmnist'
# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()
# set parameters
best_acc = 0  # best test accuracy
do_save_checkpoint = True
start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch
if not os.path.isdir(args.checkpoint):
    mkdir_p(args.checkpoint)

# ............. Load Data ...................
print('==> Preparing dataset %s' % args.dataset)
trainloader, testloader, num_classes = load_mnist(args.dataset, args.train_batch, test_batch=args.test_batch)

# ............. Create Model ................
# create model
model = ResNet_MNIST(args.depth, norm_type=args.norm)
# set model and its secondaries
model = model.cuda()
model = torch.nn.DataParallel(model)
cudnn.benchmark = True
print('    Total params: %.4fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
criterion = nn.CrossEntropyLoss()
optimizer = set_optimizer(model, args)

title = '{}-{}'.format(args.dataset, args.norm)
# Resume
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
    args.checkpoint = os.path.dirname(args.resume)
    checkpoint = torch.load(args.resume)
    best_acc = checkpoint['best_acc']
    if args.start_epoch==0:
        start_epoch = checkpoint['epoch']
    else:
        start_epoch = args.start_epoch
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
else:
    logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
    logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss','Train Acc.', 'Valid Acc.', 'Train Acc.5',
                      'Valid Acc.5'])

# .................. Train and validate ...............
for epoch in range(start_epoch, args.epochs):
    state['lr'] = adjust_learning_rate(state['lr'], optimizer, epoch, args.gamma, [51])
    print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

    if args.train_mode == "Normal":
        train_loss, train_acc, train_acc5 = train_one_epoch(trainloader, model, criterion, optimizer, use_cuda)
        test_loss, test_acc, test_acc5 = test(testloader, model, criterion, use_cuda)
        # append logger file
        logger.append([state['lr'], train_loss, test_loss, train_acc, test_acc, train_acc5, test_acc5])
    elif args.train_mode == "AdversaryPGD":
        adv_train_attack = "LinfPGD" if args.train_mode == "AdversaryPGD" else "None"
        eps = 0.03
        train_loss, train_acc, train_acc5 = train_adversary_one_epoch(trainloader, model, criterion, optimizer,
                                                                      use_cuda,
                                                                      attack=adv_train_attack, num_classes=num_classes)
        test_loss, test_acc, test_acc5 = test(testloader, model, criterion, use_cuda)
        # append logger file
        logger.append([state['lr'], train_loss, test_loss, train_acc, test_acc, train_acc5, test_acc5])
    else:
        print("Train mode is not defined!")

    # save model ap
    is_best = test_acc > best_acc
    best_acc = max(test_acc, best_acc)
    if do_save_checkpoint:
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }, is_best, checkpoint=args.checkpoint)

logger.close()
print('Best acc:', best_acc)

