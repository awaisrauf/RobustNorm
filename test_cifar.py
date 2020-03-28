import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import os
import csv
import copy
from models.create_model import create_model_cifar
from functions import test, adversary_test_attacks
from utils.utils import args_for_test_cifar


# ............. Get Basics ..................
# Get all the arguments
args = args_for_test_cifar()
state = {k: v for k, v in args._get_kwargs()}
# Validate dataset
# assert args.dataset == 'cifar10' or args.dataset == 'cifar100', 'Dataset can only be cifar10 or cifar100.'
# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

# ............. Load Data ...................
print('==> Preparing dataset %s' % args.dataset)
from utils.data import select_dataset
trainloader, testloader, num_classes = select_dataset(args)
# testloader = trainloader

# ......... create and load main model .........
title = '{}-{}-{}-{}'.format(args.dataset,args.model, args.depth, args.norm)
print("==> creating model: ", title)
# create model
model = create_model_cifar(args, num_classes)

print('    Total params: %.4fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
# set model and its secondaries
model = model.cuda()
model = torch.nn.DataParallel(model)
cudnn.benchmark = True
criterion = nn.CrossEntropyLoss()  # loss function
# Load checkpoint.
print('==> Loading checkpoint')
assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
args.checkpoint = os.path.dirname(args.resume)
checkpoint = torch.load(args.resume)
model.load_state_dict(checkpoint['state_dict'])

# ............ find results ................
epsilons = args.epsilons
targeted = args.targeted
resume_path = args.resume
file_path = args.output_file_path

loss_normal, top1_acc_normal, _ = test(testloader, model, criterion, use_cuda, test_flag=False)
for epsilon in epsilons:
    cw_initial_const = 0.1
    attacks = ["FGSM", "LinfBIM", "MIM", "CWL2", "LinfPGD"]
    attacks_loss, attacks_acc, _ = adversary_test_attacks(model, criterion, testloader, num_classes, attacks,
                                                          epsilon=epsilon, targeted=targeted, cw_initial_const=0.1,
                                                          use_cuda=True, test_flag=False)

# ........... save results to csv ................
    csv_column_names = ["Model", "Targeted", "Training", "Dataset", "Norm" ,"Epsilon", "Normal", "FGSM","LinfBIM", "MIM", "CWL2",
                        "LinfPGD"]
    # if file does not exists, create it with header
    if not os.path.exists(file_path):
        print("path not exists!")
        with open(file_path, 'w', newline='') as outcsv:
            writer = csv.writer(outcsv)
            writer.writerow(csv_column_names)

    type = resume_path.split("/")[-2].split("-")[-1]
    attack_names = ["Normal", "FGSM", "LinfBIM", "MIM", "CWL2", "LinfPGD"]

    # convert dictionary to list to be fed to csv
    acc = [args.model+str(args.depth), args.targeted, type, args.dataset, args.norm, epsilon, top1_acc_normal]
    for attack_name in attacks:
        # loss.append(attacks_loss[attack_name])
        acc.append(attacks_acc[attack_name])

    assert(len(csv_column_names) == len(acc))
    # save it in csv
    with open(file_path, 'a', newline='') as outcsv:
        writer = csv.writer(outcsv)
        # writer.writerow(loss)
        writer.writerow(acc)


