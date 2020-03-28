import os
import shutil
import time
import torch
import torch.nn.parallel
import torch.optim as optim
from utils import Bar, AverageMeter, accuracy
from PIL import ImageFile
import utils
import numpy as np

ImageFile.LOAD_TRUNCATED_IMAGES = True


def train_one_epoch(trainloader, model, criterion, optimizer, use_cuda=True, test_flag=False):
    """
    trains the model on trainloader data for one epoch
    :param trainloader: pytroch dataloader that gives one batch of data at each iteration
    :type trainloader: pytorch dataloader
    :param model: model to be trained
    :type model: pytorch model class
    :param criterion: loss function defined in pytorch
    :type criterion: pytorch class
    :param optimizer: gradient descent based optimzer to update model weights
    :type optimizer: pytorch class
    :param use_cuda: flag to set to use cuda or not
    :type use_cuda: boolean
    :param test_flag: flag to on the test mode, it will run function on one epoch only
    :type test_flag: boolean
    :return: average values of loss, top1 accuracy top5 accuracy
    :rtype: floats
    """
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(trainloader))
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # for test case only
        if test_flag is True and batch_idx > 1:
            break
        # measure data loading time
        data_time.update(time.time() - end)
        if use_cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()

        # compute gradient and do SGD step
        optimizer.zero_grad()
        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))
        #
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # plot progress
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: ' \
                     '{loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(batch=batch_idx + 1,
                                                                                 size=len(trainloader),
                                                                                 data=data_time.avg,bt=batch_time.avg,
                                                                                 total=bar.elapsed_td, eta=bar.eta_td,
                                                                                 loss=losses.avg, top1=top1.avg,
                                                                                 top5=top5.avg,)
        bar.next()
    bar.finish()
    return losses.avg, top1.avg, top5.avg


# based on this implementation: https://github.com/MadryLab/mnist_challenge/blob/master/train.py
def train_adversary_one_epoch(trainloader, model, criterion, optimizer, use_cuda=True, epsilon=0.03, attack="LinfPGD",
                              num_classes=10, test_flag=False):
    """
    trains the model on trainloader data for one epoch
    :param trainloader: pytroch dataloader that gives one batch of data at each iteration
    :type trainloader: pytorch dataloader
    :param model: model to be trained
    :type model: pytorch model class
    :param criterion: loss function defined in pytorch
    :type criterion: pytorch class
    :param optimizer: gradient descent based optimzer to update model weights
    :type optimizer: pytorch class
    :param use_cuda: flag to set to use cuda or not
    :type use_cuda: boolean
    :param epsilon: radius of adversarial attack to be used for adversarial training
    :type epsilon: float
    :param attack: name of attack to be used for adversarial training, either PGD or FGSM based (from ICLR paper)
    :type attack: string
    :param num_classes: number of classes of input data
    :type num_classes: int
    :param test_flag: flag to on the test mode, it will run function on one epoch only
    :type test_flag: boolean
    :return: average values of loss, top1 accuracy top5 accuracy
    :rtype: floats
    """
    assert attack == 'LinfPGD' or attack == 'FGSM_Adv', 'Adversarial Training is only possible for LinfPGD or FGSM_Adv'
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(trainloader))
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # for test case only
        if test_flag is True and batch_idx > 1:
            break
        # measure data loading time
        data_time.update(time.time() - end)
        if use_cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()

        # add attack
        eps_iter = epsilon/4 # 4 is from madry et al's implementation: https://github.com/MadryLab/cifar10_challenge/blob/ecc870b3576beb010330324e690d2a6b48674fba/config.json#L28
        from attacks import select_attack
        inputs_adv = select_attack(inputs, targets, attack, model, epsilon, eps_iter,
                                        nb_iters=10, num_classes=num_classes, targeted=False)
        # update with adv images
        optimizer.zero_grad()
        # compute output
        outputs = model(inputs_adv)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))
        # backward pass
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # plot progress
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: ' \
                     '{loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(batch=batch_idx + 1,
                                                                                 size=len(trainloader),
                                                                                 data=data_time.avg,bt=batch_time.avg,
                                                                                 total=bar.elapsed_td, eta=bar.eta_td,
                                                                                 loss=losses.avg, top1=top1.avg,
                                                                                 top5=top5.avg,)
        bar.next()
    bar.finish()
    return losses.avg, top1.avg, top5.avg


def test(testloader, model, criterion, use_cuda=True, test_flag=False):
    """
    checks performance of the model
    :param testloader: pytroch dataloader that gives one batch of data at each iteration
    :type testloader: pytorch dataloader
    :param model: model to be trained
    :type model: pytorch model class
    :param criterion: loss function defined in pytorch
    :type criterion: pytorch class
    :param use_cuda: whether to put inputs, targets on cuda or not
    :type use_cuda: boolean
    :param test_flag: flag to on the test mode, it will run function on one epoch only
    :type test_flag: boolean
    :return: average values of loss, top1 accuracy top5 accuracy
    :rtype: floats
    """
    # initialization of telemetery
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    # testing
    model.eval()   # switch to test mode
    bar = Bar('Processing', max=len(testloader))
    for batch_idx, (inputs, targets) in enumerate(testloader):
        # for test case only
        if test_flag and batch_idx > 1:
            break

        data_time.update(time.time() - end)
        if use_cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()

        # compute output
        with torch.no_grad():
            outputs = model(inputs)
        loss = criterion(outputs, targets)
        # measure accuracy and record loss
        prec1, prec5 = utils.accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} |' \
                      ' Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                                                                                                batch=batch_idx + 1,
                                                                                                size=len(testloader),
                                                                                                data=data_time.avg,
                                                                                                bt=batch_time.avg,
                                                                                                total=bar.elapsed_td,
                                                                                                eta=bar.eta_td,
                                                                                                loss=losses.avg,
                                                                                                top1=top1.avg,
                                                                                                top5=top5.avg)
        bar.next()
    bar.finish()

    return losses.avg, top1.avg, top5.avg


def test_adversary(testloader, model, criterion, use_cuda=True,  attack="FGSM", num_classes=10, epsilon=0.03, c=0.1,
                   nb_iters=20, targeted=False, test_flag=False):
    """
    test model on adversarially attacked images of attack type adv. noise
    :param testloader: pytroch dataloader that gives one batch of data at each iteration
    :type testloader: pytorch dataloader
    :param model: model to be trained
    :type model: pytorch model class
    :param criterion: loss function defined in pytorch
    :type criterion: pytorch class
    :param use_cuda: whether to put inputs, targets on cuda or not
    :type use_cuda: boolean
    :param attack: name of the adversarial attack to perturb inputs
    :type attack: str
    :param num_classes: number of classes of data
    :type num_classes: int
    :param epsilon: power of adv. noise |clean_img - noisy_img| < epsilon
    :type epsilon: float
    :param c: init const for carlini winger attack
    :type c: init const for carlini winger attack
    :param nb_iters: number of iterations in iterative attacks
    :type nb_iters: int
    :param targeted: if attack is targeted or not
    :type targeted: boolean
    :param test_flag: flag to on the test mode, it will run function on one epoch only
    :type test_flag: boolean
    :return: average values of loss, top1 accuracy top5 accuracy
    :rtype: float, float, float
    """
    from attacks import select_attack

    # initialize telemetery
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    # start test
    model.eval()
    print(epsilon)
    bar = Bar('Processing', max=len(testloader))
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if test_flag is True and batch_idx > 5:
            break

        # measure data loading time
        data_time.update(time.time() - end)
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        # add attack
        eps_iter = epsilon/10
        inputs = select_attack(inputs, targets, attack, model, epsilon, eps_iter,
                            nb_iters=nb_iters, num_classes=num_classes, c=c, targeted=targeted)

        # compute output
        with torch.no_grad():
            outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} ' \
                     '| Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                      batch=batch_idx + 1,
                      size=len(testloader),
                      data=data_time.avg,
                      bt=batch_time.avg,
                      total=bar.elapsed_td,
                      eta=bar.eta_td,
                      loss=losses.avg,
                      top1=top1.avg,
                      top5=top5.avg,)
        bar.next()
    bar.finish()

    return losses.avg, top1.avg, top5.avg


def adversary_test_attacks(model, criterion, testloader, num_classes, attacks, epsilon=0.008, targeted=False,
                         cw_initial_const=0.1, use_cuda=True, test_flag=False):
    """
    gives result for many attacks as dictionary, its a wrapper around test_adv and test_adv_blackbox
    :param model: pyotrch model for forward pass
    :type model: pytorch class
    :param criterion: loss function
    :type criterion: pytorch class
    :param testloader: data loader from whom to get acc
    :type testloader: pytorch dataloader
    :param num_classes: number of classes of input data
    :type num_classes: int
    :param attacks: attacks for which to get adv accuracy
    :type attacks: list
    :param epsilon: power of each attack
    :type epsilon: float
    :param targeted: if targeted attack or not
    :type targeted: bool
    :param cw_initial_const: initial constant for carlini winger attack
    :type cw_initial_const: float
    :param use_cuda: wheter to use cuda or not
    :type use_cuda: bool
    :param test_flag: used when we only testing
    :type test_flag: bool
    :return: dictionaries containing accuracy1, accuracy5 and loss when attacked with attacks
    :rtype: dict
    """

    attacks_loss, attacks_acc, attacks_acc5 = {}, {}, {}  # initialize dictionaries

    print('\nAdversary Testing')
    for attack in attacks:
        print(attack, "with epsilon =", epsilon)
        loss, top1_acc, top5_acc = test_adversary(testloader, model, criterion, use_cuda, num_classes=num_classes,
                                                  attack=attack, epsilon=epsilon, c=cw_initial_const,
                                                  targeted=targeted, test_flag=test_flag)
        attacks_loss[attack], attacks_acc[attack], attacks_acc5[attack] = loss, top1_acc, top5_acc

    return attacks_loss, attacks_acc, attacks_acc5


def set_optimizer(model, args):
    """
    sets optimizer according to ars
    :param model: deep learning model
    :type model: pytroch class
    :param args: arguments passed
    :type args:
    :return:
    :rtype:
    """
    params = [{'params': [p for p in model.parameters() if not getattr(p, 'bin_gate', False)]},
              {'params': [p for p in model.parameters() if getattr(p, 'bin_gate', False)],
               'lr': args.lr * args.bin_lr, 'weight_decay': 0}]
    optimizer = optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    return optimizer


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


def adjust_learning_rate(state, optimizer, epoch, gamma, schedule):
    if epoch in schedule:
        state *= gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] *= gamma
    return state

