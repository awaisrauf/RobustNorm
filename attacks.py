from advertorch.attacks import LinfPGDAttack, GradientAttack, GradientSignAttack, \
    CarliniWagnerL2Attack, L2PGDAttack, LinfBasicIterativeAttack, MomentumIterativeAttack, FGSM
import torch.nn as nn
import torch
from advertorch.utils import clamp
import numpy as np

def select_attack(inputs, targets, attack, model, epsilon, eps_iter, nb_iters=20, num_classes=10, c=0.1, targeted=False):
    """
    add adversarial noise in given inputs with attack adv. method.
    :param inputs: batch of inputs
    :type inputs: pytorch tensor
    :param targets: batch of outputs
    :type targets: pytorch tensor
    :param attack: attack method
    :type attack: str
    :param model: model to be used to craft attack
    :type model: class
    :param epsilon: power of adv noise
    :type epsilon: float
    :param eps_iter: for itertiave attacks, to update at each epoch
    :type eps_iter: float
    :param nb_iters: number of iterations for iterative attacks
    :type nb_iters: int
    :param num_classes: number of classification classes
    :type num_classes: int
    :param c: parameter for cw attack
    :type c: float
    :param targeted: whether attack be targeted or not
    :type targeted: bool
    :return: noisy inputs
    :rtype: pytorch tensors
    """
    clip_min, clip_max = torch.min(inputs), torch.max(inputs)

    if attack == "Grad":
        adversary = GradientAttack(
            model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=epsilon, clip_min=clip_min, clip_max=clip_max,
            targeted=targeted)
        inputs = adversary.perturb(inputs, targets)

    elif attack == "FGSM":
        adversary = FGSM( model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=epsilon,
                                        clip_min=clip_min, clip_max=clip_max, targeted=targeted)
        inputs = adversary.perturb(inputs, targets)
    elif attack == "LinfBIM":
        adversary = LinfBasicIterativeAttack(model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=epsilon,
                                             nb_iter=nb_iters, eps_iter=eps_iter, clip_min=clip_min, clip_max=clip_max,
                                             targeted=targeted)
        inputs = adversary.perturb(inputs, targets)
    elif attack == "L2PGD":
        adversary = L2PGDAttack(model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=epsilon, nb_iter=nb_iters,
                                eps_iter=eps_iter, rand_init=True, clip_min=clip_min, clip_max=clip_max, targeted=targeted)
        inputs = adversary.perturb(inputs, targets)
    elif attack == "LinfPGD":
        adversary = LinfPGDAttack(
            model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=epsilon,
            nb_iter=nb_iters, eps_iter=eps_iter, rand_init=True, clip_min=clip_min, clip_max=clip_max, targeted=targeted)
        inputs = adversary.perturb(inputs, targets)
    elif attack == "CWL2":
        adversary = CarliniWagnerL2Attack(model, num_classes, confidence=0, targeted=targeted, learning_rate=eps_iter,
                                          binary_search_steps=3, max_iterations=10, abort_early=True, initial_const=c,
                                          clip_min=clip_min, clip_max=clip_max, loss_fn=None)
        inputs = adversary.perturb(inputs, targets)
    elif attack == "MIM":
        adversary = MomentumIterativeAttack(model, loss_fn=None, eps=epsilon, nb_iter=nb_iters, decay_factor=1.0,
                                            eps_iter=eps_iter, clip_min=clip_min, clip_max=clip_max, targeted=targeted)
        inputs = adversary.perturb(inputs, targets)
    elif attack == "FGSM_Adv":
        adversary = Fgsm4AdvTraining( model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=epsilon,
                                        clip_min=clip_min, clip_max=clip_max, targeted=targeted)
        inputs = adversary.perturb(inputs, targets)
    elif attack == "Noise":
        inputs = inputs + 0.1 * torch.randn(inputs.size()).cuda()

    else:
        raise Exception('Attack is not supported. Kindly see attacks.py file.')

    return inputs
    