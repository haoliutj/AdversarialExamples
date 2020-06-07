"""
Adversarial training
"""

import copy
import numpy as np
import torch
import torch.nn as nn




def adv_train(X,y,model,adversary):
    "Adversarial training, returns pertubed mini batch"
    "delay: to decide how many epochs should be used as adv training"

    # if adversarial training, need a snapshot of the model at each batch to compute grad,
    # so as not to mess up with the optimization step
    model_cp = copy.deepcopy(model)
    for p in model_cp.parameters():
        p.requires_grad = False
    model_cp.eval()

    adversary.model = model_cp

    X_adv = adversary.perturbation(X,y)

    return X_adv