import math
from pickle import FALSE
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
class PGD(nn.Module):
    def __init__(self, model,quant=False):
        super().__init__()
        self.model = model  # must be pytorch
        self.device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
        self.quant=quant #
    def generate(self, x, **params):
        self.parse_params(**params)
        labels = self.y
        labels=labels.to(self.device)   
        x=x.to(self.device)
        adv_x = self.attack(x, labels)
        return adv_x

    def parse_params(self, eps=8/255, iter_eps=2/255, nb_iter=4, clip_min=0.0, clip_max=1.0, C=0.0,
                     y=None, ord=np.inf, rand_init=True, flag_target=False):
        self.eps = eps#max eps
        self.iter_eps = iter_eps#step eps
        self.nb_iter = nb_iter#step nums
        self.clip_min = clip_min#clip num
        self.clip_max = clip_max
        self.y = y#labels
        self.ord = ord
        self.rand_init = rand_init
        #self.model.to(self.device)
        self.model.train()
        self.flag_target = flag_target
        self.C = C

    def sigle_step_attack(self, x, pertubation, labels):
        adv_x = x + pertubation
        adv_x = Variable(adv_x)
        adv_x.requires_grad = True#
        adv_x=adv_x.to(self.device)
        loss_func = nn.CrossEntropyLoss()
        preds = self.model(adv_x)
        
        if self.flag_target:
            loss = -loss_func(preds, labels)
        else:
            loss = loss_func(preds, labels)

        self.model.zero_grad()
        loss.backward()
        
        grad = adv_x.grad.detach()
        pertubation = (torch.tensor(self.iter_eps) * torch.sign(grad)).to(self.device)
        adv_x_new= adv_x.detach() + pertubation

        pertubation = torch.clip(adv_x_new, self.clip_min, self.clip_max) - x
        pertubation=torch.clip(pertubation,-self.eps,self.eps)
        
        
        return pertubation

    def attack(self, x, labels):
        if self.rand_init:
            x_tmp = x + torch.Tensor(np.random.uniform(-self.eps, self.eps, x.shape)).type_as(x).to(self.device)
        else:
            x_tmp = x
        pertubation = torch.zeros(x.shape).type_as(x).to(self.device)
        for i in range(self.nb_iter):
            pertubation = self.sigle_step_attack(x_tmp, pertubation=pertubation, labels=labels).to(self.device)
        adv_x = x + pertubation
        adv_x = torch.clip(adv_x, self.clip_min, self.clip_max)

        adv_x.to(self.device)
        return adv_x

