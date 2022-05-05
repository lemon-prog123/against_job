import os
import math
from re import L
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Function
from torch.autograd import Variable
from torch import  Tensor
class Conv_re(nn.Conv2d):
    def __init__(self,*args, **kwargs):
        super(Conv_re, self).__init__(*args, **kwargs)
        N,C,H,W=self.weight.shape
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.v = torch.empty(C*H*W,1).to(device)
        nn.init.uniform_(self.v, -0.1, 0.1)
class Linear_re(nn.Linear):
     def __init__(self,*args,**kwargs):
        super(Linear_re, self).__init__(*args, **kwargs)
        H,W=self.weight.shape
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.v = torch.empty(W,1).to(device)
        nn.init.uniform_(self.v, -0.1, 0.1)

class Conv(nn.Conv2d):
    def __init__(self,*args, **kwargs):
        super(Conv, self).__init__(*args, **kwargs)
        N,C,H,W=self.weight.shape
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.v = torch.empty(C*H*W,1).to(device)
        nn.init.uniform_(self.v, -0.3, 0.3)
    def reconstruct(self):
        self.weight.data=self.orignal_weight
    def quant2(self,x):
        x_new=torch.abs(x.detach())
        max_v=torch.max(x_new)
        scale=max_v/torch.tensor(127)
        input=x.detach()/scale
        input=torch.round(input)
        input=torch.clip(input,-128,127)
        return input,scale
    def forward(self, x):
        x_new,scale1=self.quant2(x)
        x.data=x_new*scale1
        self.orignal_weight=self.weight.data
        weight_new,scale2=self.quant2(self.weight)
        self.weight.data=weight_new*scale2
        output=self._conv_forward(x, self.weight, self.bias)
        self.weight.data=self.orignal_weight
        return output

class Linear(nn.Linear):
    def __init__(self,*args,**kwargs):
        super(Linear, self).__init__(*args, **kwargs)
        H,W=self.weight.shape
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.v = torch.empty(W,1).to(device)
        nn.init.uniform_(self.v, -0.3, 0.3)
    def reconstruct(self):
        self.weight.data=self.orignal_weight
    def quant2(self,x):
        x_new=torch.abs(x.detach())
        max_v=torch.max(x_new)
        scale=max_v/torch.tensor(127)
        input=x.detach()/scale
        input=torch.round(input)
        input=torch.clip(input,-128,127)
        return input,scale
    def forward(self, x):
        
        x_new,scale1=self.quant2(x)
        x.data=x_new*scale1
        self.orignal_weight=self.weight.data
        weight_new,scale2=self.quant2(self.weight)
        self.weight.data=weight_new*scale2
        output=F.linear(x, self.weight, self.bias)
        self.weight.data=self.orignal_weight
        
        return output
