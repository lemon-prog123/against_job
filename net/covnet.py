import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from pwla_relu.P_relu import PWLA3d
from pwla_relu.P_relu import PWLA1d
from util.utils import setparams
from net.conv_and_linear import*
from util.squeeze import *
class ConvNet(nn.Module):
    def __init__(self, quant=False,spectral=False):
        super(ConvNet, self).__init__()
        if quant==True:
            self.conv1 = Conv(3, 15,3)    
            self.conv2 = Conv(15, 75,4)   
            self.conv3 = Conv(75,375,3)    
            self.fc1 = Linear(1500,400)    
            self.fc2 = Linear(400,120)     
            self.fc3 = Linear(120, 84)      
            self.fc4 = Linear(84, 10)
        elif spectral==True:
            self.conv1 = Conv_re(3, 15,3)    
            self.conv2 = Conv_re(15, 75,4)   
            self.conv3 = Conv_re(75,375,3)    
            self.fc1 = Linear_re(1500,400)    
            self.fc2 = Linear_re(400,120)     
            self.fc3 = Linear_re(120, 84)      
            self.fc4 = Linear_re(84, 10)
        else:
            self.conv1 = nn.Conv2d(3, 15,3)    
            self.conv2 = nn.Conv2d(15, 75,4) 
            self.conv3 = nn.Conv2d(75,375,3)    
            self.fc1 =nn.Linear(1500,400)    
            self.fc2 = nn.Linear(400,120)        
            self.fc3 = nn.Linear(120, 84)       
            self.fc4 = nn.Linear(84, 10)
        self.relu1=PWLA3d()
        self.relu2=PWLA3d()
        self.relu3=PWLA3d()
        self.relu4=PWLA1d()
        self.relu5=PWLA1d()
        self.relu6=PWLA1d()
        self.mode=0
        self.device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    def activation(self):
        self.mode=1
        setparams(self.relu1)
        setparams(self.relu2)
        setparams(self.relu3)
        setparams(self.relu4)
        setparams(self.relu5)
        setparams(self.relu6)
    def reconstruct(self):
        self.conv1.reconstruct()
        self.conv2.reconstruct()
        self.conv3.reconstruct()
        self.fc1.reconstruct()
        self.fc2.reconstruct()
        self.fc3.reconstruct()
        self.fc4.reconstruct()
    def forward(self, x):
        """
        """
            
        x = F.max_pool2d(self.relu1(self.conv1(x),self.mode), 2)  # 3*32*32  -> 150*30*30  -> 15*15*15
        x = F.max_pool2d(self.relu2(self.conv2(x),self.mode), 2)  # 15*15*15 -> 75*12*12  -> 75*6*6
        x = F.max_pool2d(self.relu3(self.conv3(x),self.mode), 2)  # 75*6*6   -> 375*4*4   -> 375*2*2
        x = x.view(x.size()[0], -1)
        x = self.relu4(self.fc1(x),self.mode)
        x = self.relu5(self.fc2(x),self.mode)
        x = self.relu6(self.fc3(x),self.mode)
        x = self.fc4(x)
        return x

    # TODO:
    def regularizationTerm(self,beta1=1e-4,beta2=0.05,mode=0):
        loss = 0
        beta=0
        if mode == 0:
            for layer in self.modules():
                if type(layer) == Linear or type(layer)==Linear_re or type(layer)==nn.Linear:
                    w = layer._parameters[ 'weight' ]
                    m = w @ w.T
                    loss=loss+torch.norm(m - torch.eye(m.shape[ 0 ]).to(self.device))
                elif type(layer) == Conv or type(layer)==Conv_re or type(layer)==nn.Conv2d:
                    w = layer._parameters[ 'weight' ]
                    N, C, H, W = w.shape
                    w = w.view(N * C, H, W)
                    m = torch.bmm(w, w.permute(0, 2, 1))
                    loss=loss+ torch.norm(m - torch.eye(H).to(self.device))
        else:
            for layer in self.modules():
                if type(layer) == Linear or type(layer)==Linear_re:
                    iteration = 1
                    w = layer._parameters[ 'weight' ]
                    v = layer.v
                    u = torch.empty((w.shape[ 0 ], 1)).to(self.device)
                    nn.init.uniform_(u,-0.1,0.1)
                    for _ in range(iteration):
                        u = torch.nn.functional.normalize(torch.mm(w.detach(), v), dim=0)
                        v = torch.nn.functional.normalize(torch.mm(w.detach().T, u), dim=0)
                    layer.v = v
                    value=torch.squeeze((u.T @ w @ v).view(-1))
                    loss=loss+value
                elif type(layer) == Conv or type(layer)==Conv_re:
                    iteration = 1
                    w = layer._parameters[ 'weight' ]
                    v = layer.v
                    N, C, H, W = w.shape
                    m = w.view(N, C * W * H)
                    u = torch.empty((m.shape[0], 1)).to(self.device)
                    nn.init.uniform_(u,-0.1,0.1)
                    for _ in range(iteration):
                        u = torch.nn.functional.normalize(torch.mm(m.detach(), v), dim=0)
                        v = torch.nn.functional.normalize(torch.mm(m.detach().T, u), dim=0)
                    layer.v = v
                    value= torch.squeeze((u.T @ m @ v).view(-1))
                    loss=loss+value
        if mode == 1:
            beta=beta2
        else:
            beta=beta1
        return beta * loss