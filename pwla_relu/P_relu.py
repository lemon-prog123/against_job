import torch
import torch.nn as nn
class PWLA1d(nn.Module):
    def __init__(self,N=16,momentum=0.9):
        super(PWLA1d, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.N = N
        self.momentum = torch.tensor(momentum).to(self.device)
        self.Br = torch.nn.Parameter(torch.tensor(10.))
        self.Bl = torch.nn.Parameter(torch.tensor(-10.))
        self.Kl = torch.nn.Parameter(torch.tensor(0.))
        self.Kr = torch.nn.Parameter(torch.tensor(1.))
        self.running_mean=torch.zeros(1).to(self.device)
        self.running_var=torch.ones(1).to(self.device)
        self.Yidx = torch.nn.Parameter(nn.functional.relu(torch.linspace(self.Bl.item(),self.Br.item(),self.N+1)))
        self.reset_parameters()
    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)
    def forward(self, x, mode):
        if mode == 1:
            d = (self.Br - self.Bl) / self.N  # Interval length
            DATAind = torch.clamp(torch.floor((x - self.Bl.item()) / d), 0,
                                  self.N - 1).to(self.device)  # Number of corresponding interval for X
            Bdata = self.Bl + DATAind * d  # LEFT Interval boundaries
            Bdata=Bdata.to(self.device)
            maskBl = x < self.Bl  # Mask for LEFT boundary
            maskBr = x >= self.Br  # Mask for RIGHT boundary
            maskOther = ~(maskBl + maskBr)  # Mask for INSIDE boundaries
            maskBl=maskBl.to(self.device)
            maskBr=maskBr.to(self.device)
            maskOther=maskOther.to(self.device)
            Ydata = self.Yidx[ DATAind.type(torch.int64) ]  # Y-value for data
            Kdata = (self.Yidx[ (DATAind).type(torch.int64) + 1 ] - self.Yidx[
                DATAind.type(torch.int64) ]) / d  # SLOPE for data
            return maskBl * ((x - self.Bl) * self.Kl + self.Yidx[ 0 ]) + maskBr * (
                    (x - self.Br) * self.Kr + self.Yidx[ -1 ]) + maskOther * ((x - Bdata) * Kdata + Ydata)
        else:
            mean = x.detach().mean([ 0, -1 ])
            var = x.detach().var([ 0, -1 ])
            self.running_mean = (self.momentum * self.running_mean) + (
                    1.0 - self.momentum) * mean  # .to(input.device)
            self.running_var = (self.momentum * self.running_var) + (1.0 - self.momentum) * (
                    x.shape[ 0 ] / (x.shape[ 0 ] - 1) * var)
            return nn.functional.relu(x)

class PWLA3d(torch.nn.Module):

    def __init__(self,N=16,momentum=0.9):
        super(PWLA3d, self).__init__()
        self.device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.N = N
        self.momentum = torch.tensor(momentum).to(self.device)
        self.Br = torch.nn.Parameter(torch.tensor(10.))
        self.Bl = torch.nn.Parameter(torch.tensor(-10.))
        self.Kl = torch.nn.Parameter(torch.tensor(0.))
        self.Kr = torch.nn.Parameter(torch.tensor(1.))
        self.running_mean=torch.zeros(1).to(self.device)
        self.running_var=torch.ones(1).to(self.device)
        self.Yidx = torch.nn.Parameter(nn.functional.relu(torch.linspace(self.Bl.item(),self.Br.item(),self.N+1)))
        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)

    def forward(self, x, mode):
        if mode==1:
            d=(self.Br-self.Bl)/self.N#Interval length
            DATAind = torch.clamp(torch.floor((x-self.Bl.item())/d),0,self.N-1).to(self.device)#Number of corresponding interval for X
            Bdata = self.Bl+DATAind*d#LEFT Interval boundaries
            Bdata=Bdata.to(self.device)
            maskBl = x<self.Bl#Mask for LEFT boundary
            maskBr = x>=self.Br#Mask for RIGHT boundary
            maskOther = ~(maskBl+maskBr)#Mask for INSIDE boundaries
            maskBl=maskBl.to(self.device)
            maskBr=maskBr.to(self.device)
            maskOther=maskOther.to(self.device)
            Ydata = self.Yidx[DATAind.type(torch.int64)]
            Kdata = (self.Yidx[(DATAind).type(torch.int64)+1]-self.Yidx[DATAind.type(torch.int64)])/d#SLOPE for data
            return  maskBl*((x-self.Bl)*self.Kl+self.Yidx[0]) + maskBr*((x-self.Br)*self.Kr + self.Yidx[-1]) + maskOther*((x-Bdata)*Kdata + Ydata)
        else:
            mean = x.detach().mean([0,1,2,-1])
            var = x.detach().var([0,1,2,-1])
            self.running_mean = (self.momentum * self.running_mean) + (1.0-self.momentum) * mean
            self.running_var = (self.momentum * self.running_var) + (1.0-self.momentum) * (x.shape[0]/(x.shape[0]-1)*var)
            return nn.functional.relu(x)