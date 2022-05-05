import torch, torch.nn as nn

def setparams(act):
    act.Bl.data = (act.running_mean.detach() - 3*act.running_var.detach()).cuda()
    act.Br.data = (act.running_mean.detach() + 3*act.running_var.detach()).cuda()
    act.Yidx.data= (nn.functional.relu(torch.linspace(act.Bl.item(),act.Br.item(),act.N+1))).cuda()
    return act