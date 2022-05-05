import torch
import os
def save_checkpoint(state,is_best,save_path='checkpoint'):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    epoch = state['epoch']
    filename = f'{save_path}/epoch_{epoch}.pth'
    torch.save(state, filename)
    if is_best:
        if os.path.exists(f'{save_path}/best.pth'):
            os.remove(f'{save_path}/best.pth')
        torch.save(state, f'{save_path}/best.pth')