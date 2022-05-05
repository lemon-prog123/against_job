import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
class CIFAR10():
    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)
    NUM_CLASSES = 10
    IMAGE_SIZE = [ 32, 32 ]
    IMAGE_CHANNELS = 3

    def __init__(self):
        #transform = transforms.Compose([ transforms.ToTensor()])
        transform = transforms.Compose([ transforms.ToTensor(),
                                        transforms.Normalize(self.MEAN, self.STD)])
        self.load_dataset(transform)

    def load_dataset(self, transform):
        self.trainset = torchvision.datasets.CIFAR10(root="/home/mowentao/weixiyu/against_job/dataset", transform=transform, download=False)
        self.testset = torchvision.datasets.CIFAR10(root="/home/mowentao/weixiyu/against_job/dataset", train=False, transform=transform, download=False)