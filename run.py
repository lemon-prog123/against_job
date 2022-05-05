import os
import math
import numpy as np
import random
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from util.setseed import *
from util.save import *
from net.covnet import *
from log.logger import *
from dataloader.cifar10 import *
from common.engine import *
from config.config import *
from util.statloader import *
import matplotlib.pyplot as plt
import pprint

if __name__ == '__main__':
    args = argparser.parse_args()
    setup_seed(1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if args.denorm:
        norm=False
    
    dataset=CIFAR10(norm)
    logger = create_logger()
    logger.info(pprint.pformat(args))
    BATCH_SIZE=32
    train_loader = DataLoader(dataset.trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = DataLoader(dataset.testset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    
    if args.iterations!=None:
        iteration1=args.iterations
        iteration2=args.iterations
        
    if args.squeeze!=None:
        if args.squeeze <0 or args.squeeze>2:
            raise Exception('Wrong Squeeze Type')
        else:
            squeeze=args.squeeze
        
    if args.savepath:
        save_path =save_path+'/'+args.savepath
    
    if args.imgpath:
        img_path=img_path+'/'+args.imgpath
        if not os.path.exists(img_path):
            os.mkdir(img_path)
    
    if args.regular:
        regular=True
        regular_mode=0
        
    if args.spectral:
        spectral=True
        regular=True
        
    if args.activate:
        activate=True
        
    if args.quant:
        quant=True
    
    
    model=ConvNet(quant=quant,spectral=spectral)
    model.to(device)
    
    if args.checkpoint:
        model=load_model(model,logger,activate=activate,stat_path=args.checkpoint)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    if args.PGD_train:
        pgd_train(train_loader,test_loader,model,criterion,optimizer,logger,quant=quant,
                  activate=activate,regular=regular,regular_mode=regular_mode,iteration=iteration2,path=save_path,img_path=img_path)
    elif args.eval==False:
        train(train_loader,test_loader,model,criterion,optimizer,logger,quant=quant,activate=activate,regular=regular,regular_mode=regular_mode,
              iteration=iteration1,path=save_path,img_path=img_path,squeeze=squeeze)
    
    if args.eval==True:
        acc=eval(test_loader,model,quant=quant,squeeze=squeeze)
        logger.info('accuracy:%.1f %%'%(acc))
    
    eps=[16/255,12/255,8/255,4/255,1/255,1/(255*4)]
    acc=[]
    
    for e in eps:
        v=pgd(test_loader,model,logger,quant=quant,img_path=img_path,squeeze=squeeze,eps=e)
        acc.append(v)
    plt.clf()
    plt.title('Accuracy vs eps',fontsize=15)
    plt.plot(eps,acc,'-')
    plt.xlabel('eps',fontsize=10)
    plt.ylabel('Accuracy',fontsize=10)
    i=0
    for i_x,i_y in zip(eps,acc):
        plt.text(i_x, i_y-1, '(%.3f,%.1f)'%(i_x,i_y),fontsize=10)
        plt.plot([i_x],[i_y],'o')
        i=i+1
    plt.grid()
    plt.savefig('PGD.png')





