from enum import Flag
import os
import time
import torch
import torch.optim
import torch.utils.data
from net.PGD import *
from util.save import *
from util.statloader import*
from util.squeeze import *
import matplotlib.pyplot as plt
from save_img import * 
def pgd_train(train_loader,test_loader,model,criterion,optimizer,logger,quant=False,activate=False,iteration=10,regular=False,regular_mode=0,path="checkpoint",img_path='img'):
    logger.info('Start PGD Training')
    pgd_net=PGD(model,quant)
    device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    best_acc = 0
    x_data=[]
    y_data=[]
    acc_data=[]
    
    for epoch in range(iteration):
        running_loss = 0
        model.train()
        x_data.append(epoch+1)
        if epoch==5 and activate==True:
            model.activation()
        for i,data in enumerate(train_loader):
            inputs,labels=data
            inputs=inputs.to(device)
            labels=labels.to(device)
            
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, labels)
            if regular==True:
                loss=loss+model.regularizationTerm(mode=regular_mode)
            loss.backward()
            optimizer.step()
            
            optimizer.zero_grad()
            adv_inputs=pgd_net.generate(x=inputs,y=labels)
            optimizer.zero_grad()
            output = model(adv_inputs)
            loss = criterion(output, labels)
            if regular==True:
                loss=loss+model.regularizationTerm(mode=regular_mode)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if i % 1000 == 999:
                logger.info('[%d,%5d] loss:%.3f' % (epoch + 1, i + 1, running_loss / 1000))
                y_data.append(running_loss/1000)
                running_loss = 0.0
                
        if epoch %1==0:
            is_best = False
            acc=eval(test_loader,model,quant)
            acc_data.append(acc)
            if acc > best_acc:
                is_best = True
                best_acc=acc
            save_checkpoint({'epoch': epoch, 'state_dict': model.state_dict()},
                            is_best=is_best,save_path=path)
            logger.info('accuracy:%d %% best_acc:%d %%' % (acc, best_acc))
    
    loss_img(x_data,y_data,img_path)
    acc_img(x_data,acc_data,img_path)

def train(train_loader,test_loader,model,criterion,optimizer,logger,quant=False,activate=False,iteration=15,regular=False,regular_mode=0,path="checkpoint",img_path='img',squeeze=-1):
    logger.info('Start Training')
    device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    best_acc = 0
    x_data=[]
    y_data=[]
    acc_data=[]
    
    for epoch in range(iteration):
        model.train()
        running_loss = 0
        x_data.append(epoch+1)
        if epoch==5 and activate==True:
            model.activation()
        for i,data in enumerate(train_loader):
            
            inputs,labels=data
        
            #with torch.no_grad():
                #if squeeze==0:
                    #inputs=squeeze_function(inputs,bit=True)
                #elif squeeze==1:
                    #inputs=squeeze_function(inputs,median=True)
                #elif squeeze==2:
                    #inputs=squeeze_function(inputs,non_local=True)

            inputs=inputs.to(device)
            labels=labels.to(device)
            
            
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, labels)
            if regular==True:
                loss=loss+model.regularizationTerm(mode=regular_mode)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 1000 == 999:
                logger.info('[%d,%5d] loss:%.3f' % (epoch + 1, i + 1, running_loss / 1000))
                y_data.append(running_loss/1000)
                running_loss = 0.0
        
        if epoch %1==0:
            is_best = False
            acc=eval(test_loader,model,quant,squeeze)
            
            acc_data.append(acc)
            if acc > best_acc:
                is_best = True
                best_acc=acc
            save_checkpoint({'epoch': epoch, 'state_dict': model.state_dict()},
                           is_best=is_best,save_path=path)
            logger.info('accuracy:%.1f %% best_acc:%.1f %%' % (acc, best_acc))
            
    loss_img(x_data,y_data,img_path)
    acc_img(x_data,acc_data,img_path)

    
def pgd(test_loader,model,logger,quant=False,img_path='img',bit=False,median=False,non_local=False,squeeze=-1,eps=8/255):

    pgd_net=PGD(model,quant)
    total=0
    correct=0
    total_attack=0
    succ_attack=0
    show_flag=True
    count=0
    img_list=[]
    img_label=[]
    advimg_list=[]
    advimg_label=[]
    
    for data in test_loader:
        
        inputs, labels = data
            
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()
        

        
        adv_inputs=pgd_net.generate(x=inputs,y=labels,eps=eps, iter_eps=eps/4)

        with torch.no_grad():
            if squeeze==0:
                adv_inputs=squeeze_function(adv_inputs,bit=True)
            elif squeeze==1:
                adv_inputs=squeeze_function(adv_inputs,median=True)
            elif squeeze==2:
                adv_inputs=squeeze_function(adv_inputs,non_local=True)
        
        if torch.cuda.is_available():
            adv_inputs=adv_inputs.cuda()
        
        model.eval()
        with  torch.no_grad():
            output1 = model(adv_inputs)
            output2=model(inputs)
        
        _, predicted1 = torch.max(output1.data, 1)
        _, predicted2 = torch.max(output2.data, 1)
        ans1=(predicted1==labels)
        ans2=(predicted2==labels)
        total_attack+=ans2.sum().item()
        succ=(ans1==False)&(ans2==True)

        if show_flag==True:
            for i in range(inputs.shape[0]):
                if succ[i]==True:
                    count+=1
                    img_list.append(inputs[i].detach().cpu().numpy())
                    img_label.append(labels[i].item())
                    advimg_list.append(adv_inputs[i].detach().cpu().numpy())
                    advimg_label.append(predicted1[i].item())
                if count==5:
                    show_flag=False
                    break
        
        succ_attack+=succ.sum().item()
        total += labels.size(0)
        correct += (predicted1 == labels).sum().item()
        
    logger.info('PGD accuracy:%.1f %% Attack Success:%.1f %%' % (100.0 * correct / total,100.0*succ_attack/total_attack))

    #print(count)
    if show_flag==False:
        im_show(img_list,img_label,advimg_list,advimg_label,img_path)
    return (100.0 * correct / total)
        
def eval(test_loader,model,quant=False,squeeze=-1):
    total=0
    correct=0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            
            if squeeze==0:
                inputs=squeeze_function(inputs,bit=True)
            elif squeeze==1:
                inputs=squeeze_function(inputs,median=True)
            elif squeeze==2:
                inputs=squeeze_function(inputs,non_local=True)
            
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()
            
            output = model(inputs)
            #if quant==True:
                #model.reconstruct()
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return ((100.0* correct) / total)
