import torch
import torch.nn as nn
from scipy import ndimage, misc
import cv2
import numpy as np
import matplotlib.pyplot as plt
from config.config import *

def bit_squeeze(inputs,out_bits=4):
    matrix_interger=torch.round(inputs*(2**out_bits-1))
    new_inputs=matrix_interger/(2**out_bits-1)
    return new_inputs

def median_filter(inputs,kernel_size=2):
    inputs=inputs.detach().cpu().numpy()
    
    inputs=ndimage.filters.median_filter(inputs, size=(1,1,kernel_size,kernel_size), mode='reflect')
    
    return torch.from_numpy(inputs)

def non_local_filter(inputs,search_window=11,patch_size=3,strength=4):
    inputs=inputs.detach().cpu().numpy()
    inputs=np.transpose(inputs,(0,2,3,1))
    inputs=np.round(inputs*255)
    inputs=inputs.astype(np.uint8)
    
    for i in range(inputs.shape[0]):
        inputs[i]= cv2.fastNlMeansDenoisingColored(inputs[i],None,search_window,search_window,patch_size,strength)
    inputs=inputs/255
    inputs=inputs.astype(np.float32)
    return torch.from_numpy(np.transpose(inputs,(0,3,1,2)))

def de_normbatch(inputs):
    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)
    for i in range(3):
        inputs[:,i,:,:]*=STD[i]
        inputs[:,i,:,:]+=MEAN[i]
    return inputs

def norm_batch(inputs):
    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)
    for i in range(3):
        inputs[:,i,:,:]-=MEAN[i]
        inputs[:,i,:,:]/=STD[i]
    return inputs

def squeeze_function(inputs,bit=False,median=False,non_local=False):
    
    #inputs=de_normbatch(inputs)
    #flag=True
    #if flag==True:
        #fig, axes = plt.subplots(4,5)
        #for i in range(5):
            #ax1=axes[0][i]
            #ax1.imshow(np.transpose(inputs[i].detach().cpu().numpy(),(1,2,0)))
            #ax1.set_axis_off()
        #ax1.set_title('Orignal')    
    
    if bit==True:
        outputs=bit_squeeze(inputs)
        
    elif median==True:
        inputs=bit_squeeze(inputs,out_bits=8)
        outputs=median_filter(inputs)
        
    elif non_local==True:
        inputs=de_normbatch(inputs)
        inputs=bit_squeeze(inputs,out_bits=8)
        outputs=non_local_filter(inputs)
        outputs=norm_batch(outputs)
        
    #if flag==True:
        #for i in range(5):
            #ax1=axes[1][i]
            #ax1.imshow(np.transpose(outputs[i].detach().cpu().numpy(),(1,2,0)))
            #ax1.set_axis_off()
        #plt.savefig('trans.png')
        #ax1.set_title('4-bit')
    
    outputs=norm_batch(outputs)
    return outputs
    
#ina = cv2.imread('/home/mowentao/weixiyu/against_job/util/1.png',flags=cv2.IMREAD_COLOR)

#ina=np.transpose(ina,(2,0,1))
#inputs=torch.from_numpy(ina)
#inputs=torch.unsqueeze(inputs,0)
#median_filter(inputs)
