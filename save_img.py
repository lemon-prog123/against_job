from tkinter import font
import numpy as np
import matplotlib.pyplot as plt

def de_norm(img):
    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)
    for i in range(3):
        img[:,:,i]*=STD[i]
        img[:,:,i]+=MEAN[i]
    return img

def loss_img(x,y,img_path='img'):
    plt.clf()
    plt.title('Train loss vs Epoches',fontsize=15)
    plt.plot(x,y,'-')
    plt.xlabel('Epoches',fontsize=10)
    plt.xlabel('Tarin loss',fontsize=10)
    i=0
    for i_x,i_y in zip(x,y):
        if i%2==0:
            plt.text(i_x+0.3, i_y, '(%.1f)'%(i_y),fontsize=10)
            plt.plot([i_x],[i_y],'o')
        i=i+1
    plt.grid()
    plt.savefig(img_path+'/Train_loss.png')
    
def acc_img(x,y,img_path='img'):
    plt.clf()
    plt.title('Accuracy vs Epoches',fontsize=15)
    plt.plot(x,y,'-')
    plt.xlabel('Epoches',fontsize=12)
    plt.xlabel('Accuracy',fontsize=12)
    i=0
    for i_x,i_y in zip(x,y):
        if i%2==0:
            plt.text(i_x+0.5, i_y-0.5, '(%.1f%%)'%(i_y),fontsize=10)
            plt.plot([i_x],[i_y],'o')
        i=i+1

    plt.grid()
    plt.savefig(img_path+'/Accuracy.png')
    
def im_show( img_list,img_label,advimg_list,advimg_label,img_path='img'):
    classes = ('plane','car','bird','cat','deer',
          'dog','frog','horse','ship','truck')
    fig, axes = plt.subplots(2,5)
    for i in range(5):
        ax1=axes[0][i]
        ax2=axes[1][i]
        ax1.imshow(de_norm(np.transpose(img_list[i],(1,2,0))))
        ax1.set_title('Orignal as '+classes[img_label[i]])
        ax1.set_axis_off()
        ax2.imshow(de_norm(np.transpose(advimg_list[i],(1,2,0))))
        ax2.set_title('After as '+classes[advimg_label[i]])
        ax2.set_axis_off()
    plt.tight_layout()
    plt.savefig(img_path+'/pgd.png')
    
    plt.figure()
    plt.imshow(de_norm(np.transpose(img_list[4],(1,2,0))))
    plt.axis('off')
    plt.savefig(img_path+'/1.png')