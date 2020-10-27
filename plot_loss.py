import os 
import matplotlib.pyplot as plt
import numpy as np 
from load_save_obj import * 
from config import *
from config import epoch


y_height= 0.25
y_step = 0.05

x_step= 50

epoch= epoch+x_step
fig_width = 15
fig_height = 5

def plot_loss_train(par,des_path,lr_value,weight_decay_value,dec_freq,outer_folder):
    train_loss = load_obj('Fold_{}/{}/loss_figs/pair_{}/train_loss_values'.format(outer_folder,labeling,par))
    ax,fig= plt.subplots()
    plt.plot(range(len(train_loss)), train_loss, 'b')
    plt.legend(['Training Loss'], loc='best')
    plt.xlabel('Time',size=10)
    plt.ylabel('Loss',size=10)
    x_range = len(train_loss)
    plt.xticks(np.arange(0, epoch+1, step=5))
    plt.yticks(np.arange(0, y_height, step=y_step))
    text_1 = 'lr = '+str(lr_value)
    text_2 = 'wdecay = '+str(weight_decay_value)
    text_3= 'dec_freq = '+str(dec_freq)
    plt.text(0.02, 0.5, text_1, fontsize=10, transform=plt.gcf().transFigure)
    plt.text(0.02, 0.4, text_2, fontsize=10, transform=plt.gcf().transFigure)
    plt.text(0.02, 0.3, text_3, fontsize=10, transform=plt.gcf().transFigure)
    plt.subplots_adjust(left=0.3)
    plt.savefig(des_path,bbox_inches='tight')

def plot_loss_trainval(par, des_path,lr_value,weight_decay_value,dec_freq,outer_folder):
    train_loss = load_obj('Fold_{}/{}/loss_figs/pair_{}/train_loss_values'.format(outer_folder,labeling,par))
    valid_loss = load_obj('Fold_{}/{}/loss_figs/pair_{}/valid_loss_values'.format(outer_folder,labeling,par))
    ax,fig= plt.subplots(figsize=(fig_width,fig_height))
    plt.plot(range(len(train_loss)), train_loss, 'b',range(len(valid_loss)),valid_loss,'r')
    plt.legend(['Training Loss','Validation Loss'], loc='best')
    plt.xlabel('Time',size=10)
    plt.ylabel('Loss',size=10)
    x_range = len(train_loss)
    plt.xticks(np.arange(0, epoch+1, step=x_step))
    plt.yticks(np.arange(0, y_height, step=y_step))
    text_1 = 'lr = '+str(lr_value)
    text_2 = 'wdecay = '+str(weight_decay_value)
    text_3= 'dec_freq = '+str(dec_freq)
    plt.text(0.02, 0.5, text_1, fontsize=10, transform=plt.gcf().transFigure)
    plt.text(0.02, 0.4, text_2, fontsize=10, transform=plt.gcf().transFigure)
    plt.text(0.02, 0.3, text_3, fontsize=10, transform=plt.gcf().transFigure)
    plt.subplots_adjust(left=0.15)
    plt.savefig(des_path,bbox_inches='tight')



# params = {
# 'weight_decay':  [1e-05],
# 'lr': [5e-04],
# 'dec_freq':[1],
# }

# par=0
# outer_folder='1'
# labeling='VAS'
# plot_path = 'Fold_{}/{}/plots/loss/{}_fixed_outer.png'.format(outer_folder,labeling,str(par))
# lr_value,weight_decay_value,dec_freq= [(x,y,z) for x in params['lr'] for y in params['weight_decay']  for z in params['dec_freq']][par]
# plot_loss_trainval('outer',str(par), plot_path,lr_value,weight_decay_value,dec_freq,outer_folder)
