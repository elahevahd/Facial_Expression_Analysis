import ast
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import os 
from  importlib import reload
import metric_class
reload(metric_class)
from metric_class import *  
from config import *
#-------------------------------
import plot_distribution
reload(plot_distribution)
from plot_distribution import *  
#-------------------------------
import dist_plot_dist
reload(dist_plot_dist)
from dist_plot_dist import *  
#-------------------------------
import plot_loss
reload(plot_loss)
from plot_loss import *  
#-------------------------------
from load_save_obj import * 
from config import *


def draw_final_fig(label_path,des_path):
    labels_arr = np.load(label_path+'/label.np.npy')
    predict_arr = np.load(label_path+'/predict.np.npy')   
     
    obj_1= metric_calculator(labels_arr, predict_arr)
    fig, ax = plt.subplots()
    plt.figure(figsize=(15,5))
    plt.subplot(131)
    obj_1.draw_MAE()
    plt.subplot(132)
    obj_1.draw_RMSE() 
    plt.subplot(133)
    obj_1.draw_MSE() 
    plt.savefig(des_path,bbox_inches='tight')



def all_plots(inner_fold,outer_folder,index):

    plot_distribution(inner_fold,'Fold_{}/{}/plots/distribution/{}_outer.png'.format(outer_folder,labeling,outer_folder),outer_folder)
    dist_plot_distribution(inner_fold,outer_folder)

    label_path = 'Fold_{}/{}/labels/{}/early_stop/outer'.format(outer_folder,labeling,index)
    draw_final_fig(label_path,'Fold_{}/{}/plots/scores/{}_outer_earlystop.png'.format(outer_folder,labeling,index))   

    label_path = 'Fold_{}/{}/labels/{}/early_stop/dist'.format(outer_folder,labeling,index)
    draw_final_fig(label_path,'Fold_{}/{}/plots/scores/{}_dist_earlystop.png'.format(outer_folder,labeling,index))


