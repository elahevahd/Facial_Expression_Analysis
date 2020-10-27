import ast
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import os 
from  importlib import reload
import mean_metric_class
reload(mean_metric_class)
from mean_metric_class import *  

import config
labeling = config.labeling
pain_levels = config.pain_levels

# params = {
# 'weight_decay': [1e-05] ,
# 'lr': [3e-03,5e-03,1e-02],
# 'dec_freq':[1],
# 'dropout_value':[0.3,0.5]
# }


params = {
'weight_decay': [1e-05] ,
'lr': [3e-03,5e-03],
'dec_freq':[1],
'dropout_value':[0.5]
}

number_of_pars=len(params['lr'])*len(params['weight_decay'])*len(params['dec_freq']*len(params['dropout_value']))


outer_fold_list = ['1','2','3','4','5']

def find_best_index(outer_fold):
    if outer_fold == '1':
        return '3'
    else:
        d= 10000000000
        best_index=-1
        for index in range(number_of_pars):
            line = list(open('Fold_{}/{}/best_model/pair_{}.txt'.format(outer_fold,labeling,index)))[3].strip()
            outer_MSE_score = float(line.split('outer_MSE_score: ')[1])
            # print(outer_MSE_score)
            if outer_MSE_score<d:
                best_index = index
                d = outer_MSE_score
        return best_index


def bins_labels(bins, **kwargs):
    bin_w = (max(bins) - min(bins)) / (len(bins) - 1)
    plt.xticks(np.arange(min(bins)+bin_w/2, max(bins), bin_w), bins, **kwargs)
    plt.xlim(bins[0], bins[-1])


def acc_result(train_dist, test_dist):
    train_res_list=[]
    for index in range(len(train_dist)):
        train_res_list= train_res_list + list(np.full((train_dist[index],),index))
    test_res_list=[]
    for index in range(len(test_dist)):
        test_res_list= test_res_list + list(np.full((test_dist[index],),index))
    return train_res_list, test_res_list


def gr_labels(filename):
    file_list = list(open(filename))
    if labeling == 'VAS':
        freq_dict={'0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0, '9': 0, '10': 0}
    elif labeling == 'OPR':
        freq_dict={'0': 0, '1': 0, '2': 0, '3': 0}
    for line in file_list:
        label =line.split('/')[-1].split('-')[-1].strip()
        freq_dict[label] += 1 
    val_list = [freq_dict[key] for key in sorted(freq_dict.keys(), key=int)]
    return val_list
    

def MAE_extension_mean(outer_fold,A,B):
    best_index = find_best_index(outer_fold)
    label_dir = 'Fold_{}/{}/labels/{}/early_stop/outer'.format(outer_fold,labeling,best_index)
    labels_arr = np.load(label_dir+'/label.np.npy')
    predict_arr = np.load(label_dir+'/predict.np.npy')
    obj_1= metric_calculator(labels_arr, predict_arr)
    A= A+np.array(obj_1.return_MAE_list())
    B= B+obj_1.return_MAE_avg()
    return A,B

def MSE_extension_mean(outer_fold,A,B):
    best_index = find_best_index(outer_fold)
    label_dir = 'Fold_{}/{}/labels/{}/early_stop/outer'.format(outer_fold,labeling,best_index)
    labels_arr = np.load(label_dir+'/label.np.npy')
    predict_arr = np.load(label_dir+'/predict.np.npy')
    obj_1= metric_calculator(labels_arr, predict_arr)
    A= A+np.array(obj_1.return_MSE_list())
    B= B+obj_1.return_MSE_avg()
    return A,B

def MAE_dist_mean(outer_fold,A,B):
    best_index = find_best_index(outer_fold)
    label_dir = 'Fold_{}/{}/labels/{}/early_stop/dist'.format(outer_fold,labeling,best_index)
    labels_arr = np.load(label_dir+'/label.np.npy')
    predict_arr = np.load(label_dir+'/predict.np.npy')
    obj_1= metric_calculator(labels_arr, predict_arr)
    A= A+np.array(obj_1.return_MAE_list())
    B= B+obj_1.return_MAE_avg()
    return A,B

def MSE_dist_mean(outer_fold,A,B):
    best_index = find_best_index(outer_fold)
    label_dir = 'Fold_{}/{}/labels/{}/early_stop/dist'.format(outer_fold,labeling,best_index)
    labels_arr = np.load(label_dir+'/label.np.npy')
    predict_arr = np.load(label_dir+'/predict.np.npy')
    obj_1= metric_calculator(labels_arr, predict_arr)
    A= A+np.array(obj_1.return_MSE_list())
    B= B+obj_1.return_MSE_avg()
    return A,B

def extension_draw_mean_MAE():
    A=np.zeros(pain_levels)
    B=0
    for outer_fold in outer_fold_list:  #add '5'
        A,B = MAE_extension_mean(outer_fold,A,B)
    barlist = plt.bar(range(pain_levels),A/len(outer_fold_list))
    for i in range(pain_levels):
        barlist[i].set_color('r')
    plt.xlabel('Pain Level',size=15)
    plt.ylabel('MAE ',size=15)
    plt.xticks(np.arange(0, pain_levels, step=1))
    plt.yticks(np.arange(0, pain_levels, step=1))
    plt.legend(['MAE Avg = '+str(round(B/len(outer_fold_list),3))], loc='best')


def extension_draw_mean_MSE():
    A=np.zeros(pain_levels)
    B=0
    for outer_fold in outer_fold_list:  #add '5'
        A,B = MSE_extension_mean(outer_fold,A,B)
    barlist = plt.bar(range(pain_levels),A/len(outer_fold_list))
    for i in range(pain_levels):
        barlist[i].set_color('r')
    plt.xlabel('Pain Level',size=15)
    plt.ylabel('MSE ',size=15)
    plt.xticks(np.arange(0, pain_levels, step=1))

    if labeling == 'VAS':
        MSE_step= 5
    elif labeling == 'OPR':
        MSE_step =1

    plt.yticks(np.arange(0, (pain_levels-1)**2+5, step=MSE_step))
    plt.legend(['MSE Avg = '+str(round(B/len(outer_fold_list),3))], loc='best')

def dist_draw_mean_MAE():
    A=np.zeros(pain_levels)
    B=0
    for outer_fold in outer_fold_list:  #add '5'
        A,B = MAE_dist_mean(outer_fold,A,B)
    barlist = plt.bar(range(pain_levels),A/len(outer_fold_list))
    for i in range(pain_levels):
        barlist[i].set_color('r')
    plt.xlabel('Pain Level',size=15)
    plt.ylabel('MAE ',size=15)
    plt.xticks(np.arange(0, pain_levels, step=1))
    plt.yticks(np.arange(0, pain_levels, step=1))
    plt.legend(['MAE Avg = '+str(round(B/len(outer_fold_list),3))], loc='best')



def dist_draw_mean_MSE():
    A=np.zeros(pain_levels)
    B=0
    for outer_fold in outer_fold_list:  #add '5'
        A,B = MSE_dist_mean(outer_fold,A,B)
    barlist = plt.bar(range(pain_levels),A/len(outer_fold_list))
    for i in range(pain_levels):
        barlist[i].set_color('r')
    plt.xlabel('Pain Level',size=15)
    plt.ylabel('MSE ',size=15)
    plt.xticks(np.arange(0, pain_levels, step=1))
    if labeling == 'VAS':
        MSE_step= 5
    elif labeling == 'OPR':
        MSE_step =1
    plt.yticks(np.arange(0,(pain_levels-1)**2+5, step=MSE_step))
    plt.legend(['MSE Avg = '+str(round(B/len(outer_fold_list),3))], loc='best')

def draw_all_means():

    fig, ax = plt.subplots()
    plt.figure(figsize=(10,5))
    plt.subplot(121)
    extension_draw_mean_MAE()
    plt.subplot(122)
    extension_draw_mean_MSE()
    plt.suptitle('Mean Results for Extension Data ' ,  fontsize=20)
    fig_name = 'mean_results/{}/mean_extension_fig.png'.format(labeling)
    plt.savefig(fig_name,bbox_inches='tight')

    fig, ax = plt.subplots()
    plt.figure(figsize=(10,5))
    plt.subplot(121)
    dist_draw_mean_MAE()
    plt.subplot(122)
    dist_draw_mean_MSE()
    plt.suptitle('Mean Results for Distribution Data ' ,  fontsize=20)
    fig_name = 'mean_results/{}/mean_distribution_fig.png'.format(labeling)
    plt.savefig(fig_name,bbox_inches='tight')









