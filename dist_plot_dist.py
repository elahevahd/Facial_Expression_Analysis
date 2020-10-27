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

import config
labeling = config.labeling


def actual_pred(predict_file): #return the actual labels of predict_file 
    B=np.load(predict_file)
    pred_lst=[]  
    for i in range(B.shape[0]):
        pred=B[i,:]
        pred = torch.from_numpy(pred)
        x=torch.topk(pred,1)[1].item()
        pred_lst.append(x)
    return pred_lst


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
    


def dist_plot_distribution(inner_fold,outer_folder):

    if inner_fold == 'outer':
        train_list = gr_labels('data_split/'+labeling+'/{}_train.txt'.format(outer_folder))
    else: 
        train_list = gr_labels('data_split/'+labeling+'/{}_{}_train.txt'.format(outer_folder,inner_fold))


    train_res_list=[]
    for index in range(len(train_list)):
        train_res_list= train_res_list + list(np.full((train_list[index],),index))

    test_dist = gr_labels('data_split/'+labeling+'/final_distribution_{}_labels.txt'.format(labeling))
    test_res_list=[]
    for index in range(len(test_dist)):
        test_res_list= test_res_list + list(np.full((test_dist[index],),index))

    fig, ax = plt.subplots()

    if labeling == 'VAS':
        bins = list(range(0,12,1))
        plt.xlabel('Pain Level',size=10)
    elif labeling == 'OPR':
        bins = list(range(0,5,1))
        plt.xlabel('Pain Level',size=3)

    plt.hist([train_res_list,test_res_list], bins=bins, label=['Extension Data','Distribution Data'])
    
    bins_labels(bins, fontsize=10)
    plt.ylabel('Number of videos',size=10)
    plt.yticks(np.arange(0, max(train_list), step=20))
    for i, v in enumerate(train_list):
        ax.text(i+.1 , v + .25, str(v), color='blue')
    for i, v in enumerate(test_dist):
        ax.text(i+.5, v + .25, str(v), color='red')

    plt.title('Distribution of OPR Pain Levels')
    plt.legend(loc='upper right')
    plt.tight_layout()
    # plt.show()
    des_path = 'Fold_{}/{}/plots/distribution/'.format(outer_folder,labeling)+outer_folder+'_dist.png'
    plt.savefig(des_path,bbox_inches='tight')



def plot_distribution_all(outer_folder):
    train_list = list(np.array(gr_labels('data_split/'+labeling+'/{}_train.txt'.format(outer_folder)))+np.array(gr_labels('data_split/'+labeling+'/{}_test.txt'.format(outer_folder))))
    train_res_list=[]
    for index in range(len(train_list)):
        train_res_list= train_res_list + list(np.full((train_list[index],),index))
    
    fig, ax = plt.subplots()

    if labeling == 'VAS':
        bins = list(range(0,12,1))
        plt.xlabel('Pain Level',size=10)
    elif labeling == 'OPR':
        bins = list(range(0,5,1))
        plt.xlabel('Pain Level',size=3)

    plt.hist([train_res_list], bins=bins, label=['Data'])
    bins_labels(bins, fontsize=10)
    plt.ylabel('Number of videos',size=10)
    plt.yticks(np.arange(0, max(train_list), step=20))
    for i, v in enumerate(train_list):
        ax.text(i+.1 , v + .25, str(v), color='blue')
    plt.title('Extension Data' ,  fontsize=20)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(plot_distribution_all_path,bbox_inches='tight')

