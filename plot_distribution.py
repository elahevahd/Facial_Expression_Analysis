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
pain_levels = config.pain_levels

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
    

def bins_labels(bins, **kwargs):
    bin_w = (max(bins) - min(bins)) / (len(bins) - 1)
    plt.xticks(np.arange(min(bins)+bin_w/2, max(bins), bin_w), bins, **kwargs)
    plt.xlim(bins[0], bins[-1])


def plot_distribution(inner_fold,des_path,outer_folder):
    if inner_fold == 'outer':
        train_list = gr_labels('data_split/'+labeling+'/{}_train.txt'.format(outer_folder))
        test_list = gr_labels('data_split/'+labeling+'/{}_test.txt'.format(outer_folder))
    else:
        train_list = gr_labels('data_split/'+labeling+'/{}_{}_train.txt'.format(outer_folder,inner_fold))
        valid_list = gr_labels('data_split/'+labeling+'/{}_{}_valid.txt'.format(outer_folder,inner_fold))
        test_list = gr_labels('data_split/'+labeling+'/{}_test.txt'.format(outer_folder))

    train_res_list=[]
    for index in range(len(train_list)):
        train_res_list= train_res_list + list(np.full((train_list[index],),index))
    test_res_list=[]
    for index in range(len(test_list)):
        test_res_list= test_res_list + list(np.full((test_list[index],),index))
    
    fig, ax = plt.subplots()

    bins = list(range(0,pain_levels+1,1))

    plt.hist([train_res_list,test_res_list], bins=bins, label=['Train','Test'])
    plt.xlabel('Pain Level',size=10)
    bins_labels(bins, fontsize=10)
    plt.ylabel('Number of videos',size=10)
    plt.yticks(np.arange(0, max(train_list), step=20))
    for i, v in enumerate(train_list):
        ax.text(i+.1 , v + .25, str(v), color='blue')
    for i, v in enumerate(test_list):
        ax.text(i+.5, v + .25, str(v), color='red')

    plt.title('Distribution Fold '+str(outer_folder) ,  fontsize=20)
    #     plt.title('Distribution Inner Fold '+str(outer_folder+'.'+inner_fold) ,  fontsize=20)

    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(des_path,bbox_inches='tight')




def plot_distribution_whole():
    # train_list = gr_labels('data_split/VAS/ext_data.txt')
    train_list = gr_labels('data_split/VAS/final_distribution_VAS_labels.txt')
    train_res_list=[]
    for index in range(len(train_list)):
        train_res_list= train_res_list + list(np.full((train_list[index],),index))

    fig, ax = plt.subplots()

    bins = list(range(0,pain_levels+1,1))

    plt.hist([train_res_list], bins=bins, label=[''])
    plt.xlabel('Pain Level',size=10)
    bins_labels(bins, fontsize=10)
    plt.ylabel('Number of videos',size=10)
    plt.yticks(np.arange(0, max(train_list), step=20))
    for i, v in enumerate(train_list):
        ax.text(i+.1 , v + .25, str(v), color='blue')

    plt.title('Distribution Data',  fontsize=20)

    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('VAS_Distribution.png',bbox_inches='tight')


plot_distribution_whole()