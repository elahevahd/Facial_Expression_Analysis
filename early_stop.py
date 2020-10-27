from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from numpy.random import normal
from numpy.linalg import svd
from math import sqrt
import numpy as np
import math
import os
import matplotlib.pyplot as plt
import random
from torch.utils import data
import torchvision
from resnet import resnet34, resnet10, resnet18, resnet50
from dataloader import ASL_Dataloader
import time
from torch.optim import lr_scheduler
import itertools
import sklearn
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tempfile import TemporaryFile
import operator
from load_save_obj import * 
from config import *
import config
from config import * 
labeling=config.labeling
scale_factor= config.scale_factor

def rev_norm_func(x):
    x =scale_factor*x
    return np.array([[x]])

def actual_pred(predict_file): 
    B=predict_file
    pred_lst=[]  
    for i in range(B.shape[0]):
        pred=B[i,:]
        pred = torch.from_numpy(pred)
        x=torch.topk(pred,1)[1].item()
        pred_lst.append(x)
    return pred_lst

def early_stop_best_valid_model(outer_folder,par,patience_number,saving_rate):
    valid_loss_values = load_obj('Fold_{}/{}/loss_figs/pair_{}/valid_loss_values'.format(outer_folder,labeling,par))
    best_index=0
    moving_index=1
    p=0
    while best_index < len(valid_loss_values) and moving_index <len(valid_loss_values) and p<patience_number:
        if valid_loss_values[best_index] >= valid_loss_values[moving_index]:
            best_index = moving_index
            moving_index +=1 
            p=0
        else:
            p+=1
            moving_index +=1 
    model_list=os.listdir('Fold_{}/{}/models/pair_{}'.format(outer_folder,labeling,par))
    model_list =[item+'net.pkl' for item in sorted([item.split('net.pkl')[0] for item in model_list],key=int)]
    model_name = model_list[int(best_index/saving_rate)]
    if os.path.exists('Fold_{}/{}/best_model/pair_{}.txt'.format(outer_folder,labeling,par)):
        os.remove('Fold_{}/{}/best_model/pair_{}.txt'.format(outer_folder,labeling,par))
    f=open('Fold_{}/{}/best_model/pair_{}.txt'.format(outer_folder,labeling,par),'a')
    line = 'best_index: '+str(best_index)+'\n'
    f.write(line)
    line = 'model_name: '+str(model_name)+'\n'
    f.write(line)
    f.close()
    print(best_index)
    print(model_name)
    return model_name


def early_stop_folders(outer_folder,par):

    if not os.path.exists('Fold_{}/{}/labels/{}'.format(outer_folder,labeling,par)):
        os.mkdir('Fold_{}/{}/labels/{}'.format(outer_folder,labeling,par))
    if not os.path.exists('Fold_{}/{}/labels/{}/early_stop'.format(outer_folder,labeling,par)):
        os.mkdir('Fold_{}/{}/labels/{}/early_stop'.format(outer_folder,labeling,par))
    if not os.path.exists('Fold_{}/{}/labels/{}/early_stop/outer'.format(outer_folder,labeling,par)):
        os.mkdir('Fold_{}/{}/labels/{}/early_stop/outer'.format(outer_folder,labeling,par))
    if not os.path.exists('Fold_{}/{}/labels/{}/early_stop/dist'.format(outer_folder,labeling,par)):
        os.mkdir('Fold_{}/{}/labels/{}/early_stop/dist'.format(outer_folder,labeling,par))

    if not os.path.exists('Fold_{}/{}/best_model'.format(outer_folder,labeling)):
        os.mkdir('Fold_{}/{}/best_model'.format(outer_folder,labeling))


def early_stop_operations(outer_folder,par,patience_number=1000,saving_rate=10):
    early_stop_folders(outer_folder,par)

    model_name = early_stop_best_valid_model(outer_folder,par,patience_number,saving_rate)
    net = torch.load('Fold_{}/{}/models/pair_{}/{}'.format(outer_folder,labeling,par,model_name))
    net = net.cuda()
    net.eval()
    model_name = model_name.split('.pkl')[0]

    n = len(list(open('./data_split/'+labeling+'/{}_test.txt'.format(outer_folder))))   
    predict_array = np.empty([n])
    gt_label = np.empty([n])
    data_path = '/media/super-server/6f0518ce-dbdf-4676-ba55-6739f8f3ecd4/Eli/Pain_ConfidentialData/Processed_data/'
    dst = ASL_Dataloader(data_path,outer_folder, is_train = False, is_transform=True, img_size=112, duration = 128,fold_number='outer', batch_size=1)
    testloader = torch.utils.data.DataLoader(dst, batch_size=1, shuffle=False,num_workers=12)    
    step_index = 0
    for i, data in enumerate(testloader):
        video_clip, label = data
        label = label[:,0]
        video_clip, label = Variable(video_clip).cuda(), Variable(label).cuda()        
        pred = net(video_clip)
        pred_list = pred.data.cpu().numpy()
        predict_array[step_index] = rev_norm_func(pred_list[0][0])
        gt_label[step_index] = rev_norm_func(label.item())
        step_index = step_index + 1
    MAE_score = np.mean(np.abs(gt_label - predict_array))
    MSE_score = np.mean(np.square(gt_label-predict_array))

    f=open('Fold_{}/{}/best_model/pair_{}.txt'.format(outer_folder,labeling,par),'a')
    line = 'outer_MAE_score: '+str(MAE_score)+'\n'
    f.write(line)
    line = 'outer_MSE_score: '+str(MSE_score)+'\n'
    f.write(line)
    f.close()

    label_path = 'Fold_{}/{}/labels/{}/early_stop/outer'.format(outer_folder,labeling,par)
    np.save(label_path+'/predict.np.npy', predict_array)
    np.save(label_path+'/label.np.npy', gt_label)

    n = len(list(open('data_split/'+labeling+'/final_distribution_{}_labels.txt'.format(labeling))))  
    predict_array = np.empty([n])
    gt_label = np.empty([n])
    data_path = '/media/super-server/6f0518ce-dbdf-4676-ba55-6739f8f3ecd4/Eli/Pain_ConfidentialData/Processed_data_dist/'
    dst = ASL_Dataloader(data_path,outer_folder, is_train = False, is_transform=True, img_size=112, duration = 128,fold_number='dist', batch_size=1)
    testloader = torch.utils.data.DataLoader(dst, batch_size=1, shuffle=False,num_workers=12)    
    step_index = 0
    for i, data in enumerate(testloader):
        video_clip, label = data
        label = label[:,0]
        video_clip, label = Variable(video_clip).cuda(), Variable(label).cuda()        
        pred = net(video_clip)
        pred_list = pred.data.cpu().numpy()
        predict_array[step_index] = rev_norm_func(pred_list[0][0])
        gt_label[step_index] = rev_norm_func(label.item())
        step_index = step_index + 1

    MAE_score = np.mean(np.abs(gt_label - predict_array))
    MSE_score = np.mean(np.square(gt_label-predict_array))

    f=open('Fold_{}/{}/best_model/pair_{}.txt'.format(outer_folder,labeling,par),'a')
    line = 'dist_MAE_score: '+str(MAE_score)+'\n'
    f.write(line)
    line = 'dist_MSE_score: '+str(MSE_score)+'\n'
    f.write(line)
    f.close()

    label_path = 'Fold_{}/{}/labels/{}/early_stop/dist'.format(outer_folder,labeling,par)
    np.save(label_path+'/predict.np.npy', predict_array)
    np.save(label_path+'/label.np.npy', gt_label)
