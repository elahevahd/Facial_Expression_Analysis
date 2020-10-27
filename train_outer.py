from __future__ import print_function
from __future__ import division
import argparse
from load_save_obj import * 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from numpy.random import normal
from numpy.linalg import svd
from math import sqrt
import numpy as np
from numpy import array
from collections import Counter
import math
import os
import matplotlib.pyplot as plt
import random
from torch.utils import data
import torchvision
from resnet import resnet34, resnet10, resnet18, resnet50, resnet101, resnet152
from dataloader import ASL_Dataloader
from dataloader import balance_data_freq
import time
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from config import *
from  importlib import reload
import plot_loss
reload(plot_loss)
from plot_loss import *  

import config
from config import * 
labeling=config.labeling
pain_levels= config.pain_levels
epoch_num = config.epoch
scale_factor= config.scale_factor

parser = argparse.ArgumentParser(description='Video Super Resolution With Generative Advresial Network')

parser.add_argument('--batch_size', type=int, default=30, metavar='N',
                    help='input batch size for training (default: 14)')

parser.add_argument('--iteration', type=int, default=40000, metavar='N',
                    help='number of iterations to train (default: 40000)')

parser.add_argument('--temporal_duration', type=int, default=128, metavar='TS',
                    help=' number (default: 64)')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA taining')

parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')

parser.add_argument('--save', type=str,  default='models/',
                    help='path to save the final model')

parser.add_argument('--log', type=str,  default='log/',
                    help='path to the log information')

parser.add_argument('--gpu_id', type=str,  default='0,1,2,3',
                    help='GPU used to train the network')



def training(fold_number,dropout_value, lr_value,weight_decay_value,dec_freq,par,outer_folder, args):


    if not os.path.exists('Fold_{}/{}/loss_figs/'.format(outer_folder,labeling)):
        os.mkdir('Fold_{}/{}/loss_figs/'.format(outer_folder,labeling))
    if not os.path.exists('Fold_{}/{}/loss_figs/pair_{}'.format(outer_folder,labeling,par)):
        os.mkdir('Fold_{}/{}/loss_figs/pair_{}'.format(outer_folder,labeling,par))

    net = resnet34(num_classes = pain_levels, shortcut_type = 'A', sample_size = 112, sample_duration = 128,dropout_rate = dropout_value)
    net = net.cuda()
    print('--------------------        load the pretrained model        --------------------------------------')
    pretrain = torch.load('./pre_trained/Chalearn_34_layers_64_frames/65000net.pkl')
    saved_state_dict = pretrain.state_dict()
    print('----------------------------------------------------------')
    new_params = net.state_dict().copy()
    for name, param in new_params.items():
        if name in saved_state_dict and param.size() == saved_state_dict[name].size():
            new_params[name].copy_(saved_state_dict[name])
            print('copying ' + name + '  from   ' + 'module.'+name)
    net.load_state_dict(new_params)
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
    net = net.to('cuda')

    optimizer = optim.SGD(
            net.parameters(),
            lr=lr_value,
            momentum=0.95,
            weight_decay=weight_decay_value)

    # numb_vids = len(list(open('data_split/'+labeling+'/{}_train.txt'.format(outer_folder))))
    numb_vids = len(list(open('data_split/'+labeling+'/{}_{}_train.txt'.format(outer_folder,fold_number))))

    # valid_numb_vids = len(list(open('data_split/'+labeling+'/{}_test.txt'.format(outer_folder))))
    valid_numb_vids = len(list(open('data_split/'+labeling+'/{}_{}_valid.txt'.format(outer_folder,fold_number))))

    print('Training size ' +str(numb_vids)+' Validation size '+str(valid_numb_vids))

    iter_num = math.ceil(numb_vids/args.batch_size)

    criterion = nn.MSELoss()
    valid_criterion = nn.MSELoss()

    if not os.path.exists('Fold_{}/{}/models/'.format(outer_folder,labeling)):
        os.mkdir('Fold_{}/{}/models/'.format(outer_folder,labeling))
    if not os.path.exists('Fold_{}/{}/models/pair_{}'.format(outer_folder,labeling,par)):
        os.mkdir('Fold_{}/{}/models/pair_{}'.format(outer_folder,labeling,par))

    step_index = 0
    train_loss_values=[]
    valid_loss_values=[]

    for epoch in range(epoch_num):
        
        net.train(True)

        train_running_loss = 0.0
        dst = ASL_Dataloader(data_path,outer_folder, is_transform=True, img_size=112, duration = 128, epoch=epoch, fold_number=fold_number, batch_size=args.batch_size)
        trainloader =  torch.utils.data.DataLoader(dst, batch_size=args.batch_size, shuffle=False, num_workers=12, drop_last=True)
        
        for i, data in enumerate(trainloader):
            start_time = time.time()
            step_index = step_index + 1
            video_clip, label = data
            # label = label[:,0]
            video_clip, label = Variable(video_clip).cuda(),  Variable(label).cuda()            

            net.zero_grad()
            pred = net(video_clip)

            loss = criterion(pred,label.float())

            train_running_loss += loss.item() 
            print('[%d][%d] loss: %f  time: %f ' % (epoch, step_index, loss.item(), time.time() - start_time))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters() , max_norm=3)  #***** new 
            optimizer.step()

        print('step index: '+str(step_index))
        if(epoch % 10) ==0:
            print('----------------- Save The Network ------------------------\n')
            with open('Fold_{}/{}/models/pair_{}/{}net.pkl'.format(outer_folder,labeling,par,epoch), 'wb') as f:
                torch.save(net, f)

        train_loss_values.append(train_running_loss / iter_num)


        net.eval()
        with torch.no_grad():

            valid_running_loss = 0.0
            validation_dst = ASL_Dataloader(data_path,outer_folder, is_train = False, is_transform=True, img_size=112, duration = 128, epoch=epoch, fold_number=fold_number, batch_size=1)
            validation_loader = torch.utils.data.DataLoader(validation_dst, batch_size=1, shuffle=False,num_workers=12) 
            for i, data in enumerate(validation_loader):
                video_clip, valid_label = data
                # valid_label = valid_label[:,0]
                video_clip, valid_label = Variable(video_clip).cuda(),  Variable(valid_label).cuda()            
                valid_pred = net(video_clip)

                valid_loss = valid_criterion(valid_pred,valid_label.float())

                valid_running_loss += valid_loss.item() 

            valid_loss_values.append(valid_running_loss/valid_numb_vids)

            save_obj(train_loss_values,'Fold_{}/{}/loss_figs/pair_{}/train_loss_values'.format(outer_folder,labeling,par))
            save_obj(valid_loss_values,'Fold_{}/{}/loss_figs/pair_{}/valid_loss_values'.format(outer_folder,labeling,par))

            plot_path = 'Fold_{}/{}/plots/loss/{}_outer.png'.format(outer_folder,labeling,str(par))
            plot_loss_trainval(str(par), plot_path,lr_value,weight_decay_value,dec_freq,outer_folder)

    print('----------------- Save The Last Network ------------------------\n')
    with open('Fold_{}/{}/models/pair_{}/{}net.pkl'.format(outer_folder,labeling,par,step_index+1), 'wb') as f:
        torch.save(net, f)    

        
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


# reload(plot_loss)
# from plot_loss import *  
# lr_value,weight_decay_value,dec_freq=(0.0001, 0.0001, 1)
# plot_path = 'Fold_{}/{}/plots/loss/{}_outer.png'.format('1','OPR',str(0))
# plot_loss_trainval('outer',str(0), plot_path,lr_value,weight_decay_value,dec_freq,'1')
