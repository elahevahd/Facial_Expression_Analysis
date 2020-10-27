import os
import collections
import json
import torch
import torchvision
import torchvision.transforms as transforms
from videotransforms import video_transforms, volume_transforms
import numpy as np
import PIL.Image as Image
import scipy.misc as m
import scipy.io as io
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
from torch.utils import data
from PIL import Image
import os
import os.path
import cv2
import PIL
from PIL import Image
import h5py
import scipy
import random
import math
import config
from config import * 

labeling=config.labeling
scale_factor= config.scale_factor

def norm_func(x):
    x = x/scale_factor 
    return x

def balance_data_freq(fold_number, batch_size,outer_folder):

    if labeling== 'VAS':
        gr_dict= {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0, '9': 0, '10': 0}
        int_label = 10
    elif labeling == 'OPR':
        gr_dict= {'0': 0, '1': 0, '2': 0, '3': 0}  
        int_label = 3  

    if fold_number == 'outer':
        filename = './data_split/'+labeling+'/{}_train.txt'.format(outer_folder)
    else:
        filename = './data_split/'+labeling+'/{}_{}_train.txt'.format(outer_folder,str(fold_number))

    for line in list(open(filename)):
        label = line.strip().split('/')[-1].split('-')[-1]
        gr_dict[label] +=1 
    file_length= len(list(open(filename)))
    freq_dict = dict([(key, max(1,int(gr_dict[key]*batch_size/file_length))) for key in gr_dict.keys()])
    bool_var = (sum(freq_dict.values())==batch_size)
    while not bool_var and int_label>-1:
        freq_dict[str(int_label)]+=1 
        bool_var = (sum(freq_dict.values())==batch_size)
        int_label -= 1 
    if int_label==-1 and not bool_var:
        while not bool_var:
            freq_dict[str(0)]+=1 
            bool_var = (sum(freq_dict.values())==batch_size)            
    return freq_dict



def balance_data(fold_number, batch_size,outer_folder):
    if fold_number == 'outer':
        filename = './data_split/'+labeling+'/{}_train.txt'.format(outer_folder)
    else:
        filename = './data_split/'+labeling+'/{}_{}_train.txt'.format(outer_folder,str(fold_number))

    file_length= len(list(open(filename)))
    iter_num = math.ceil(file_length/batch_size)
    freq_dict = balance_data_freq(fold_number, batch_size,outer_folder)
    
    input_list = list( open(filename))
    random.shuffle(input_list)  #********* SHUFFLE

    if labeling== 'VAS':
        d={'0': [],'1': [],'2': [],'3': [],'4': [], '5': [],'6': [],'7': [],'8': [],'9': [],'10': []}
    elif labeling == 'OPR':
        d={'0': [],'1': [],'2': [],'3': []}   #pain levels 

    for line in input_list:
        label = line.strip().split('/')[-1].split('-')[-1]
        d[label].append(line)
    additive_label= dict([(label, max(0,iter_num* freq_dict[label] - len(d[label]))) for label in d.keys()])

    for label in d.keys():
        if additive_label[label] <= len(d[label]):
            additive_num = additive_label[label]
            tail_list = d[label][0:additive_num]
            for item in tail_list:
                d[label].append(item)
        else:
            x = len(d[label])
            while additive_label[label] > 0:
                tail_list = d[label]
                for item in tail_list[0:x]:
                    d[label].append(item)
                additive_label[label] -= x

    final_Data=[]
    for batch in range(iter_num):
        for key in freq_dict.keys():
            l = freq_dict[key]
            start = l*batch 
            end = l*(batch+1)
            for item in d[key][start:end]:
                final_Data.append(item)
    return final_Data


class ASL_Dataloader(data.Dataset):
    def __init__(self, root,outer_folder, is_train = True, is_transform=True, img_size=112, duration = 16,epoch=1, fold_number=1, batch_size=35):
        self.root = root
        self.img_size = img_size
        self.is_train = is_train
        self.is_transform = is_transform
        self.epoch = epoch
        self.fold_number = fold_number
        self.batch_size = batch_size
        self.outer_folder = outer_folder
        if self.is_train:
            self.video_transform_list = video_transforms.Compose(
                [
                    video_transforms.Resize((112,112)), 
                    video_transforms.RandomHorizontalFlip(),
                    video_transforms.RandomRotation(5),
                    # video_transforms.CenterCrop((self.img_size , self.img_size )), #added
                    # video_transforms.RandomCrop((54, 54)),
                    volume_transforms.ClipToTensor(),
                    video_transforms.Normalize((0.52169, 0.37033, 0.35018), (0.15437, 0.1288, 0.12234))
                ]           
                )
        else:
            self.video_transform_list = video_transforms.Compose(
                [
                    video_transforms.Resize((112,112)),
                    # video_transforms.CenterCrop((self.img_size , self.img_size )),
                    volume_transforms.ClipToTensor(),
                    video_transforms.Normalize((0.52169, 0.37033, 0.35018), (0.15437, 0.1288, 0.12234))
                ]
                )
        self.label_transform = transforms.Compose([transforms.ToTensor()])
        self.mask_transform = transforms.Compose([transforms.ToTensor()])
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.temporal_duration = duration

        #########################################################################
        # load the image feature and image name 
        ##########################################################################
        
        if self.is_train:
            vid_list = balance_data(self.fold_number,self.batch_size,self.outer_folder)
        else:
            if self.fold_number == 'outer':
                vid_list = open('./data_split/'+labeling+'/{}_test.txt'.format(self.outer_folder))   
            elif self.fold_number == 'dist':
                vid_list = open('./data_split/'+labeling+'/final_distribution_{}_labels.txt'.format(labeling))  
            elif self.fold_number == 'outer_train':
                vid_list = open('./data_split/'+labeling+'/{}_1_train.txt'.format(self.outer_folder))  
            else:
                vid_list = open('./data_split/'+labeling+'/{}_{}_valid.txt'.format(self.outer_folder,str(self.fold_number)))       

            vid_list = list(vid_list)
    

        self.data_list = [] 
        self.start_frame = []
        self.end_frame = []
        self.label_list = []
        for line in vid_list:
            line = line.strip('\r\n')
            line_list = line.split('/')
            label = int(line_list[-1].split('-')[-1])
            normed_label = norm_func(label)
            start = int(line_list[-1].split('-')[-3])
            end = int(line_list[-1].split('-')[-2])
            path = line_list[0]+'/'+line_list[1]
            self.data_list.append(path)
            self.start_frame.append(start)
            self.end_frame.append(end)
            self.label_list.append(normed_label)

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, index):
        video_path = self.root + self.data_list[index]
        start_frame = self.start_frame[index]
        end_frame = self.end_frame[index]
        label = self.label_list[index]

        if self.is_train:
            start = random.randint(0,10)
            end = random.randint(0,10)
            
            if start_frame+start > end_frame-end:
                img_index = np.linspace(start_frame,end_frame,self.temporal_duration)
            else:
                img_index = np.linspace(start_frame+start,end_frame-end,self.temporal_duration)
        else:
            img_index = np.linspace(start_frame,end_frame,self.temporal_duration)

        video_clip = []
        for index in range(self.temporal_duration):
            num = math.floor(img_index[index])
            num = int(num)
            img_name = '/img_'+str(num) + '.png'
            img_path = video_path + img_name
            img = Image.open(img_path).convert('RGB')
            video_clip.append(img)

        video_clip = self.video_transform_list(video_clip)
        label = [label]
        label = torch.tensor(np.array(label))
        return video_clip, label

