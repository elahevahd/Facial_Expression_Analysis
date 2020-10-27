import os 
import config
from config import *
import torch
import train_outer
import plot_main
from load_save_obj import * 
from plot_loss import * 
import confusion_matrix_code 
args = parser.parse_args()
from early_stop import * 
from mean_plot_data_early_stop import *
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


def train_early_stop_operations(outer_folder,par,des_path):
    # line=list(open('Fold_{}/{}/best_model/pair_{}.txt'.format(outer_folder,labeling,par)))[1]
    # model_name = line.strip().split('model_name: ')[1]

    model_list=os.listdir('Fold_{}/{}/models/pair_{}'.format(outer_folder,labeling,par))
    model_list =[item+'net.pkl' for item in sorted([item.split('net.pkl')[0] for item in model_list],key=int)]
    model_name = model_list[-1]
    net = torch.load('Fold_{}/{}/models/pair_{}/{}'.format(outer_folder,labeling,par,model_name))
    net = net.cuda()
    net.eval()
    model_name = model_name.split('.pkl')[0]

    n = len(list(open('./data_split/'+labeling+'/{}_1_train.txt'.format(outer_folder))))   
    predict_array = np.empty([n])
    gt_label = np.empty([n])

    data_path = '/media/super-server/6f0518ce-dbdf-4676-ba55-6739f8f3ecd4/Eli/Pain_ConfidentialData/Processed_data/'
    dst = ASL_Dataloader(data_path,outer_folder, is_train = False, is_transform=True, img_size=112, duration = 128,fold_number='outer_train', batch_size=1)
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

    obj_1= metric_calculator(gt_label, predict_array)
    fig, ax = plt.subplots()
    plt.figure(figsize=(15,5))
    plt.subplot(131)
    obj_1.draw_MAE()
    plt.subplot(132)
    obj_1.draw_RMSE() 
    plt.subplot(133)
    obj_1.draw_MSE() 
    plt.savefig(des_path,bbox_inches='tight')



# for outer_folder in ['1']:

#     for par in range(6):

#         if not os.path.exists('Fold_{}/VAS/plots/train_plot'.format(outer_folder)):
#             os.mkdir('Fold_{}/VAS/plots/train_plot'.format(outer_folder))

#         des_path = 'Fold_{}/VAS/plots/train_plot/{}.png'.format(outer_folder,par)

#         train_early_stop_operations(outer_folder,par,des_path)

#         print('Finished ------------------------------------------------')
#         torch.cuda.empty_cache() 

for outer_folder in ['3','4','5']:

    for par in range(2):

        if not os.path.exists('Fold_{}/VAS/plots/train_plot'.format(outer_folder)):
            os.mkdir('Fold_{}/VAS/plots/train_plot'.format(outer_folder))

        des_path = 'Fold_{}/VAS/plots/train_plot/{}.png'.format(outer_folder,par)

        train_early_stop_operations(outer_folder,par,des_path)


        print('Finished ------------------------------------------------')
        torch.cuda.empty_cache() 


