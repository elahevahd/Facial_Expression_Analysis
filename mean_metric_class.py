import sklearn
import  numpy as np 
import torch
import itertools
import sklearn
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
import math

import config
labeling=config.labeling
pain_levels= config.pain_levels

class metric_calculator():
    def __init__(self,labels_arr,predict_arr):
        self.labels_arr = labels_arr
        self.predict_arr = predict_arr

        if labeling== 'VAS':
            self.pain_dict={'0': [], '1': [], '2': [], '3': [], '4': [], '5': [], '6': [], '7': [], '8': [], '9': [], '10': []}
            self.MSE_step = 5
        elif labeling == 'OPR':
            self.MSE_step = 1 
            self.pain_dict={'0': [], '1': [], '2': [], '3': []}

        n = self.labels_arr.shape[0]
        for index in range(n):
            key = str(int(self.labels_arr[index]))
            self.pain_dict[key].append(self.predict_arr[index])

    def return_MAE_list(self):
        MAE_list=[]
        for key in self.pain_dict.keys():
            pred_arr=np.array(self.pain_dict[key])
            m=len(pred_arr)
            label_arr=np.repeat(int(key),m)
            mean_value=np.mean(np.abs(label_arr - pred_arr))
            MAE_list.append(mean_value)
        return MAE_list

    def return_MAE_avg(self):
        actual = self.labels_arr
        predicted= self.predict_arr
        return np.mean(np.abs(actual - predicted))

    def draw_MAE(self):
        barlist = plt.bar(range(pain_levels),self.return_MAE_list())
        for i in range(pain_levels):
            barlist[i].set_color('r')
        plt.xlabel('Pain Level',size=15)
        plt.ylabel('MAE ',size=15)
        plt.xticks(np.arange(0, pain_levels, step=1))
        plt.yticks(np.arange(0, pain_levels, step=1))
        plt.legend(['MAE Avg = '+str(round(self.return_MAE_avg(),3))], loc='best')
 

    def return_MSE_list(self):
        MSE_list=[]
        for key in self.pain_dict.keys():
            pred_arr=np.array(self.pain_dict[key])
            m=len(pred_arr)
            label_arr=np.repeat(int(key),m)
            mean_value=np.mean(np.square(label_arr - pred_arr))
            MSE_list.append(mean_value)
        return MSE_list

    def return_MSE_avg(self):
        actual = self.labels_arr
        predicted= self.predict_arr
        return np.mean(np.square(actual-predicted))

    def draw_MSE(self):
        barlist = plt.bar(range(pain_levels),self.return_MSE_list())
        for i in range(pain_levels):
            barlist[i].set_color('r')
        plt.xlabel('Pain Level',size=15)
        plt.ylabel('MSE ',size=15)
        plt.xticks(np.arange(0, pain_levels, step=1))
        plt.yticks(np.arange(0, (pain_levels-1)**2+5, step=self.MSE_step))
        plt.legend(['MSE Avg = '+str(round(self.return_MSE_avg(),3))], loc='best')


    def return_RMSE_list(self):
        RMSE_list=[]
        for key in self.pain_dict.keys():
            pred_arr=np.array(self.pain_dict[key])
            m=len(pred_arr)
            label_arr=np.repeat(int(key),m)
            mean_value=np.mean(np.square(label_arr - pred_arr))
            RMSE_list.append(np.sqrt(mean_value))
        return RMSE_list

    def return_RMSE_avg(self):
        actual = self.labels_arr
        predicted= self.predict_arr
        return np.sqrt(np.mean(np.square(actual-predicted)))

    def draw_RMSE(self):
        barlist = plt.bar(range(pain_levels),self.return_RMSE_list())
        for i in range(pain_levels):
            barlist[i].set_color('r')
        plt.xlabel('Pain Level',size=15)
        plt.ylabel('RMSE ',size=15)
        plt.xticks(np.arange(0, pain_levels, step=1))
        plt.yticks(np.arange(0, pain_levels, step=1))
        plt.legend(['RMSE Avg = '+str(round(self.return_RMSE_avg(),3))], loc='best')


    def final_drawing(self):
        plt.figure(figsize=(15,5))
        plt.subplot(131) #subplot(nrows, ncols, index, **kwargs)
        self.draw_MAE()
        plt.subplot(132)
        self.draw_MSE()
        plt.subplot(133)
        self.draw_RMSE()

