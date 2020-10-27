from sklearn.metrics import confusion_matrix
import os 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

import config
labeling=config.labeling

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # print("Normalized confusion matrix")
    else:
        # print('Confusion matrix, without normalization')
        pass
    # print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax



###############################################################################
def generate_conf_matrix(outer_folder,index):

    label_path = 'Fold_{}/{}/labels/{}/early_stop/outer/label.np.npy'.format(outer_folder,labeling,index)
    pred_path = 'Fold_{}/{}/labels/{}/early_stop/outer/predict.np.npy'.format(outer_folder,labeling,index)

    label_list = list(np.load(label_path))
    pred_list = list(np.load(pred_path))
    label_list = [int(item) for item in label_list]
    pred_list= [int(round(item)) for item in pred_list]
    pred_list = [max(0,item) for item in pred_list]

    if labeling=='VAS':
        class_names = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    elif labeling=='OPR':
        class_names = np.array([0, 1, 2, 3])

    plot_confusion_matrix(label_list, pred_list, classes=class_names,title='Confusion matrix, without normalization')
    plt.savefig('Fold_{}/{}/plots/confusion_matrix/confusion_{}.png'.format(outer_folder,labeling,index),bbox_inches='tight')

    plot_confusion_matrix(label_list, pred_list, classes=class_names, normalize=True,title='Normalized confusion matrix')
    plt.savefig('Fold_{}/{}/plots/confusion_matrix/normed_confusion_{}.png'.format(outer_folder,labeling,index),bbox_inches='tight')

def dist_generate_conf_matrix(outer_folder,index):

    label_path = 'Fold_{}/{}/labels/{}/early_stop/dist/label.np.npy'.format(outer_folder,labeling,index)
    pred_path = 'Fold_{}/{}/labels/{}/early_stop/dist/predict.np.npy'.format(outer_folder,labeling,index)

    label_list = list(np.load(label_path))
    pred_list = list(np.load(pred_path))
    label_list = [int(item) for item in label_list]
    pred_list= [int(round(item)) for item in pred_list]
    pred_list = [max(0,item) for item in pred_list]

    if labeling=='VAS':
        class_names = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    elif labeling=='OPR':
        class_names = np.array([0, 1, 2, 3])

    plot_confusion_matrix(label_list, pred_list, classes=class_names,title='Confusion matrix, without normalization')
    plt.savefig('Fold_{}/{}/plots/confusion_matrix/dist_confusion_{}.png'.format(outer_folder,labeling,index),bbox_inches='tight')

    plot_confusion_matrix(label_list, pred_list, classes=class_names, normalize=True,title='Normalized confusion matrix')
    plt.savefig('Fold_{}/{}/plots/confusion_matrix/dist_normed_confusion_{}.png'.format(outer_folder,labeling,index),bbox_inches='tight')

