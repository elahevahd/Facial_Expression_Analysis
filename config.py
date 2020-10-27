import os 
import argparse

parser = argparse.ArgumentParser(description='Video Super Resolution With Generative Advresial Network')

parser.add_argument('--batch_size', type=int, default=30, metavar='N',
                    help='input batch size for training (default: 14)')

parser.add_argument('--temporal_duration', type=int, default=128, metavar='TS',
                    help=' number (default: 64)')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA taining')

parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

parser.add_argument('--gpu_id', type=str,  default='0,1,2,3',
                    help='GPU used to train the network')

labeling = 'VAS'


data_path = '/media/super-server/6f0518ce-dbdf-4676-ba55-6739f8f3ecd4/Eli/Pain_ConfidentialData/Processed_data/'
dist_plot_distribution_path = 'mean_results/dist_data.png'
plot_distribution_all_path = 'mean_results/extension_data.png'

epoch= 500

if labeling== 'VAS':
    pain_levels = 11  # 0,1,...,10
    scale_factor = 10
elif labeling == 'OPR':
    pain_levels = 4  # 0,1,2,3
    scale_factor = 3


def create_folders(outer_folder):

    if not os.path.exists('mean_results'):
        os.mkdir('mean_results')
    if not os.path.exists('mean_results/'+labeling):
        os.mkdir('mean_results/'+labeling)

    if not os.path.exists('Fold_{}'.format(outer_folder)):
        os.mkdir('Fold_{}'.format(outer_folder))

    if not os.path.exists('Fold_{}/{}'.format(outer_folder,labeling)):
        os.mkdir('Fold_{}/{}'.format(outer_folder,labeling))

    if not os.path.exists('Fold_{}/{}/labels'.format(outer_folder,labeling)):
        os.mkdir('Fold_{}/{}/labels'.format(outer_folder,labeling))
 
    if not os.path.exists('Fold_{}/{}/models'.format(outer_folder,labeling)):
        os.mkdir('Fold_{}/{}/models'.format(outer_folder,labeling))

    if not os.path.exists('Fold_{}/{}/loss_figs'.format(outer_folder,labeling)):
        os.mkdir('Fold_{}/{}/loss_figs'.format(outer_folder,labeling))

    if not os.path.exists('Fold_{}/{}/plots'.format(outer_folder,labeling)):
        os.mkdir('Fold_{}/{}/plots'.format(outer_folder,labeling))

    if not os.path.exists('Fold_{}/{}/plots/distribution'.format(outer_folder,labeling)):
        os.mkdir('Fold_{}/{}/plots/distribution'.format(outer_folder,labeling))

    if not os.path.exists('Fold_{}/{}/plots/scores'.format(outer_folder,labeling)):
        os.mkdir('Fold_{}/{}/plots/scores'.format(outer_folder,labeling))

    if not os.path.exists('Fold_{}/{}/plots/loss'.format(outer_folder,labeling)):
        os.mkdir('Fold_{}/{}/plots/loss'.format(outer_folder,labeling))

    if not os.path.exists('Fold_{}/{}/plots/confusion_matrix'.format(outer_folder,labeling)):
        os.mkdir('Fold_{}/{}/plots/confusion_matrix'.format(outer_folder,labeling))

