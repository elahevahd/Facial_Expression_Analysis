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
from create_pres__early_stop import * 

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


def main_function_experiment(patience_number):

    for outer_folder in ['3','4','5']:

        print('************************** outer_folder '+ outer_folder)
        number_of_pars=len(params['lr'])*len(params['weight_decay'])*len(params['dec_freq']*len(params['dropout_value']))

        for index in range(number_of_pars):

            lr_value,weight_decay_value,dec_freq,dropout_value=[(x,y,z,k) for x in params['lr'] for y in params['weight_decay']  for z in params['dec_freq'] for k in params['dropout_value']][index]

            os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
            args.cuda = not args.no_cuda and torch.cuda.is_available()
            torch.manual_seed(args.seed)
            if args.cuda:
                torch.cuda.manual_seed(args.seed)

            print(outer_folder)

            # config.create_folders(outer_folder)

            inner_fold='1'

            if outer_folder == '3' and index==0:
                pass 
            else:
                print('Train Outer:')
                train_outer.training(inner_fold,dropout_value,lr_value,weight_decay_value,dec_freq,str(index),outer_folder, args)

            print('Find early stop model')
            early_stop_operations(outer_folder,index,patience_number,saving_rate=10)

            print('Drawing the plots:')
            plot_path = 'Fold_{}/{}/plots/loss/{}_outer.png'.format(outer_folder,labeling,str(index))
            plot_loss_trainval(str(index), plot_path,lr_value,weight_decay_value,dec_freq,outer_folder)

            plot_main.all_plots(inner_fold,outer_folder,index)

            print('Draw confusion matrix: ')
            confusion_matrix_code.generate_conf_matrix(outer_folder,index)
            confusion_matrix_code.dist_generate_conf_matrix(outer_folder,index)

            print('Finished ------------------------------------------------')
            torch.cuda.empty_cache() 

    # draw_all_means()
    # slides_experiment(patience_number)

main_function_experiment(1000)


# for patience_number in [100,150,200,250]:
#     print('*************** patience_number ********'+str(patience_number))
#     main_function_experiment(patience_number)



