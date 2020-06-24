import os
os.sys.path.append('..')

import torch
from advGAN.advWGAN import advGan_attack
from advGAN.models import target_model_1 as target_model_shs
from website_fingerprinting.models import target_model as target_model_wf
from website_fingerprinting import utils_wf
from advGAN import utils_advGAN
import utils
import argparse, sys



# use_cuda=True
# input_nc=1
# epochs = 60
# x_box_min = -1
# x_box_max = 1
# pert_box_min = -0.3
# pert_box_max = 0.3
# num_class = 95
# input_size = 512
# mode = 'wf'  # 'wf': website_fingerprinting, 'shs':smart_home_speaker
#
# class opt:
#     batch_size = 64
#     test_batch_size = 1
#
# # Define what device we are using
# print("CUDA Available: ",torch.cuda.is_available())
# device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
#
# pretrained_model = "../model/wf_model/target_model.pth"
# if mode == 'wf':
#     params = utils_wf.params(num_class,input_size)
#     targeted_model = target_model_wf(params).to(device)
# elif mode == 'shs':
#     params = utils_advGAN.params(num_class)
#     targeted_model = target_model_shs(params).to(device)
#
# targeted_model.load_state_dict(torch.load(pretrained_model,map_location=device))
# targeted_model.eval()
# model_num_labels = 95
#
#
# # load data
# train_dataloader = utils.load_data_main('../data/NoDef/train_NoDef_burst.csv',opt.batch_size) # input_shape (batch_size,1,wide 256)
# # test_dataloader = utils.load_data_main('../traffic_test.csv',opt.test_batch_size)
#
# advGAN = advGan_attack(device,
#                        targeted_model,
#                        model_num_labels,
#                        input_nc,
#                        x_box_min,
#                        x_box_max,
#                        pert_box_min,pert_box_max)
#
# advGAN.train(train_dataloader, epochs)





#**********************************

class advGAN:
    def __init__(self,mode,x_box_min=-1,x_box_max=0,pert_box=0.3):

        self.input_nc = 1
        self.x_box_min = x_box_min
        self.x_box_max = x_box_max
        self.pert_box = pert_box
        self.mode = mode
        self.device = ('cuda:0' if torch.cuda.is_available() else 'cpu')
        print("CUDA Available: ", torch.cuda.is_available())

        if self.mode == 'wf':
            print('Website Fingerprinting...')
        elif self.mode == 'shs':
            print('Smart Home Speaker Fingerprinting...')



    def train_advGAN(self,epochs,batch_size,num_class,input_size,target_model_path,data_path):

        "load train data"
        train_dataloader = utils.load_data_main(data_path,batch_size)

        "load target model"

        if self.mode == 'wf':   # website fingerprinting
            params = utils_wf.params(num_class,input_size)
            target_model = target_model_wf(params).to(self.device)
        elif self.mode == 'shs': #'smart home speaker
            params = utils_advGAN.params(num_class,input_size)
            target_model = target_model_shs(params).to(self.device)

        target_model.load_state_dict(torch.load(target_model_path,map_location=self.device))
        target_model.eval()


        "train step"
        advGAN_train_process = advGan_attack(self.mode,
                                    target_model,
                                    num_class,
                                    self.pert_box,
                                    self.x_box_min,
                                    self.x_box_max,
                                    )

        advGAN_train_process.train(train_dataloader,epochs)



def main(opt):
    adversarialGAN = advGAN(opt['mode'],opt['x_box_min'],opt['x_box_max'],opt['pert_box'])
    adversarialGAN.train_advGAN(opt['epochs'],opt['batch_size'],opt['num_class'],opt['input_size'],
                                opt['target_model_path'],opt['input_data_path'])



def set_params():

    return{
    'epochs':50,
    'batch_size':64,
    'pert_box':0.3,

    #-----------------------------------
    # params for website fingerprinting (wf)
    #-----------------------------------

    'x_box_min': -1,
    'x_box_max':0,
    'num_class':95,
    'input_size':512,
    'mode': 'wf',
    'target_model_path':'../model/wf/target_model.pth',
    'input_data_path':'../data/NoDef/train_NoDef_burst.csv',



    # -----------------------------------
    # params for smart home speaker (shs)
    # -----------------------------------

    # 'x_box_min': -1,
    # 'x_box_max': 0,
    # 'num_class': 101,
    # 'input_size': 256,
    # 'mode': 'shs',   #shs and wf
    # 'target_model_path': '../model/target_model.pth',
    # 'input_data_path': '../data/traffic_train.csv',

    }




if __name__ == '__main__':


    params = set_params()
    main(params)
