"""
#1: test target model against adversarial examples of test data
#2: test target model with adversarial training against adversarial examples of test data
"""

import os
os.sys.path.append('..')

import torch
from attacks.white_box.adv_box.attacks import FGSM,DeepFool,LinfPGDAttack
from train import utils_wf,utils_gan,utils_shs
from train import models
import sys


"""
adversarial examples of test data against target_model/adv_target_model
"""

class against_adv_x:
    def __init__(self,opts,x_box_min=-1,x_box_max=0,pert_box=0.3):
        self.opts = opts
        self.mode = opts['mode']
        self.adv_mode = opts['adv_mode']
        self.target_model_type = opts['target_model_type']
        self.model_path = '../model/' + self.mode
        self.pert_box = pert_box
        self.x_box_min = x_box_min
        self.x_box_max = x_box_max
        self.input_nc ,self.gen_input_nc = 1,1
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print("CUDA Available: ", torch.cuda.is_available())

        if self.mode == 'wf':
            print('Website Fingerprinting...')
        elif self.mode == 'shs':
            print('Smart Home Speaker Fingerprinting...')

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

    #------------------------------------------
    "performance of model"
    #------------------------------------------
    def test_model(self):

        "define test data type"
        if self.adv_mode == 'online':
            test_data_path = self.opts['test_data_path']
        elif self.adv_mode == 'offline':
            test_data_path = self.opts['adv_test_data_path']
        else:
            print('adv_mode should in ["online","offline"]; System will exit!')
            sys.exit()

        "load data and model"
        if self.mode == 'wf':
            "load data"
            test_data = utils_wf.load_data_main(test_data_path,self.opts['batch_size'])

            "load target model structure"
            params = utils_wf.params(self.opts['num_class'],self.opts['input_size'])
            target_model = models.target_model_wf(params).to(self.device)

        elif self.mode == 'shs':
            "load data"
            test_data = utils_shs.load_data_main(test_data_path,self.opts['batch_size'])

            "load target model structure"
            params = utils_shs.params(self.opts['num_class'],self.opts['input_size'])
            target_model = models.target_model_shs(params).to(self.device)
        else:
            print('mode not in ["wf","shs"], system will exit.')
            sys.exit()


        if self.target_model_type == 'adv_target_model':
            model_name = self.model_path + '/adv_target_model_' + self.opts['Adversary'] + '.pth'
        elif self.target_model_type == 'target_model':
            model_name = self.model_path + '/target_model.pth'
        else:
            print('target model type not in ["target_model","adv_target_model"], system will exit.')
            sys.exit()

        target_model.load_state_dict(torch.load(model_name, map_location=self.device))
        target_model.eval()


        "test on adversarial examples"
        correct_x = 0
        total_case = 0
        for i, data in enumerate(test_data, 0):
            test_x, test_y = data
            test_x, test_y = test_x.to(self.device), test_y.to(self.device)

            "prediction on original input x"
            pred_x = target_model(test_x)
            _,pred_x = torch.max(pred_x, 1)
            correct_x += (pred_x == test_y).sum()
            total_case += len(test_y)

        acc = float(correct_x.item()) / float(total_case)

        print('*'*30)
        print('"{}" with {} against {}.'.format(self.mode,self.opts['Adversary'],self.target_model_type) )
        print('correct test after attack is {}'.format(correct_x.item()))
        print('total test instances is {}'.format(total_case))
        print('accuracy of test after {} attack : correct/total= {:.5f}'.format(self.opts['Adversary'],acc))
        print('success rate of the attack is : {}'.format(1 - acc))
        print('\n')


def main(opts):

    against_adv = against_adv_x(opts)
    against_adv.test_model()


def get_opts_wf(Adversary,model_type):
    "parameters of website fingerprinting"
    return {
        'test_data_path': '../data/wf/test_NoDef_burst.csv',
        'adv_test_data_path': '../data/wf/adv_test_' + Adversary + '.csv',
        'target_model_type': model_type,
        'mode': 'wf',
        'adv_mode':'offline',
        'num_class': 95,
        'input_size': 512,
        'alpha': 10,
        'Adversary': Adversary,
        'batch_size': 64,
        'pert_box':0.3,
        'x_box_min':-1,
        'x_box_max':0,

    }


def get_opts_shs(Adversary,model_type):
    "parameters of smart home speaker fingerprinting"
    return {
        'test_data_path': '../data/shs/traffic_test.csv',
        'adv_test_data_path': '../data/shs/adv_test_' + Adversary + '.csv',
        'target_model_type': model_type,
        'mode': 'shs',
        'adv_mode': 'offline',
        'num_class': 101,
        'input_size': 256,
        'alpha': 1,
        'Adversary': Adversary,
        'batch_size': 64,
        'pert_box': 0.3,
        'x_box_min': -1,
        'x_box_max': 1,

    }


if __name__ == '__main__':

    Adversary = ['DeepFool','PGD','GAN','FGSM']

    mode = 'shs'
    model_type = 'adv_target_model'     # "model type includes: ['target_model','adv_target_model']"

    for adv in Adversary:
        if mode == 'wf':
            opts = get_opts_wf(adv,model_type)
        elif mode == 'shs':
            opts = get_opts_shs(adv,model_type)


        main(opts)
