"generate adversarial examples of FGSM/PGD/DeepFool. GAN not include yet"

import os
os.sys.path.append('..')

import torch
import pandas as pd
from attacks.white_box.adv_box.attacks import FGSM, DeepFool, LinfPGDAttack
from train import models,utils_wf,utils_shs,utils_gan
import sys,copy



class gen_adv_x:
    def __init__(self,opts, x_box_min=-1,x_box_max=0,pert_box=0.3):

        self.opts = opts
        self.mode = opts['mode']
        self.model_path = '../model/' + self.mode
        self.data_path = '../data/' + self.mode
        self.pert_box = pert_box
        self.x_box_min = x_box_min
        self.x_box_max = x_box_max
        self.input_nc, self.gen_input_nc = 1, 1
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print("CUDA Available: ", torch.cuda.is_available())

        if self.mode == 'wf':
            print('Website Fingerprinting...')
        elif self.mode == 'shs':
            print('Smart Home Speaker Fingerprinting...')

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        "creat data folder"
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)


    def x_adv_gen(self,x, y, model, adversary):
        """
        for white-box based adv training, target model training should separate with adv examples generation.
        cause adv_training related to target model, and adv example generation involved target model.
        therefore, should set target model as evaluation mode when producing adv examples in adv training
        """
        model_cp = copy.deepcopy(model)
        for p in model_cp.parameters():
            p.requires_grad = False
        model_cp.eval()

        adversary.model = model_cp
        _,x_adv = adversary.perturbation(x, y,self.opts['alpha'])

        return x_adv


    def generate(self):
        "generate adv_x given x, append with its original label y instead with y_pert "

        "load data and target model"
        if self.mode == 'wf':
            "load data"
            train_data = utils_wf.load_data_main(self.opts['train_data_path'],self.opts['batch_size'])
            test_data = utils_wf.load_data_main(self.opts['test_data_path'],self.opts['batch_size'])

            "load target model structure"
            params = utils_wf.params(self.opts['num_class'],self.opts['input_size'])
            target_model = models.target_model_wf(params).to(self.device)

        elif self.mode == 'shs':
            "load data"
            train_data = utils_shs.load_data_main(self.opts['train_data_path'],self.opts['batch_size'])
            test_data = utils_shs.load_data_main(self.opts['test_data_path'],self.opts['batch_size'])

            "load target model structure"
            params = utils_shs.params(self.opts['num_class'],self.opts['input_size'])
            target_model = models.target_model_shs(params).to(self.device)

        else:
            print('mode not in ["wf","shs"], system will exit.')
            sys.exit()

        model_name = self.model_path + '/target_model.pth'
        target_model.load_state_dict(torch.load(model_name, map_location=self.device))
        target_model.eval()

        "set adversary"
        Adversary = self.opts['Adversary']

        if self.mode == 'wf':
            fgsm_epsilon = 0.1
            pgd_a = 0.01
        else:
            fgsm_epsilon = 0.1
            pgd_a = 0.01

        if Adversary == 'GAN':
            pretrained_generator_path = self.model_path + '/adv_generator.pth'
            pretrained_G = models.Generator(self.gen_input_nc, self.input_nc).to(self.device)
            pretrained_G.load_state_dict(torch.load(pretrained_generator_path, map_location=self.device))
            pretrained_G.eval()
        elif Adversary == 'FGSM':
            adversary = FGSM(self.mode, self.x_box_min, self.x_box_max, self.pert_box, epsilon=fgsm_epsilon)
        elif Adversary == 'DeepFool':
            adversary = DeepFool(self.mode, self.x_box_min, self.x_box_max, self.pert_box, num_classes=5)
        elif Adversary == 'PGD':
            adversary = LinfPGDAttack(self.mode, self.x_box_min, self.x_box_max, self.pert_box, k=5,a=pgd_a, random_start=False)


        "gen adv_x for test data"
        adv_xs = []
        labels = []
        for i,(x,y) in enumerate(test_data):
            print('generate adversary example of test data {} ...'.format(i))

            x, y = x.to(self.device), y.to(self.device)

            if Adversary == 'GAN':
                pert = pretrained_G(x)
                "cal adv_x given different mode wf/shs"
                adv_x = utils_gan.get_advX_gan(x, pert, mode, pert_box=self.opts['pert_box'],
                                               x_box_min=self.opts['x_box_min'],
                                               x_box_max=self.opts['x_box_max'], alpha=self.opts['alpha'])
            elif Adversary in ['FGSM', 'PGD','DeepFool']:
                _, y_pred = torch.max(target_model(x), 1)
                "cal adv_x given different mode wf/shs. the mode of adversary set before"
                adv_x = self.x_adv_gen(x, y_pred, target_model, adversary)

            else:
                print('No Adversary found! System will exit.')
                sys.exit()


            # x,y = x.to(self.device),y.to(self.device)
            # adversary.model = target_model
            # _,adv_x = adversary.perturbation(x,y,self.opts['alpha'])

            if self.mode == 'shs':
                adv_x = (adv_x.data.cpu().numpy().squeeze() * 1500).round()
            elif self.mode == 'wf':
                normalization = utils_wf.normalizer(x)
                adv_x = normalization.inverse_Normalizer(adv_x)
                adv_x = adv_x.squeeze()
            else:
                print('mode should in ["wf","shs"], system will exit.')
                sys.exit()

            adv_xs.append(adv_x)
            labels.append(y.data.cpu().numpy().squeeze())

        adv_xs = pd.DataFrame(adv_xs)
        adv_xs['label'] = labels
        output_path = self.data_path + '/adv_test_' + self.opts['Adversary'] + '.csv'
        adv_xs.to_csv(output_path,index=0)
        print('adversary examples of test data is generated')


        "gen adv_x for train data"
        adv_xs = []
        labels = []
        for i,(x,y) in enumerate(train_data):
            print('generate adversary example of train data {} ...'.format(i))

            # x,y = x.to(self.device),y.to(self.device)
            # adversary.model = target_model
            # _,adv_x = adversary.perturbation(x,y,self.opts['alpha'])

            x, y = x.to(self.device), y.to(self.device)

            if Adversary == 'GAN':
                pert = pretrained_G(x)
                "cal adv_x given different mode wf/shs"
                adv_x = utils_gan.get_advX_gan(x, pert, mode, pert_box=self.opts['pert_box'],
                                               x_box_min=self.opts['x_box_min'],
                                               x_box_max=self.opts['x_box_max'], alpha=self.opts['alpha'])
            elif Adversary in ['FGSM', 'PGD','DeepFool']:
                _, y_pred = torch.max(target_model(x), 1)
                "cal adv_x given different mode wf/shs. the mode of adversary set before"
                adv_x = self.x_adv_gen(x, y_pred, target_model, adversary)

            else:
                print('No Adversary found! System will exit.')
                sys.exit()


            if self.mode == 'shs':
                adv_x = (adv_x.data.cpu().numpy().squeeze()*1500).round()
            elif self.mode == 'wf':
                normalization = utils_wf.normalizer(x)
                adv_x = normalization.inverse_Normalizer(adv_x)
                adv_x = adv_x.squeeze()
            else:
                print('mode should in ["wf","shs"], system will exit.')
                sys.exit()

            adv_xs.append(adv_x)
            labels.append(y.data.cpu().numpy().squeeze())

        adv_xs = pd.DataFrame(adv_xs)
        adv_xs['label'] = labels
        output_path = self.data_path + '/adv_train_' + self.opts['Adversary'] + '.csv'
        adv_xs.to_csv(output_path, index=0)
        print('adversary examples of train data is generated')


def main(opts):

    gen_advX = gen_adv_x(opts)
    gen_advX.generate()




def get_opts_wf(mode,Adversary):
    return {
        'train_data_path': '../data/wf/train_NoDef_burst.csv',
        'test_data_path': '../data/wf/test_NoDef_burst.csv',
        'mode': mode,
        'alpha':1,
        'num_class': 95,
        'input_size': 512,
        'Adversary': Adversary,
        'batch_size': 1,
        'pert_box': 0.3,
        'x_box_min': -1,
        'x_box_max': 1,

    }


def get_opts_shs(mode,Adversary):
    return {
        'train_data_path': '../data/shs/traffic_train.csv',
        'test_data_path': '../data/shs/traffic_test.csv',
        'mode': mode,
        'alpha':None,
        'num_class': 101,
        'input_size': 256,
        'Adversary': Adversary,
        'batch_size': 1,
        'pert_box':0.3,
        'x_box_min':-1,
        'x_box_max':0,

    }




if __name__ == '__main__':

    adveraries = ['FGSM','PGD','GAN','DeepFool']
    mode = 'wf'

    for adv in adveraries:

        if mode == 'wf':
            opts = get_opts_wf(mode,adv)
        else:
            opts = get_opts_shs(mode,adv)

        main(opts)




