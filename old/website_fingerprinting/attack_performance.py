import os
os.sys.path.append('..')

import torch
import time,sys
from white_box_attacks.adv_box.attacks import FGSM, DeepFool, LinfPGDAttack
from website_fingerprinting import models as models_wf
from website_fingerprinting import utils_wf
from advGAN import models as model_gan
from advGAN import utils_gan
import utils as utils_gb



# Adversary = 'GAN'
# batch_size = 64
# gen_input_nc = image_nc = 1
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#
# class config:
#     pert_clamp = 0.2
#     box_min,box_max = -1,0
#
#
# # set adversary
# "GAN adversary"
# if Adversary == 'GAN':
#     print('adv with GAN')
#     pretrained_generator_path = '../model/adv_generator.pth'
#     pretrained_G = model_gan.Generator(gen_input_nc, image_nc).to(device)
#     pretrained_G.load_state_dict(torch.load(pretrained_generator_path, map_location=device))
#     pretrained_G.eval()
# elif Adversary == 'FGSM':
#     print('adv testing with FGSM')
#     adversary = FGSM(epsilon=0.1)
# elif Adversary == 'DeepFool':
#     print('adv testing with DeepFool')
#     adversary = DeepFool(num_classes=5)
# elif Adversary == 'PGD':
#     print('adv testing with PGD')
#     adversary = LinfPGDAttack(k=5, random_start=False)
#
#
#
# "load target model"
# params = utils_wf.params()
# model = models.target_model(params)
# model.load_state_dict(torch.load('../model/wf_model/target_model.pth', map_location = device))
# model.to(device)
# model.eval()
#
#
# "load data"
# test = utils_wf.load_data_main('../data/NoDef/test_NoDef_burst.csv', batch_size)
# print('test_loader size: {}'.format(len(test)))
#
#
# "test performance"
# correct = 0
# correct_x = 0
# i = 0
# start_time = time.time()
# for data, label in test:
#     data, label = data.to(device),label.to(device)
#
#     pred_x = model(data)
#     pred_x = torch.max(pred_x, 1)    #return (max,index)
#     pred_x = pred_x[1]
#
#     if Adversary in ['FGSM','DeepFool','PGD']:
#         adversary.model = model
#         y_adv,x_adv = adversary.perturbation(data,label)
#
#     elif Adversary == 'GAN':
#         perturbation = pretrained_G(data)
#         perturbation = torch.clamp(perturbation, -config.pert_clamp, config.pert_clamp)
#         adv_x = perturbation + data
#         adv_x = torch.clamp(adv_x, config.box_min, config.box_max)
#         outputs = model(adv_x.to(device))
#         _, y_adv = torch.max(outputs.data, 1)  # retrun (max,index)
#
#     # label = label.to(device)
#     # outputs = model.forward(x_adv.to(device))
#     # _,y_adv = torch.max(outputs.data,1)   #retrun (max,index)
#
#     correct += (y_adv == label).sum()
#     correct_x += (pred_x == label).sum()
#
#     print('origial groudtruth label is: {}'.format(label.data.cpu().numpy()))
#     print('original predict laebl is: {}'.format(pred_x.data.cpu().numpy()))
#     print('adv label is: {}'.format(y_adv.data.cpu().numpy()))
#
#     i += 1
#
# end_time = time.time()
# time_diff = end_time - start_time
# total_case = len(test) * batch_size
# print('#'*20)
# print('average time is {} seconds'.format(time_diff / float(total_case)))
# print('total test is {}'.format(total_case))
# print('correct test after attack is {}'.format(correct))
# print('Accuracy of test after attack: correct/total= {:.5f}'.format(float(correct) / total_case))
# print('success rate of attack is: 1 - correct/total = {:.5f}'.format(1-(float(correct) / total_case)))
# print('model classfication accucary without being attacked is {:.5f}'.format(float(correct_x) / float(total_case)))



class attack_performance:
    def __init__(self,opts,pert_clamp=0.3,box_min=-1,box_max=0):

        slef.opts = opts
        self.mode = opts['mode']
        self.model_path = '../model/' + self.mode
        # self.batch_size = batch_size
        self.gen_input_nc = 1
        self.input_nc = 1
        self.pert_clamp = pert_clamp
        self.box_min = box_min
        self.box_max = box_max
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('CUDA AVAILABLE:', torch.cuda.is_available())

        if self.mode == 'wf':
            print('website fingerprinting...')
        elif self.mode == 'shs':
            print('smart home speaker fingerprinting...')
        else:
            print('mode should in ["wf","shs"], system will exit.')
            sys.exit()
        # self.test_data = utils_wf.load_data_main(input_data_path, self.batch_size)


    def against_adv_example(self):

        "set adversary"
        Adversary = self.opts['Adversary']
        if Adversary == 'GAN':
            pretrained_generator_path = self.model_path + '/adv_generator.pth'
            pretrained_G = model_gan.Generator(self.gen_input_nc, self.input_nc).to(self.device)
            pretrained_G.load_state_dict(torch.load(pretrained_generator_path, map_location=self.device))
            pretrained_G.eval()
        elif Adversary == 'FGSM':
            adversary = FGSM(self.mode,self.box_min,self.box_max,epsilon=0.1)
        elif Adversary == 'DeepFool':
            adversary = DeepFool(self.mode,self.box_min,self.box_max,num_classes=5)
        elif Adversary == 'PGD':
            adversary = LinfPGDAttack(self.mode,self.box_min,self.box_max,k=5, random_start=False)

        "load target model"
        params = utils_wf.params(num_class,input_size)
        model = models.target_model(params)
        model.load_state_dict(torch.load(self.model_path + '/target_model.pth', map_location = self.device))
        model.eval()

        "load data"
        if self.mode == 'wf':
            test_data = utils_wf.load_data_main(self.opts['test_data_path'], self.opts['test_batch_size'])
        elif self.mode == 'shs':
            test_data = utils_gb.load_data_main(self.opts['test_data_path'], self.opts['test_batch_size'])

        "load target model structure"
        if self.mode == 'wf':   # website fingerprinting
            params = utils_wf.params(self.opts['num_class'],self.opts['input_size'])
            target_model = models_wf.target_model(params).to(self.device)
        elif self.mode == 'shs': #'smart home speaker
            params = utils_gb.params(self.opts['num_class'],self.opts['input_size'])
            target_model = models_gan.target_model_1(params).to(self.device)

        #---------------------
        "test performance"
        #---------------------
        total_case = 0
        correct = 0
        correct_x = 0
        start_time = time.time()
        for data, label in test_data:
            data, label = data.to(self.device),label.to(self.device)

            pred_x = model(data)
            pred_x = torch.max(pred_x, 1)    #return (max,index)
            pred_x = pred_x[1]

            if self.Adversary in ['FGSM','DeepFool','PGD']:
                adversary.model = model
                y_adv,x_adv = adversary.perturbation(data,label)

            elif self.Adversary == 'GAN':
                perturbation = pretrained_G(data)
                adv_x = utils_gan.get_advX_gan(data,pert,self.mode)
                outputs = model(adv_x.to(self.device))
                _, y_adv = torch.max(outputs.data, 1)  # retrun (max,index)

            correct += (y_adv == label).sum()
            correct_x += (pred_x == label).sum()
            total_case += len(label)

        end_time = time.time()
        time_diff = end_time - start_time

        print('average time is {} seconds'.format(time_diff / float(total_case)))
        print('total test instances is {}'.format(total_case))
        print('correct test after attack is {}'.format(correct))
        print('accuracy of test after {} attack : correct/total= {:.5f}'.format(self.Adversary,(float(correct) / total_case)))
        print('success rate of attack under {} is: 1 - correct/total = {:.5f}'.format(self.Adversary,(1-(float(correct) / total_case))))
        print('accucary without being attacked is {:.5f}'.format(float(correct_x) / float(total_case)))
        print('\n')


def main(opts):

    "set batch_szie, deepfool only work at batch_size=1"
    if adv == 'DeepFool':
        batch_size = 1
        print('batch_size {}'.format(batch_size))
    else:
        batch_size = 64
        print('batch_size {}'.format(batch_size))
        pass

    "test performance"
    attack_perf = attack_performance(opts)
    attack_perf.against_adv_example()


def get_opts(Adversary):
    retrun{
    'mode': 'wf',
    'Adversary':Adversary,
    'num_class':95,
    'input_size':512,
    'batch_size':64,
    'test_batch_size':64,
    'test_data_path':'../data/NoDef/test_NoDef_burst.csv'
    }


if __name__ =='__main__':

    Adversary = ['FGSM','PGD','DeepFool','GAN']

    for i,adv in enumerate(Adversary):
        print('#' * 20)
        print('{} against adversary {}'.format(i+1,adv))

        opts = get_opts(adv)
        main(opts)
