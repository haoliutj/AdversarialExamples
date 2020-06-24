# import os
# os.sys.path.append('..')
#
# import torch
# import models
# from models import target_model_1
# import utils
# from white_box_attacks.adv_box.attacks import FGSM,DeepFool,LinfPGDAttack
#
#
#
#
# use_cuda=True
#
#
# class config:
#     image_nc = 1
#     gen_input_nc = image_nc
#     batch_size = 64
#     test_batch_size = 8
#     pert_clamp = 0.3
#     box_min,box_max = -1,0
#     adversary = 'GAN'
#
# # Define what device we are using
# print("CUDA Available: ",torch.cuda.is_available())
# device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
#
# # load the pretrained model
# pretrained_model = "../model/adv_target_model_GAN.pth"
# params = utils.params()
# target_model = target_model_1(params).to(device)
# target_model.load_state_dict(torch.load(pretrained_model,map_location=device))
# target_model.eval()
#
#
# # set adversary
# "GAN adversary"
# if config.adversary == 'GAN':
#     print('adv with GAN')
#     pretrained_generator_path = '../model/adv_generator.pth'
#     pretrained_G = models.Generator(config.gen_input_nc, config.image_nc).to(device)
#     pretrained_G.load_state_dict(torch.load(pretrained_generator_path, map_location=device))
#     pretrained_G.eval()
# elif config.adversary == 'FGSM':
#     print('adv with FGSM')
#     adversary = FGSM(epsilon=0.1)
# elif config.adversary == 'DeepFool':
#     print('adv with DeepFool')
#     adversary = DeepFool(num_classes=5)
#     # adversary = DeepFool_batch_train(num_classes=10)
# elif config.adversary == 'PGD':
#     print('adv with PGD')
#     adversary = LinfPGDAttack(k=5, random_start=False)
#
#
# #load data
# train_dataloader = utils.load_data_main('../data/traffic_train.csv', config.batch_size) # input_shape (batch_size,1,wide 256)
# test_dataloader = utils.load_data_main('../data/traffic_test.csv', config.test_batch_size)
#
#
#
# # test adversarial examples in testing dataset
# num_correct = 0
#
# for i, data in enumerate(test_dataloader, 0):
#     test_img, test_label = data
#     test_img, test_label = test_img.to(device), test_label.to(device)
#
#     if config.adversary in ['FGSM', 'DeepFool' , 'PGD']:
#         adversary.model = target_model
#         _, adv_x = adversary.perturbation(test_img, test_label)
#     elif config.adversary == 'GAN':
#         perturbation = pretrained_G(test_img)
#         perturbation = torch.clamp(perturbation, -config.pert_clamp, config.pert_clamp)
#         adv_x = perturbation + test_img
#         adv_x = torch.clamp(adv_x, config.box_min, config.box_max)
#
#     pred_lab = torch.argmax(target_model(adv_x.to(device)), 1)
#     num_correct += torch.sum(pred_lab==test_label,0)
#
# print('num_correct: ', num_correct.item())
# print('total_num: ', len(test_dataloader) * config.test_batch_size)
# print('accuracy of adv in testing set: %f\n' % (num_correct.item() / (len(test_dataloader) * config.test_batch_size)))
# print('success rate of adv in testing set: %f\n' % (1-(num_correct.item() / (len(test_dataloader) * config.test_batch_size))))






#*********************************************

import os
os.sys.path.append('..')

import torch
from white_box_attacks.adv_box.attacks import FGSM,DeepFool,LinfPGDAttack
from website_fingerprinting import utils_wf
from website_fingerprinting import models as models_wf
from advGAN import models as models_gan
from advGAN import utils_advGAN
import utils as utils_shs




class test_adv:
    def __init__(self,opts,x_box_min=-1,x_box_max=0,pert_box=0.3):
        self.opts = opts
        self.mode = opts['mode']
        self.pert_box = pert_box
        self.x_box_min = x_box_min
        self.x_box_max = x_box_max
        self.input_nc = 1
        self.gen_input_nc = 1
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print("CUDA Available: ", torch.cuda.is_available())

        if self.mode == 'wf':
            print('Website Fingerprinting...')
        elif self.mode == 'shs':
            print('Smart Home Speaker Fingerprinting...')


    def test_adv_performance(self):

        "load data"
        test_dataloader = utils_wf.load_data_main(self.opts['test_data_path'],self.opts['batch_size'])

        if self.mode == 'wf':
            test_dataloader = utils_wf.load_data_main(self.opts['test_data_path'], self.opts['test_batch_size'])
        elif self.mode == 'shs':
            test_dataloader = utils_shs.load_data_main(self.opts['test_data_path'], self.opts['test_batch_size'])

        "load target model structure"
        if self.mode == 'wf':  # website fingerprinting
            params = utils_wf.params(self.opts['num_class'], self.opts['input_size'])
            target_model = models_wf.target_model(params).to(self.device)
        elif self.mode == 'shs':  # 'smart home speaker
            params = utils_advGAN.params(self.opts['num_class'], self.opts['input_size'])
            target_model = models_gan.target_model_1(params).to(self.device)
        target_model.load_state_dict(torch.load(self.opts['adv_target_model_path'], map_location=self.device))
        target_model.eval()


        "set adversary"
        adversary_name = self.opts['Adversary']
        if adversary_name == 'FGSM':
            print('adv testing with FGSM')
            adversary = FGSM(self.opts['mode'], epsilon=0.1)
        elif adversary_name == 'DeepFool':
            print('adv testing with DeepFool')
            adversary = DeepFool(self.opts['mode'], num_classes=5)
        elif adversary_name == 'PGD':
            print('adv testing with PGD')
            adversary = LinfPGDAttack(self.opts['mode'], k=5, random_start=False)
        elif adversary_name == 'GAN':
            print('adv with GAN')
            pretrained_G = models_gan.Generator(self.gen_input_nc, self.input_nc).to(self.device)
            pretrained_G.load_state_dict(torch.load(self.opts['adv_generator_path'], map_location=self.device))
            pretrained_G.eval()


        "test on adversarial examples"
        num_correct = 0
        total_case = 0
        for i, data in enumerate(test_dataloader, 0):
            test_x, test_y = data
            test_x, test_y = test_x.to(self.device), test_y.to(self.device)

            if adversary_name in ['FGSM', 'DeepFool' , 'PGD']:

                adversary.model = target_model
                adv_y,adv_x = adversary.perturbation(test_x,test_y)
            elif adversary_name == 'GAN':
                pert = pretrained_G(test_x)
                pert = torch.clamp(pert,-self.pert_box,self.pert_box)
                adv_x = pert + test_x

                "clamp adv_x"
                if self.mode == 'wf':
                    pass
                elif self.mode == 'shs':
                    adv_x = torch.clamp(adv_x, self.x_box_min, self.x_box_max)
                adv_y = torch.argmax(target_model(adv_x.to(self.device)),1)

            num_correct += torch.sum(adv_y == test_y, 0)
            total_case += len(test_y)

        print('testing dataset:')
        print('num_correct: ', num_correct.item())
        print('total_num: ', total_case)
        print('accuracy of adv examples in testing set: %f\n' % (num_correct.item() / total_case))



def main(opts):

    against_adv = test_adv(opts)
    against_adv.test_adv_performance()


def get_opts(Adversary):
    return {
        'test_data_path': '../data/NoDef/test_NoDef_burst.csv',
        'adv_target_model_path': '../model/wf/adv_target_model_' + Adversary +  '.pth',
        'adv_generator_path': '../model/wf/adv_generator.pth',
        'mode': 'wf',
        'num_class': 95,
        'input_size': 512,
        'Adversary': Adversary,
        'batch_size': 64,
        'test_batch_size': 16,
    }


if __name__ == '__main__':

    Adversary = ['GAN','FGSM','PGD']
    for adv in Adversary:

        opts = get_opts(adv)

        "set batch_szie, deepfool only work at batch_size=1"
        if adv == 'DeepFool':
            opts['batch_size'] = 1
            print('batch_size {}'.format(opts['batch_size']))
        else:
            opts['batch_size'] = 64
            print('batch_size {}'.format(opts['batch_size']))
            pass

        main(opts)