import os
os.sys.path.append('..')

import torch
import pandas as pd
from white_box_attacks.adv_box.attacks import FGSM, DeepFool, LinfPGDAttack
from website_fingerprinting import models,utils_wf
import sys



class gen_adv_x:
    def __init__(self,mode,adversary_name,adversary,target_model,train_data,test_data):
        self.mode = mode
        self.data_dir = '../data/' + self.mode
        self.adversary_name = adversary_name
        self.adversary = adversary
        self.target_model = target_model
        self.train_data, self.test_data = train_data, test_data
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(self.device)


    def get_adv_x(self):
        "generate adv_x given x, append with its original label y instead with y_pert "

        "creat data folder"
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)


        "gen adv_x for test data"
        adv_xs = []
        labels = []
        for i,(x,y) in enumerate(self.test_data):
            print('generate adversary example of test data {} ...'.format(i))

            x,y = x.to(self.device),y.to(self.device)
            self.adversary.model = self.target_model
            _,adv_x = self.adversary.perturbation(x,y)

            if self.mode == 'shs':
                adv_x = (adv_x.data.cpu().numpy().squeeze() * 1500).round()
            elif self.mode == 'wf':
                adv_x = (adv_x.data.cpu().numpy().squeeze()).round()
            else:
                print('mode should in ["wf","shs"], system will exit.')
                sys.exit()

            adv_xs.append(adv_x)
            labels.append(y.data.cpu().numpy().squeeze())

        adv_xs = pd.DataFrame(adv_xs)
        adv_xs['label'] = labels
        output_path = self.data_dir + '/adv_test_' + self.adversary_name + '.csv'
        adv_xs.to_csv(output_path,index=0)
        print('adversary examples of test data is generated')



        "gen adv_x for train data"
        # adv_xs = []
        # labels = []
        # for i,(x,y) in enumerate(self.train_data):
        #     print('generate adversary example of train data {} ...'.format(i))
        #
        #     x,y = x.to(self.device),y.to(self.device)
        #     self.adversary.model = self.target_model
        #     _,adv_x = self.adversary.perturbation(x,y)
        #
        #     if self.mode == 'shs':
        #         adv_x = (adv_x.data.cpu().numpy().squeeze()*1500).round()
        #     elif self.mode == 'wf':
        #         adv_x = (adv_x.data.cpu().numpy().squeeze()).round()
        #     else:
        #         print('mode should in ["wf","shs"], system will exit.')
        #         sys.exit()
        #
        #     adv_xs.append(adv_x)
        #     labels.append(y.data.cpu().numpy().squeeze())
        #
        # adv_xs = pd.DataFrame(adv_xs)
        # adv_xs['label'] = labels
        # output_path = self.data_dir + '/adv_train_' + self.adversary_name + '.csv'
        # adv_xs.to_csv(output_path, index=0)
        # print('adversary examples of train data is generated')


def main(opts):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    adversary_name = opts['Adversary']

    # set adversary
    if adversary_name == 'FGSM':
        print('adv testing with FGSM')
        adversary = FGSM(opts['mode'],epsilon=0.1)
    elif adversary_name == 'DeepFool':
        print('adv testing with DeepFool')
        adversary = DeepFool(opts['mode'],num_classes=5)
    elif adversary_name == 'PGD':
        print('adv testing with PGD')
        adversary = LinfPGDAttack(opts['mode'],k=5, random_start=False)


    "load target model"
    "loading target model..."
    params = utils_wf.params(opts['num_class'],opts['input_size'])
    model_path = '../model/' + opts['mode'] + '/target_model.pth'
    target_model = models.target_model(params)
    target_model.load_state_dict(torch.load(model_path, map_location = device))
    target_model.to(device)
    target_model.eval()


    "load data"
    print('loading data...')
    train_data = utils_wf.load_data_main(opts['train_data_path'], opts['batch_size'])
    test_data = utils_wf.load_data_main(opts['test_data_path'], opts['batch_size'])
    print('train_loader batch size: {}'.format(len(train_data)))
    print('test_loader batch size: {}'.format(len(test_data)))


    "generate adversary examples"
    gen_adv = gen_adv_x(opts['mode'],adversary_name,adversary,target_model,train_data,test_data)
    gen_adv.get_adv_x()



def get_opts(Adversary):
    return {
        'train_data_path': '../data/NoDef/train_NoDef_burst.csv',
        'test_data_path': '../data/NoDef/test_NoDef_burst.csv',
        'mode': 'wf',
        'num_class': 95,
        'input_size': 512,
        'Adversary': Adversary,
        'batch_size': 64,
        'test_batch_size': 16,
    }



if __name__ == '__main__':

    adveraries = ['FGSM','PGD']

    for adversary in adveraries:

        opts = get_opts(adversary)

        "set batch_szie, deepfool only work at batch_size=1"
        if adversary == 'DeepFool':
            opts['batch_size'] = 1
            print('batch_size {}'.format(opts['batch_size']))
        else:
            opts['batch_size'] = 64
            print('batch_size {}'.format(opts['batch_size']))
            pass

        main(opts)




