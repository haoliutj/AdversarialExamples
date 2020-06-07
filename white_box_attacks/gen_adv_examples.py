import os
os.sys.path.append('..')

import torch
import time
import pandas as pd
from adv_box.attacks import FGSM, DeepFool, LinfPGDAttack
from advGAN import models
import utils


class gen_adv_x:
    def __init__(self,adversary_name,adversary,target_model,train_data,test_data):
        self.adversary_name = adversary_name
        self.adversary = adversary
        self.target_model = target_model
        self.train_data, self.test_data = train_data, test_data
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(self.device)


    def get_adv_x(self):
        "generate adv_x given x, append with its original label y instead with y_pert "


        "gen adv_x for test data"
        # adv_xs = []
        # labels = []
        # for i,(x,y) in enumerate(self.test_data):
        #     print('generate adversary example of test data {} ...'.format(i))
        #
        #     x,y = x.to(self.device),y.to(self.device)
        #     self.adversary.model = self.target_model
        #     _,adv_x = self.adversary.perturbation(x,y)
        #
        #     adv_x = (adv_x.data.cpu().numpy().squeeze() * 1500).round()
        #     adv_xs.append(adv_x)
        #     labels.append(y.data.cpu().numpy().squeeze())
        #
        # adv_xs = pd.DataFrame(adv_xs)
        # adv_xs['label'] = labels
        # adv_xs.to_csv('../data/traffic_test_adv_' + self.adversary_name +'.csv',index=0)
        # print('adversary examples of test data is generated')



        "gen adv_x for train data"
        adv_xs = []
        labels = []
        for i,(x,y) in enumerate(self.train_data):
            print('generate adversary example of train data {} ...'.format(i))

            x,y = x.to(self.device),y.to(self.device)
            self.adversary.model = self.target_model
            _,adv_x = self.adversary.perturbation(x,y)

            adv_x = (adv_x.data.cpu().numpy().squeeze()*1500).round()
            adv_xs.append(adv_x)
            labels.append(y.data.cpu().numpy().squeeze())

        adv_xs = pd.DataFrame(adv_xs)
        adv_xs['label'] = labels
        adv_xs.to_csv('../data/traffic_train_adv_' + self.adversary_name + '.csv', index=0)
        print('adversary examples of train data is generated')


def main(adversary_name,batch_size):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # set adversary
    if adversary_name == 'FGSM':
        print('adv testing with FGSM')
        adversary = FGSM(epsilon=0.1)
    elif adversary_name == 'DeepFool':
        print('adv testing with DeepFool')
        adversary = DeepFool(num_classes=5)
        # adversary = DeepFool_batch_train(num_classes=10)
    elif adversary_name == 'PGD':
        print('adv testing with PGD')
        adversary = LinfPGDAttack(k=5, random_start=False)


    "load target model"
    "loading target model..."
    params = utils.params()
    target_model = models.target_model_1(params)
    target_model.load_state_dict(torch.load('../model/target_model.pth', map_location = device))
    target_model.to(device)
    target_model.eval()


    "load data"
    print('loading data...')
    train_data = utils.load_data_main('../data/traffic_train.csv', batch_size)
    test_data = utils.load_data_main('../data/traffic_test.csv', batch_size)
    print('train_loader size: {}'.format(len(train_data)))
    print('test_loader size: {}'.format(len(test_data)))


    "generate adversary examples"
    gen_adv = gen_adv_x(adversary_name,adversary,target_model,train_data,test_data)
    gen_adv.get_adv_x()


if __name__ == '__main__':

    adveraries = ['DeepFool']
    batch_size = 1

    for adverary in adveraries:
        main(adverary,batch_size)




