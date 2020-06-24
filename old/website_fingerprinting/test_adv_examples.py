import os
os.sys.path.append('..')

import torch
from white_box_attacks.adv_box.attacks import FGSM,DeepFool,LinfPGDAttack
from website_fingerprinting import utils_wf
from website_fingerprinting import models




class test_adv:
    def __init__(self,opts):
        self.opts = opts
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print("CUDA Available: ", torch.cuda.is_available())


    def test_adv_performance(self):

        "load data"
        test_dataloader = utils_wf.load_data_main(self.opts['test_data_path'],self.opts['batch_size'])

        "load target model"
        params = utils_wf.params(self.opts['num_class'],self.opts['input_size'])
        target_model = models.target_model(params).to(self.device)
        target_model.load_state_dict(torch.load(self.opts['adv_target_model_path'],map_location=self.device))
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


        "test on adversarial examples"
        num_correct = 0
        total_case = 0
        for i, data in enumerate(test_dataloader, 0):
            # print(i, 'test adv examples ...')
            test_x, test_y = data
            test_x, test_y = test_x.to(self.device), test_y.to(self.device)

            adversary.model = target_model
            adv_y, adv_x = adversary.perturbation(test_x, test_y)
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
        'mode': 'wf',
        'num_class': 95,
        'input_size': 512,
        'Adversary': Adversary,
        'batch_size': 64,
        'test_batch_size': 16,
    }


if __name__ == '__main__':

    Adversary = ['FGSM','PGD']
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