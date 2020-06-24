import os
os.sys.path.append('..')

import torch
from train import models
from train import utils_shs,utils_wf,utils_gan
import torch.nn as nn
import torch.optim as optim
from attacks.white_box.adv_box.attacks import FGSM,DeepFool,LinfPGDAttack
from datetime import datetime
import copy,sys



class adv_train:

    def __init__(self,opts):

        self.opts = opts
        self.mode = opts['mode']
        self.model_path = '../model/' + self.mode
        self.gen_input_nc, self.input_nc = 1,1
        self.loss_criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.target_model.parameters(),lr=1e-3)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('CUDA AVAILABLE:', torch.cuda.is_available())
        print('Mode: ', self.mode)

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)


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
        _,x_adv = adversary.perturbation(x, y)

        return x_adv


    def adv_train_process(self,delay=0.5):
        "Adversarial training, returns pertubed mini batch"
        "delay: parameter to decide how many epochs should be used as adv training"

        if self.mode == 'wf':
            "load data"
            train_data = utils_wf.load_data_main(self.opts['train_data_path'],self.opts['batch_size'])
            test_data = utils_wf.load_data_main(self.opts['test_data_path'],self.opts['batch_size'])

            "load target model structure"
            params = utils_wf.params(self.opts['num_class'],self.opts['input_size'])
            target_model = models.target_model_wf(params).to(self.device)
            target_model.train()

        elif self.mode == 'shs':
            "load data"
            train_data = utils_shs.load_data_main(self.opts['train_data_path'],self.opts['batch_size'])
            test_data = utils_shs.load_data_main(self.opts['test_data_path'],self.opts['batch_size'])

            "load target model structure"
            params = utils_shs.params(self.opts['num_class'],self.opts['input_size'])
            target_model = models.target_model_shs(params).to(self.device)
            target_model.train()

        else:
            print('mode not in ["wf","shs"], system will exit.')
            sys.exit()

        "set adversary"
        Adversary = self.opts['Adversary']
        if Adversary == 'FGSM':
            print('adv training with FGSM')
            adversary = FGSM(self.mode,x_box_min=self.opts['x_box_min'],x_box_max=self.opts['x_box_max'],pert_box=self.opts['pert_x'], epsilon=0.1)
        elif Adversary == 'DeepFool':
            print('adv training with DeepFool')
            adversary = DeepFool(self.mode,x_box_min=self.opts['x_box_min'],x_box_max=self.opts['x_box_max'],pert_box=self.opts['pert_x'],num_classes=5)
        elif Adversary == 'PGD':
            print('adv training with PGD')
            adversary = LinfPGDAttack(self.mode,x_box_min=self.opts['x_box_min'],x_box_max=self.opts['x_box_max'],pert_box=self.opts['pert_x'],k=5,random_start=False)
        elif Adversary == 'GAN':
            generator = models.Generator(self.gen_input_nc,self.input_nc).to(self.device)
            generator.load_state_dict(torch.load(self.opts['adv_generator_path'],map_location=self.device))
            generator.eval()


        "start training process"
        steps = 0
        flag = False
        start_time = datetime.now()

        for epoch in range(self.opts['epoch']):
            print('Starting epoch %d / %d' % (epoch + 1, self.opts['epoch']))

            if flag:
                print('adversarial training mode...')

            for x,y in train_data:
                steps += 1
                self.optimizer.zero_grad()
                x,y = x.to(self.device), y.to(self.device)
                outputs = target_model(x)
                loss = self.loss_criterion(outputs,y)

                "adversarial training"
                if epoch + 1 >= int((1-delay)*self.opts['epoch']):
                    flag = True
                    if self.mode == 'shs':
                        _,y_pred = torch.max(self.target_model(x),1)
                        x_adv = self.x_adv_gen(x,y_pred, self.target_model,adversary)

                    elif self.mode == 'wf':
                        "produce adversarial examples"
                        pert = generator(X)
                        x_adv = utils_gan.getget_advX_gan(x,pert,mode,pert_box=self.opts['pert_box'],x_box_min=self.opts['x_box_min'],x_box_max=self.opts['x_box_max'])

                    loss_adv = self.loss_criterion(self.target_model(x_adv.to(self.device)),y)
                    loss = (loss + loss_adv) / 2
                "print results every 100 steps"
                if steps % 100 == 0:
                    end_time = datetime.now()
                    time_diff = (end_time - start_time).seconds
                    time_usage = '{:3}m{:3}s'.format(int(time_diff / 60), time_diff % 60)
                    msg = "Step {:5}, Loss:{:6.2f}, Time usage:{:9}."
                    print(msg.format(steps, loss, time_usage))

                loss.backward()
                self.optimizer.step()

            if epoch % 10 == 0:
                torch.save(self.target_model.state_dict(), self.model_path + '/adv_target_model_' + Adversary + '.pth')


        #****************************
        "test trained model"
        # ****************************
        target_model.eval()
        test_loss = 0.
        test_correct = 0
        total_case = 0
        start_time = datetime.now()
        i = 0
        for data, label in test_data:
            data, label = data.to(self.device), label.to(self.device)
            outputs = target_model(data)
            loss = self.loss_criterion(outputs, label)
            test_loss += loss * len(label)
            _, predicted = torch.max(outputs, 1)
            correct = int(sum(predicted == label))
            test_correct += correct
            total_case += len(label)

            "delete caches"
            del data, label, outputs, loss
            torch.cuda.empty_cache()

        accuracy = test_correct / total_case
        loss = test_loss / total_case
        print("Test Loss: {:5.2f}, Accuracy: {:6.2%}".format(loss, accuracy))

        end_time = datetime.now()
        time_diff = (end_time - start_time).seconds
        print("Time Usage: {:5.2f} mins.".format(time_diff / 60.))



def main(opts):
    adv_training = adv_train(opts)
    adv_training.adv_train_process()


def get_opts_shs(Adversary):
    return {
        'mode': 'wf',
        'Adversary': Adversary,
        'x_box_min': -1,
        'x_box_max': 0,
        'num_class': 101,
        'input_size': 256,
        'train_data_path': '../data/traffic_train.csv',
        'test_data_path': '../data/traffic_test.csv',
        'epochs': 50,
        'batch_size': 64,
        'test_batch_size': 64,
        'pert_box': 0.3,
        'delay': 0.5,
    }


def get_opts_wf(Adversary):
    return {
        'mode': 'wf',
        'Adversary': Adversary,
        'x_box_min': -1,
        'x_box_max': 0,
        'pert_box': 0.3,
        'num_class': 95,
        'input_size': 512,
        'train_data_path': '../data/NoDef/train_NoDef_burst.csv',
        'test_data_path': '../data/NoDef/test_NoDef_burst.csv',
        'epochs': 50,
        'batch_size': 64,
        'test_batch_size': 64,
        'delay': 0.5,
    }


if __name__ == '__main__':
    Adversary = ['FGSM','PGD','GAN']
    mode = 'wf'

    for adv in Adversary:
        if mode == 'wf':
            opts = get_opts_wf(adv)
        elif mode == 'shs':
            opts = get_opts_shs(adv)
            
        main(opts)
