import os
os.sys.path.append('..')

import torch
from website_fingerprinting import models
from website_fingerprinting import utils_wf
import torch.nn as nn
import torch.optim as optim
from white_box_attacks.adv_box.attacks import FGSM,DeepFool,LinfPGDAttack
from datetime import datetime
import copy,os



class adv_train:

    def __init__(self,target_model,train_data,test_data,opts,learning_rate=1e-3):

        self.Adversary = opts['Adversary']
        self.epoch = opts['epoch']
        self.mode = opts['mode']
        self.model_dir = '../model/' + self.mode
        self.x_box_min = opts['x_box_min']
        self.x_box_max = opts['x_box_max']
        self.opts = opts
        self.target_model = target_model
        self.train_data, self.test_data = train_data, test_data
        self.learning_rate = learning_rate
        self.loss_criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.target_model.parameters(),lr=self.learning_rate)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print("CUDA Available: ", torch.cuda.is_available())


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


    def adv_train_process(self,delay=0.6):
        "Adversarial training, returns pertubed mini batch"
        "delay: parameter to decide how many epochs should be used as adv training"


        # set target model to train mode
        self.target_model.train()
        self.target_model.to(self.device)
        print('trianing mode...')

        # set adversary
        if self.Adversary == 'FGSM':
            print('adv training with FGSM')
            adversary = FGSM(self.mode,self.x_box_min,self.x_box_max,epsilon=0.1)
        elif self.Adversary == 'DeepFool':
            print('adv training with DeepFool')
            adversary = DeepFool(self.mode,self.x_box_min,self.x_box_max,num_classes=5)
        elif self.Adversary == 'PGD':
            print('adv training with PGD')
            adversary = LinfPGDAttack(self.mode,self.x_box_min,self.x_box_max,k=5,random_start=False)


        # start training
        steps = 0
        flag = False
        start_time = datetime.now()

        for epoch in range(self.epoch):
            print('Starting epoch %d / %d' % (epoch + 1, self.epoch))
            for x,y in self.train_data:
                steps += 1
                self.optimizer.zero_grad()

                x,y = x.to(self.device), y.to(self.device)
                outputs = self.target_model(x)
                loss = self.loss_criterion(outputs,y)

                # adversarial training
                if epoch + 1 >= int((1-delay)*self.epoch):
                    flag = True
                    _,y_pred = torch.max(self.target_model(x),1)
                    x_adv = self.x_adv_gen(x,y_pred, self.target_model,adversary)
                    loss_adv = self.loss_criterion(self.target_model(x_adv.to(self.device)),y)
                    loss = (loss + loss_adv) / 2

                # print results every 100 steps
                if steps % 100 == 0:
                    _, predicted = torch.max(self.target_model(x), 1)
                    correct = int(sum(predicted == y))
                    accuracy = correct / len(y)
                    end_time = datetime.now()
                    time_diff = (end_time - start_time).seconds
                    time_usage = '{:3}m{:3}s'.format(int(time_diff / 60), time_diff % 60)
                    msg = "Step {:5}, Loss:{:6.2f}, Accuracy:{:8.2%}, Time usage:{:9}."
                    if flag:
                        print('adversarial training mode...')
                    print(msg.format(steps, loss, accuracy, time_usage))

                loss.backward()
                self.optimizer.step()

        "creat model folder"
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        model_name = self.model_dir + '/adv_target_model_' + self.Adversary + '.pth'
        torch.save(self.target_model.state_dict(), model_name)
        print('trainng process is completed, model has been saved')
        print('*' * 50)


    def testing_process(self):


        print('testing mode...')

        "load adv target model"
        params = utils_wf.params(self.opts['num_class'],self.opts['input_size'])
        target_model = models.target_model(params)
        target_model.load_state_dict(torch.load(self.model_dir + '/adv_target_model_' + self.Adversary + '.pth', map_location=self.device))
        target_model.to(self.device)
        target_model.eval()



        test_loss = 0.
        test_correct = 0
        total_case = 0
        start_time = datetime.now()
        i = 0
        for data, label in self.test_data:
            i += 1
            print('testing {}'.format(i))
            data, label = data.to(self.device),label.to(self.device)
            outputs = target_model(data)
            loss = self.loss_criterion(outputs, label)
            test_loss += loss * len(label)
            _, predicted = torch.max(outputs, 1)
            correct = int(sum(predicted == label))
            test_correct += correct

            total_case += len(label)

            # delete caches
            del data, label, outputs, loss
            torch.cuda.empty_cache()


        accuracy = test_correct / total_case
        loss = test_loss / total_case
        print("Test Loss: {:5.2f}, Accuracy: {:6.2%}".format(loss, accuracy))

        end_time = datetime.now()
        time_diff = (end_time - start_time).seconds
        print("Time Usage: {:5.2f} mins.".format(time_diff / 60.))




def get_opts(Adversary):
    return {
        'train_data_path': '../data/NoDef/train_NoDef_burst.csv',
        'test_data_path': '../data/NoDef/test_NoDef_burst.csv',
        'mode': 'wf',
        'num_class': 95,
        'input_size': 512,
        'Adversary': Adversary,
        'epoch': 50,
        'batch_size': 64,
        'test_batch_size': 16,
        'x_box_min': -1,
        'x_box_max': 0,

    }


def train_main(Adversary):

    opts = get_opts(Adversary)

    "load model"
    params = utils_wf.params(opts['num_class'], opts['input_size'])
    target_model = models.target_model(params)

    "load data"
    train_data = utils_wf.load_data_main(opts['train_data_path'],opts['batch_size'])
    test_data = utils_wf.load_data_main(opts['test_data_path'], opts['test_batch_size'])

    adv_training = adv_train(target_model,train_data,test_data,opts)
    adv_training.adv_train_process()
    # adv_training.testing_process()


def test_main(Adversary):

    opts = get_opts(Adversary)

    "load model"
    params = utils_wf.params(opts['num_class'], opts['input_size'])
    target_model = models.target_model(params)

    "load data"
    train_data = []
    test_data = utils_wf.load_data_main(opts['test_data_path'], opts['test_batch_size'])

    adv_training = adv_train(target_model, train_data, test_data, opts)
    adv_training.testing_process()


if __name__ == '__main__':

  "adv_trian"
  Adversary = ['FGSM','PGD']
  for adv in Adversary:
      train_main(adv)

  "test performance"
  # Adversary = ['FGSM']
  # for adv in Adversary:
  #   test_main(adv)


