import os
os.sys.path.append('..')

import torch
from advGAN import models as models_gan
import utils as utils_gb
import torch.nn as nn
import torch.optim as optim
from white_box_attacks.adv_box.attacks import FGSM,DeepFool,LinfPGDAttack
from datetime import datetime
import copy


# class config:
#     batch_size = 64
#     test_batch_size = 16
#     epoch = 50
#     print_per_step = 100
#     learning_rate = 1e-3
#     adversary = 'PGD'
#
#
# "load target model"
# params = utils.params()
# target_model = models.target_model_1(params)
#
# "load train and test data"
# train_data = utils.load_data_main('../data/traffic_train.csv',config.batch_size) # input_shape (batch_size,1,wide 256)
# test_data = utils.load_data_main('../data/traffic_test.csv',config.test_batch_size)



class adv_train:

    def __init__(self,opts):

        self.opts = opts
        self.mode = opts['mode']
        self.model_path = '../model/' + self.mode
        # self.target_model = target_model
        # self.train_data, self.test_data = train_data, test_data
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

        "load data"
        train_data = utils_gb.load_data_main(self.opts['train_data_path'], self.opts['batch_size'])
        test_data = utils_gb.load_data_main(self.opts['test_data_path'], self.opts['test_batch_size'])

        "load target model structure"
        params = utils_gb.params(self.opts['num_class'], self.opts['input_size'])
        target_model = models_gan.target_model_1(params).to(self.device)
        target_model.train()

        "set adversary"
        Adversary = self.opts['Adversary']
        if Adversary == 'FGSM':
            print('adv training with FGSM')
            adversary = FGSM(epsilon=0.1)
        elif Adversary == 'DeepFool':
            print('adv training with DeepFool')
            adversary = DeepFool(num_classes=5)
        elif Adversary == 'PGD':
            print('adv training with PGD')
            adversary = LinfPGDAttack(k=5,random_start=False)


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
                    _,y_pred = torch.max(self.target_model(x),1)
                    x_adv = self.x_adv_gen(x,y_pred, self.target_model,adversary)
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




    # def testing_process(self):
    #
    #     print('testing mode...')
    #
    #     "load adv target model"
    #     params = utils.params()
    #     target_model = models.target_model_1(params)
    #     target_model.load_state_dict(torch.load('../model/adv_target_model_' + config.adversary + '.pth', map_location=self.device))
    #     target_model.to(self.device)
    #     target_model.eval()
    #
    #     # self.target_model.eval()
    #
    #     test_loss = 0.
    #     test_correct = 0
    #     start_time = datetime.now()
    #     i = 0
    #     for data, label in self.test_data:
    #         i += 1
    #         print('testing {}'.format(i))
    #         data, label = data.to(self.device),label.to(self.device)
    #         outputs = target_model(data)
    #         loss = self.loss_criterion(outputs, label)
    #         test_loss += loss * config.test_batch_size
    #         _, predicted = torch.max(outputs, 1)
    #         correct = int(sum(predicted == label))
    #         test_correct += correct
    #
    #         # delete caches
    #         del data, label, outputs, loss
    #         torch.cuda.empty_cache()
    #
    #     accuracy = test_correct / (len(self.test_data.dataset)*config.test_batch_size)
    #     loss = test_loss / (len(self.test_data.dataset)*config.test_batch_size)
    #     print("Test Loss: {:5.2f}, Accuracy: {:6.2%}".format(loss, accuracy))
    #
    #     end_time = datetime.now()
    #     time_diff = (end_time - start_time).seconds
    #     print("Time Usage: {:5.2f} mins.".format(time_diff / 60.))


def main(opts):
    adv_training = adv_train(opts)
    adv_training.adv_train_process()


def get_opts(Adversary):
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


if __name__ == '__main__':
    Adversary = ['FGSM','PGD']
    
    for adv in Adversary:
        opts = get_opts(adv)
        main(opts)

