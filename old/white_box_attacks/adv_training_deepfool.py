import os
os.sys.path.append('..')

import torch
from advGAN import models as models_gan
import utils as utils_gb
import torch.nn as nn
import torch.optim as optim
from white_box_attacks.dv_box.attacks import FGSM,DeepFool,DeepFool_batch_train,LinfPGDAttack
from datetime import datetime
import copy


class config:
    batch_size = 64
    test_batch_size = 16
    epoch = 60
    print_per_step = 100
    learning_rate = 1e-3
    adversary = 'DeepFool'


"load target model"
params = utils.params()
target_model = models.target_model_1(params)

"load train and test data"
train_data = utils.load_data_main('../data/traffic_train.csv',config.batch_size) # input_shape (batch_size,1,wide 256)
test_data = utils.load_data_main('../data/traffic_test.csv',config.test_batch_size)
adv_trian_data = utils.load_data_main('../data/traffic_train_advDeepFool.csv',config.batch_size) # adv train with deepfool batch_size =1



class adv_train:

    def __init__(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.target_model = target_model
        self.train_data, self.test_data, self.adv_train_data = train_data, test_data,adv_trian_data
        self.loss_criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.target_model.parameters(),lr=config.learning_rate)


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

        print('use cuda: ', ('True' if torch.cuda.is_available() else 'False'))

        # set target model to train mode
        self.target_model.train()
        self.target_model.to(self.device)
        print('trianing mode...')

        # set adversary
        if config.adversary == 'FGSM':
            print('adv training with FGSM')
            adversary = FGSM(epsilon=0.1)
        elif config.adversary == 'DeepFool':
            print('adv training with DeepFool')
            adversary = DeepFool(num_classes=5)
            # adversary = DeepFool_batch_train(num_classes=10)
        elif config.adversary == 'PGD':
            print('adv training with PGD')
            adversary = LinfPGDAttack(k=10,random_start=False)


        # start training
        steps = 0
        flag = False
        start_time = datetime.now()

        for epoch in range(config.epoch):
            print('Starting epoch %d / %d' % (epoch + 1, config.epoch))

            # for x,y in self.train_data:
            for (x,y),(x_adv,y_adv) in zip(self.train_data,self.adv_train_data):
                steps += 1
                self.optimizer.zero_grad()

                x,y = x.to(self.device), y.to(self.device)
                outputs = self.target_model(x)
                loss = self.loss_criterion(outputs,y)

                # adversarial training
                if epoch + 1 >= int((1-delay)*config.epoch):
                    x_adv, y_adv = x_adv.to(self.device), y_adv.to(self.device)
                    flag = True
                    # _,y_pred = torch.max(self.target_model(x),1)
                    # x_adv = self.x_adv_gen(x,y_pred, self.target_model,adversary)
                    loss_adv = self.loss_criterion(self.target_model(x_adv),y_adv)
                    loss = (loss + loss_adv) / 2

                # print results every 100 steps
                if steps % config.print_per_step == 0:
                    _, predicted = torch.max(self.target_model(x), 1)
                    correct = int(sum(predicted == y))
                    accuracy = correct / config.batch_size
                    end_time = datetime.now()
                    time_diff = (end_time - start_time).seconds
                    time_usage = '{:3}m{:3}s'.format(int(time_diff / 60), time_diff % 60)
                    msg = "Step {:5}, Loss:{:6.2f}, Accuracy:{:8.2%}, Time usage:{:9}."
                    if flag:
                        print('adversarial training mode...')
                    print(msg.format(steps, loss, accuracy, time_usage))

                loss.backward()
                self.optimizer.step()


        torch.save(self.target_model.state_dict(), '../model/adv_target_model_' + config.adversary + '.pth')
        print('trainng process is completed, model has been saved')
        print('*' * 50)


    def testing_process(self):


        print('testing mode...')

        "load adv target model"
        params = utils.params()
        target_model = models.target_model_1(params)
        target_model.load_state_dict(torch.load('../model/adv_target_model_' + config.adversary + '.pth', map_location=self.device))
        target_model.to(self.device)
        target_model.eval()

        # self.target_model.eval()

        test_loss = 0.
        test_correct = 0
        start_time = datetime.now()
        i = 0
        for data, label in self.test_data:
            i += 1
            print('testing {}'.format(i))
            data, label = data.to(self.device),label.to(self.device)
            outputs = target_model(data)
            loss = self.loss_criterion(outputs, label)
            test_loss += loss * config.test_batch_size
            _, predicted = torch.max(outputs, 1)
            correct = int(sum(predicted == label))
            test_correct += correct

            # delete caches
            del data, label, outputs, loss
            torch.cuda.empty_cache()

        accuracy = test_correct / (len(self.test_data.dataset)*config.test_batch_size)
        loss = test_loss / (len(self.test_data.dataset)*config.test_batch_size)
        print("Test Loss: {:5.2f}, Accuracy: {:6.2%}".format(loss, accuracy))

        end_time = datetime.now()
        time_diff = (end_time - start_time).seconds
        print("Time Usage: {:5.2f} mins.".format(time_diff / 60.))


def train_main():
    adv_training = adv_train()
    adv_training.adv_train_process()
    # adv_training.testing_process()


def test_main():
    adv_training = adv_train()
    adv_training.testing_process()


if __name__ == '__main__':

  # adv_trian
    train_main()

  # test performance
  #   test_main()
