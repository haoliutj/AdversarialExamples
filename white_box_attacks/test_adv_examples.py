import os
os.sys.path.append('..')

import torch
import torch.nn as nn
from advGAN import models
from adv_box.attacks import FGSM,DeepFool,DeepFool_batch_train,LinfPGDAttack
import utils
import time



class config:
    batch_size = 64
    test_batch_size = 1
    use_cuda = True
    adversary = 'DeepFool'


# Define what device we are using
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if (config.use_cuda and torch.cuda.is_available()) else "cpu")


# load the pretrained model
pretrained_model = '../model/adv_target_model_' + config.adversary + '.pth'
params = utils.params()
target_model = models.target_model_1(params).to(device)
target_model.load_state_dict(torch.load(pretrained_model,map_location=device))
target_model.eval()


"load test data set "
test_dataloader = utils.load_data_main('../data/traffic_test.csv', config.test_batch_size)


# set adversary
if config.adversary == 'FGSM':
    print('adv training with FGSM')
    adversary = FGSM(epsilon=0.1)
elif config.adversary == 'DeepFool':
    print('adv training with DeepFool')
    adversary = DeepFool(num_classes=5)
    # adversary = DeepFool_batch_train(num_classes=5)
elif config.adversary == 'PGD':
    print('adv training with PGD')
    adversary = LinfPGDAttack(k=5, random_start=False)


"test on adversarial examples"
num_correct = 0
for i, data in enumerate(test_dataloader, 0):
    print(i,'test adv examples ...')
    test_x, test_y = data
    test_x, test_y = test_x.to(device), test_y.to(device)

    adversary.model = target_model
    adv_y,adv_x = adversary.perturbation(test_x, test_y)
    num_correct += torch.sum(adv_y == test_y, 0)

print('testing dataset:')
print('num_correct: ', num_correct.item())
print('total_num: ', (len(test_dataloader) * config.test_batch_size))
print('accuracy of adv examples in testing set: %f\n' % (num_correct.item() / (len(test_dataloader) * config.test_batch_size)))



