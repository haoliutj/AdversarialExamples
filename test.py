import os
os.sys.path.append('..')

import torch
from advGAN import models
import utils
import torch.nn as nn
import torch.optim as optim

from datetime import datetime
import copy

class config:
    batch_size = 64
    test_batch_size = 16
    epoch = 50
    print_per_step = 100
    learning_rate = 1e-3
    adversary = 'DeepFool'


"load train and test data"
train_data = utils.load_data_main('./data/traffic_train.csv',config.batch_size) # input_shape (batch_size,1,wide 256)
test_data = utils.load_data_main('./data/traffic_test.csv',config.test_batch_size)
adv_trian_data = utils.load_data_main('./data/traffic_train_advDeepFool.csv',config.batch_size) # adv train with deepfool batch_size =1



for (x,y),(z,u) in zip(train_data,adv_trian_data):
    print(x,y)
    print(z,u)