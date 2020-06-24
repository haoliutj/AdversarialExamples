import os
os.sys.path.append('..')

import torch
import time
from adv_box.attacks import FGSM, DeepFool, LinfPGDAttack
from advGAN import models
import utils


adversary = 'PGD'
input_size = 256
num_class = 101
batch_size = 64
mode = 'shs'
x_box_min = -1
x_box_max = 0
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# set adversary
if adversary == 'FGSM':
    print('adv testing with FGSM')
    adversary = FGSM(mode,x_box_min,x_box_max,epsilon=0.1)
elif adversary == 'DeepFool':
    print('adv testing with DeepFool')
    adversary = DeepFool(mode,x_box_min,x_box_max,num_classes=5)
elif adversary == 'PGD':
    print('adv testing with PGD')
    adversary = LinfPGDAttack(mode,x_box_min,x_box_max,k=5, random_start=False)



"load target model"
params = utils.params(num_class,input_size)
model = models.target_model_1(params)
model.load_state_dict(torch.load('../model/target_model.pth', map_location = device))
model.to(device)
model.eval()


"load data"
# train = utils.load_data_main('../data/traffic_train.csv', 64) # input_shape (batch_size,1,wide 256)
test = utils.load_data_main('../data/traffic_test.csv', batch_size)
# print('train_loader size: {}'.format(len(train)))
print('test_loader size: {}'.format(len(test)))





"test performance"
correct = 0
correct_x = 0
i = 0
start_time = time.time()
for data, label in test:
    data, label = data.to(device),label.to(device)

    pred_x = model(data)
    pred_x = torch.max(pred_x, 1)    #return (max,index)
    pred_x = pred_x[1]

    adversary.model = model
    y_adv,x_adv = adversary.perturbation(data,label)

    # label = label.to(device)
    # outputs = model.forward(x_adv.to(device))
    # _,y_adv = torch.max(outputs.data,1)   #retrun (max,index)

    correct += (y_adv == label).sum()
    correct_x += (pred_x == label).sum()

    print('origial groudtruth label is: {}'.format(label))
    print('original predict laebl is: {}'.format(pred_x))
    print('adv label is: {}'.format(y_adv))

    i += 1

end_time = time.time()
time_diff = end_time - start_time
total_case = len(test) * batch_size
print('#'*20)
print('average time is {} seconds'.format(time_diff / float(total_case)))
print('total test is {}'.format(total_case))
print('correct test after attack is {}'.format(correct))
print('Accuracy of test after attack: correct/total= {:.5f}'.format(float(correct) / total_case))
print('success rate of attack is: 1 - correct/total = {:.5f}'.format(1-(float(correct) / total_case)))
print('model classfication accucary without being attacked is {:.5f}'.format(float(correct_x) / float(total_case)))


