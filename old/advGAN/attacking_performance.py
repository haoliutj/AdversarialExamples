"adv_examples of GAN against target model"


import os
os.sys.path.append('..')

import torch
import time
import utils as utils_gb
from website_fingerprinting import utils_wf
from website_fingerprinting import models as models_wf
from advGAN import models as models_gan
from white_box_attacks.adv_box.attacks import FGSM, DeepFool, LinfPGDAttack
from sklearn.preprocessing import MinMaxScaler





device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class config:
    image_nc = 1
    gen_input_nc = image_nc
    batch_size = 64
    test_batch_size = 64
    pert_clamp = 0.3
    box_min,box_max = -1,0
    num_class = 95
    input_size =512
    adversary = 'GAN'

# set adversary
"GAN adversary"
if config.adversary == 'GAN':
    print('adv with GAN')
    pretrained_generator_path = '../model/wf/adv_generator.pth'
    pretrained_G = models_gan.Generator(config.gen_input_nc, config.image_nc).to(device)
    pretrained_G.load_state_dict(torch.load(pretrained_generator_path, map_location=device))
    pretrained_G.eval()
elif config.adversary == 'FGSM':
    print('adv with FGSM')
    adversary = FGSM(epsilon=0.1)
elif config.adversary == 'DeepFool':
    print('adv with DeepFool')
    adversary = DeepFool(num_classes=5)
elif config.adversary == 'PGD':
    print('adv with PGD')
    adversary = LinfPGDAttack(k=5, random_start=False)



"load target model"
params = utils_wf.params(config.num_class,config.input_size)
model = models_wf.target_model(params)
model.load_state_dict(torch.load('../model/wf/target_model.pth', map_location = device))
model.to(device)
model.eval()


"load data"
test = utils_wf.load_data_main('../data/NoDef/test_NoDef_burst.csv', config.test_batch_size)
print('test_loader size: {}'.format(len(test)))


"test performance"
correct = 0
correct_x = 0
total_case = 0
i = 0
start_time = time.time()
for data, label in test:
    data, label = data.to(device),label.to(device)
    "prediction on x"
    pred_x = model(data)
    pred_x = torch.max(pred_x, 1)    #return (max,index)
    pred_x = pred_x[1]

    "prediction on x_adv"
    perturbation = pretrained_G(data)
    perturbation = torch.clamp(perturbation, -config.pert_clamp, config.pert_clamp)
    # adv_x = perturbation + data

    "normalization"
    input_shape = data.shape
    data_temp = data.view(-1,1)
    data_temp = data_temp.data.cpu().numpy()
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(data_temp)
    data_norm = scaler.transform(data_temp)
    data_norm = torch.Tensor(data_norm)
    data_norm = data_norm.view(input_shape)

    adv_x_norm = utils_gb.add_perturbation(data_norm,perturbation)
    adv_x_norm = adv_x_norm.view(-1,1)
    adv_x = scaler.inverse_transform(adv_x_norm.data.cpu().numpy())
    adv_x = torch.Tensor(adv_x)
    adv_x = adv_x.view(input_shape)

    "prediction"
    outputs = model(adv_x.to(device))
    _,y_adv = torch.max(outputs.data,1)   #retrun (max,index)

    correct += (y_adv == label).sum()
    correct_x += (pred_x == label).sum()

    # print('origial groudtruth label is: {}'.format(label.item()))
    # print('original predict laebl is: {}'.format(pred_x.item()))
    # print('adv label is: {}'.format(y_adv))

    i += 1
    total_case += len(label)

end_time = time.time()
time_diff = end_time - start_time
print('#'*20)
print('average time is {} seconds'.format(time_diff / float(total_case)))
print('total test is {}'.format(total_case))
print('correct test after attack is {}'.format(correct))
print('Accuracy of test after attack: correct/total= {:.5f}'.format(float(correct) / total_case))
print('success rate of attack is: 1 - correct/total = {:.5f}'.format(1-(float(correct) / total_case)))
print('model classfication accucary without being attacked is {:.5f}'.format(float(correct_x) / float(total_case)))


