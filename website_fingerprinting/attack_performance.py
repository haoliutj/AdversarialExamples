import os
os.sys.path.append('..')

import torch
import time
from white_box_attacks.adv_box.attacks import FGSM, DeepFool, LinfPGDAttack
from website_fingerprinting import models,utils_wf
from advGAN import models as model_gan


Adversary = 'GAN'
batch_size = 64
gen_input_nc = image_nc = 1
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class config:
    pert_clamp = 0.2
    box_min,box_max = -1,0


# set adversary
"GAN adversary"
if Adversary == 'GAN':
    print('adv with GAN')
    pretrained_generator_path = '../model/adv_generator.pth'
    pretrained_G = model_gan.Generator(gen_input_nc, image_nc).to(device)
    pretrained_G.load_state_dict(torch.load(pretrained_generator_path, map_location=device))
    pretrained_G.eval()
elif Adversary == 'FGSM':
    print('adv testing with FGSM')
    adversary = FGSM(epsilon=0.1)
elif Adversary == 'DeepFool':
    print('adv testing with DeepFool')
    adversary = DeepFool(num_classes=5)
elif Adversary == 'PGD':
    print('adv testing with PGD')
    adversary = LinfPGDAttack(k=5, random_start=False)



"load target model"
params = utils_wf.params()
model = models.target_model(params)
model.load_state_dict(torch.load('../model/wf_model/target_model.pth', map_location = device))
model.to(device)
model.eval()


"load data"
test = utils_wf.load_data_main('../data/NoDef/test_NoDef.csv', batch_size)
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

    if Adversary in ['FGSM','DeepFool','PGD']:
        adversary.model = model
        y_adv,x_adv = adversary.perturbation(data,label)

    elif Adversary == 'GAN':
        perturbation = pretrained_G(data)
        perturbation = torch.clamp(perturbation, -config.pert_clamp, config.pert_clamp)
        adv_x = perturbation + data
        adv_x = torch.clamp(adv_x, config.box_min, config.box_max)
        outputs = model(adv_x.to(device))
        _, y_adv = torch.max(outputs.data, 1)  # retrun (max,index)

    # label = label.to(device)
    # outputs = model.forward(x_adv.to(device))
    # _,y_adv = torch.max(outputs.data,1)   #retrun (max,index)

    correct += (y_adv == label).sum()
    correct_x += (pred_x == label).sum()

    print('origial groudtruth label is: {}'.format(label.data.cpu().numpy()))
    print('original predict laebl is: {}'.format(pred_x.data.cpu().numpy()))
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


