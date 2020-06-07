import os
os.sys.path.append('..')

import torch
import models
from models import target_model_1
import utils
from white_box_attacks.adv_box.attacks import FGSM,DeepFool,LinfPGDAttack




use_cuda=True


class config:
    image_nc = 1
    gen_input_nc = image_nc
    batch_size = 64
    test_batch_size = 8
    pert_clamp = 0.3
    box_min,box_max = -1,0
    adversary = 'GAN'

# Define what device we are using
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

# load the pretrained model
pretrained_model = "../model/adv_target_model_GAN.pth"
params = utils.params()
target_model = target_model_1(params).to(device)
target_model.load_state_dict(torch.load(pretrained_model,map_location=device))
target_model.eval()


# set adversary
"GAN adversary"
if config.adversary == 'GAN':
    print('adv with GAN')
    pretrained_generator_path = '../model/adv_generator.pth'
    pretrained_G = models.Generator(config.gen_input_nc, config.image_nc).to(device)
    pretrained_G.load_state_dict(torch.load(pretrained_generator_path, map_location=device))
    pretrained_G.eval()
elif config.adversary == 'FGSM':
    print('adv with FGSM')
    adversary = FGSM(epsilon=0.1)
elif config.adversary == 'DeepFool':
    print('adv with DeepFool')
    adversary = DeepFool(num_classes=5)
    # adversary = DeepFool_batch_train(num_classes=10)
elif config.adversary == 'PGD':
    print('adv with PGD')
    adversary = LinfPGDAttack(k=5, random_start=False)


#load data
train_dataloader = utils.load_data_main('../data/traffic_train.csv', config.batch_size) # input_shape (batch_size,1,wide 256)
test_dataloader = utils.load_data_main('../data/traffic_test.csv', config.test_batch_size)



# test adversarial examples in testing dataset
num_correct = 0

for i, data in enumerate(test_dataloader, 0):
    test_img, test_label = data
    test_img, test_label = test_img.to(device), test_label.to(device)

    if config.adversary in ['FGSM', 'DeepFool' , 'PGD']:
        adversary.model = target_model
        _, adv_x = adversary.perturbation(test_img, test_label)
    elif config.adversary == 'GAN':
        perturbation = pretrained_G(test_img)
        perturbation = torch.clamp(perturbation, -config.pert_clamp, config.pert_clamp)
        adv_x = perturbation + test_img
        adv_x = torch.clamp(adv_x, config.box_min, config.box_max)

    pred_lab = torch.argmax(target_model(adv_x.to(device)), 1)
    num_correct += torch.sum(pred_lab==test_label,0)

print('num_correct: ', num_correct.item())
print('total_num: ', len(test_dataloader) * config.test_batch_size)
print('accuracy of adv in testing set: %f\n' % (num_correct.item() / (len(test_dataloader) * config.test_batch_size)))
print('success rate of adv in testing set: %f\n' % (1-(num_correct.item() / (len(test_dataloader) * config.test_batch_size))))

