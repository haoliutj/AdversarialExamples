import os
os.sys.path.append('..')
import torch
import models
from models import target_model_1
import matplotlib.pyplot as plt
import numpy as np
import utils



use_cuda=True
image_nc=1
gen_input_nc = image_nc
class opt:
    batch_size = 1
    test_batch_size = 1
    x_box_min= -1
    x_box_max = 0
    pert_box_min = -0.1
    pert_box_max = -pert_box_min


# Define GPU or CPU device
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")


# load target model
pretrained_model = "../model/target_model.pth"
params = utils.params()
target_model = target_model_1(params).to(device)
target_model.load_state_dict(torch.load(pretrained_model,map_location=device))
target_model.eval()


# load train and test data
train_dataloader = utils.load_data_main('../data/traffic_train.csv',opt.batch_size) # input_shape (batch_size,1,wide 256)
test_dataloader = utils.load_data_main('../data/traffic_test.csv',opt.test_batch_size)


# load the adversary
pretrained_generator_path = '../model/adv_generator.pth'
pretrained_G = models.Generator(gen_input_nc, image_nc).to(device)
pretrained_G.load_state_dict(torch.load(pretrained_generator_path,map_location=device))
pretrained_G.eval()



traffics = []
adv_traffics = []
noise = []
for i, data in enumerate(train_dataloader, 0):

    traffic, label = data
    traffic, label = traffic.to(device), label.to(device)
    perturbation = pretrained_G(traffic)
    perturbation = torch.clamp(perturbation, opt.pert_box_min,opt.pert_box_max)
    adv_img = perturbation + traffic
    adv_img = torch.clamp(adv_img,opt.x_box_min,opt.x_box_max)

    traffics.append(np.squeeze(traffic.data.cpu().numpy()))
    adv_traffics.append(np.squeeze(adv_img.data.cpu().numpy()))
    noise.append(np.squeeze(perturbation.data.cpu().numpy()))

    # "subplot multiple plot"
    # if (i+1) % 4 == 0:
    #     fig_id = (i+1) // 4
    #     utils.traffic_plot(fig_id,traffics,adv_traffics)
    #     traffics = []
    #     adv_traffics = []
    #     noise = []
    # if i+1 > 13:
    #     break

    "single fig"
    if i + 1 < 5:
        fig_id = i
        utils.single_traffic_plot(fig_id, traffics[i], adv_traffics[i])
        utils.noise_plot('pert' + str(fig_id), noise[i])




