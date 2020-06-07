import torch
from advGAN import models
import read_data
import numpy as np
import utils
from adv_box.attacks import FGSM,DeepFool,DeepFool_batch_train,LinfPGDAttack



class config:
    use_cuda = True
    image_nc = 1
    gen_input_nc = image_nc
    batch_size = 1
    test_batch_size = 1
    adversary = 'FGSM'


# Define GPU or CPU device
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if (config.use_cuda and torch.cuda.is_available()) else "cpu")


# load target model
pretrained_model = "../model/target_model.pth"
params = utils.params()
target_model = models.target_model_1(params).to(device)
target_model.load_state_dict(torch.load(pretrained_model,map_location=device))
target_model.eval()


# load train and test data
train_dataloader = utils.load_data_main('../data/traffic_train.csv', config.batch_size) # input_shape (batch_size,1,wide 256)
test_dataloader = utils.load_data_main('../data/traffic_test.csv', config.test_batch_size)


# set adversary

if config.adversary == 'GAN':
    print('adv with GAN')
    pretrained_generator_path = '../model/adv_generator.pth'
    pretrained_G = models.Generator(config.gen_input_nc, config.image_nc).to(device)
    pretrained_G.load_state_dict(torch.load(pretrained_generator_path,map_location=device))
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



traffics = []
adv_traffics = []
noise = []
for i, data in enumerate(train_dataloader, 0):

    traffic, label = data
    traffic, label = traffic.to(device), label.to(device)
    if config.adversary == 'FGSM' or 'DeepFool' or 'PGD':
        adversary.model = target_model
        _,adv_x = adversary.perturbation(traffic,label)
    elif config.adversary == 'GAN':
        perturbation = pretrained_G(traffic)
        perturbation = torch.clamp(perturbation, -0.3, 0.3)
        adv_x = perturbation + traffic
        adv_x = torch.clamp(adv_x, 0, 1)

    pert = adv_x.data.cpu().numpy() - traffic.data.cpu().numpy()

    traffics.append(np.squeeze(traffic.data.cpu().numpy()))
    adv_traffics.append(np.squeeze(adv_x.data.cpu().numpy()))
    noise.append(np.squeeze(pert))

    "subplot multi-fig"
    if (i+1) % 4 == 0:
        fig_id = (i+1) // 4
        utils.traffic_plot(fig_id,traffics,adv_traffics)
        traffics = []
        adv_traffics = []
        noise = []
    if i+1 > 13:
        break

    "single fig"
    # if i + 1 < 5:
    #     fig_id = i
    #     utils.single_traffic_plot(fig_id,traffics[i],adv_traffics[i])
        # utils.noise_plot('pert'+str(fig_id),noise[i])





