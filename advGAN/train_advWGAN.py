import os
os.sys.path.append('..')
import torch
from advWGAN import advGan_attack
from models import target_model_1
import utils


def params():
    return {
        'conv1_input_channel': 1,
        'conv2_input_channel': 128,
        'conv3_input_channel': 128,
        'conv4_input_channel': 64,
        'conv1_output_channel': 128,
        'conv2_output_channel': 128,
        'conv3_output_channel': 64,
        'conv4_output_channel': 256,
        'kernel_size1': 7,
        'kernel_size2': 19,
        'kernel_size3': 13,
        'kernel_size4': 23,
        'stride1': 1,
        'stride2': 1,
        'stride3': 1,
        'stride4': 1,
        'padding1': 3,
        'padding2': 9,
        'padding3': 6,
        'padding4': 11,
        'drop_rate1': 0.1,
        'drop_rate2': 0.3,
        'drop_rate3': 0.1,
        'drop_rate4': 0.0,
        'pool1': 2,
        'pool2': 2,
        'pool3': 2,
        'pool4': 2,
        'num_classes': 101,
        'dim': 256
    }


use_cuda=True
input_nc=1
epochs = 60
x_box_min = -1
x_box_max = 0
pert_box_min = -0.3
pert_box_max = 0.3

class opt:
    batch_size = 64
    test_batch_size = 1

# Define what device we are using
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

pretrained_model = "../model/target_model.pth"
params = params()
targeted_model = target_model_1(params).to(device)
targeted_model.load_state_dict(torch.load(pretrained_model,map_location=device))
targeted_model.eval()
model_num_labels = 101


# load data
train_dataloader = utils.load_data_main('../traffic_train.csv',opt.batch_size) # input_shape (batch_size,1,wide 256)
# test_dataloader = utils.load_data_main('../traffic_test.csv',opt.test_batch_size)

advGAN = advGan_attack(device,
                       targeted_model,
                       model_num_labels,
                       input_nc,
                       x_box_min,
                       x_box_max,
                       pert_box_min,pert_box_max)

advGAN.train(train_dataloader, epochs)
