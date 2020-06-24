import os
os.sys.path.append('..')


import models
from models import target_model_1
import utils
import torch


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


# Define what device we are using
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load the pretrained model
pretrained_model = "../model/target_model.pth"
params = params()
target_model = target_model_1(params).to(device)
target_model.load_state_dict(torch.load(pretrained_model,map_location=device))
target_model.eval()

# load data
train_dataloader = utils.load_data_main('../data/traffic_train.csv',64) # input_shape (batch_size,1,wide 256)
test_dataloader = utils.load_data_main('../data/traffic_test.csv',64)
print('train_loader size: {}'.format(len(train_dataloader)))
print('test_loader size: {}'.format(len(test_dataloader)))

# test model
num_correct = 0
for i, data in enumerate(train_dataloader, 0):
    train_img, train_label = data
    train_img, train_label = train_img.to(device), train_label.to(device)
    pred_lab = torch.argmax(target_model(train_img), 1)
    num_correct += torch.sum(pred_lab == train_label, 0)

print('accuracy in training set: %f\n' % (num_correct.item() / (len(train_dataloader) * 64)))


num_correct = 0
for i, data in enumerate(test_dataloader, 0):
    test_img, test_label = data
    test_img, test_label = test_img.to(device), test_label.to(device)
    pred_lab = torch.argmax(target_model(test_img), 1)
    num_correct += torch.sum(pred_lab == test_label, 0)

print('accuracy in testing set: %f\n' % (num_correct.item() / (len(test_dataloader) * 64)))