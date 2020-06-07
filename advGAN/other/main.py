import torch
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from advGAN import AdvGAN_Attack
from models import Net
import data_loader

use_cuda=True
image_nc=1
epochs = 60
batch_size = 128
BOX_MIN = -1
BOX_MAX = 0

class opt:
    batch_size = 64
    test_batch_size = 1

# Define what device we are using
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

pretrained_model = "cnn_torch.pth"
targeted_model = Net().to(device)
targeted_model.load_state_dict(torch.load(pretrained_model,map_location=device))
targeted_model.eval()
model_num_labels = 101

# MNIST train dataset and dataloader declaration
path = '../generic_class.csv'
train_dataloader,test_dataloader = data_loader.main(path,opt) # input_shape (batch_size,1,wide 256)
advGAN = AdvGAN_Attack(device,
                          targeted_model,
                          model_num_labels,
                          image_nc,
                          BOX_MIN,
                          BOX_MAX)

advGAN.train(train_dataloader, epochs)
