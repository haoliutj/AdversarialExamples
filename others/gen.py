
import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import data_loader

class Config:
    batch_size = 64
    test_batch_size = 64
    learning_rate = 0.00005
    epochs = 50
    input_hight = 1
    input_wide = 256
    clip_value = 0.01       # lower and upper clip value for disc. weights
    n_discriminator = 5     # number of training steps for discriminator per iteration
    sample_interval = 400   # interval between input data samples
    latent_dim = 100        # the dimensionlity of the generator's first input channel. default 100, can change



input_shape = (1,Config.input_hight,Config.input_wide)     # (channels,h,w)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(Config.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(input_shape))),
            # nn.Tanh()
        )

    def forward(self, z):
        output = self.model(z)
        output = output.view(output.shape[0], *input_shape)
        return output


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(input_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, output):
        output_flat = output.view(output.shape[0], -1)
        validity = self.model(output_flat)
        return validity



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



# build generator
generator = Generator()
discriminator = Discriminator()
generator.load_state_dict(torch.load('./model/generator.pth',map_location=device))
discriminator.load_state_dict(torch.load('./model/discriminator.pth', map_location=device))
generator.eval()
discriminator.eval()
generator.to(device)
discriminator.to(device)


# random input
z = torch.Tensor(np.random.normal(0, 1, (Config.test_batch_size,Config.latent_dim)))   # mean=0,standard deviation=1 Gaussian distribution
z = Variable(z.to(device))

gen_data = generator(z)
p = discriminator(gen_data)
print(p.data)


# load data
path = 'test.csv'
# path = 'generic_class.csv'
train_loader,test_loader = data_loader.main(path,Config)

for data,_ in test_loader:
    data = data.view(-1,256)
    np.savetxt('gen_data_1.csv', data.data.numpy(), delimiter=',')

gen_data = gen_data.view(-1,256)
np.savetxt('gen_data.csv',gen_data.data.numpy(),delimiter=',')

