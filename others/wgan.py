import data_loader, conv_models
import torch
from torch.autograd import Variable
import numpy as np
import torch.nn as nn


class Config:
    batch_size = 64
    test_batch_size = 1
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
            nn.Tanh()
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




# obtain device placeholder
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# load data
path = 'test.csv'
# path = 'generic_class.csv'
train_loader,test_loader = data_loader.main(path,Config)
print('train_loader size: {}'.format(len(train_loader)))
print('test_loader size: {}'.format(len(test_loader)))

# build GAN
generator = Generator()
discriminator = Discriminator()
generator.to(device)
discriminator.to(device)

# optimizers
optimizer_G = torch.optim.RMSprop(generator.parameters(),lr=Config.learning_rate)
optimizer_D = torch.optim.RMSprop(discriminator.parameters(),lr=Config.learning_rate)

# Training
batches_done = 0
for epoch in range(1,Config.epochs+1):

    for i, (data,_) in enumerate(train_loader):

        # configure input
        real_data = Variable(data.to(device))

        # ----------------------
        # train discriminator
        # ----------------------

        optimizer_D.zero_grad()

        # sample noise as generator input
        z = torch.Tensor(np.random.normal(0, 1, (data.shape[0],Config.latent_dim)))   # mean=0,standard deviation=1 Gaussian distribution
        z = Variable(z.to(device))

        # generate a batch of data
        fake_data = generator(z).detach()   # return the results to fake_data (with grad False), in the mean time, remove the tensor from the computating graph

        # print('fake_data',fake_data.shape)

        # adversarial loss
        loss_D = torch.mean(discriminator(fake_data)) - torch.mean(discriminator(real_data))

        loss_D.backward()
        optimizer_D.step()

        # clip weights of discriminator
        # ????????
        for p in discriminator.parameters():
            p.data.clamp_(-Config.clip_value,Config.clip_value)

        # train the generator every n_discriminator iteration
        if i % Config.n_discriminator == 0:

            # -------------------
            # train generator
            # -------------------

            optimizer_G.zero_grad()

            # generate a batch of data
            gen_data = generator(z)

            # adversarial loss
            loss_G = -torch.mean(discriminator(gen_data))

            loss_G.backward()
            optimizer_G.step()

            print(
                '[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]'
                % (epoch,Config.epochs,batches_done % len(train_loader), len(train_loader), loss_D.item(),loss_G.item())
            )
        batches_done += 1

        # save the sample
        # if batches_done % Config.sample_interval == 0:

torch.save(generator.state_dict(),'./model/generator.pth')
torch.save(discriminator.state_dict(),'./model/discriminator.pth')

