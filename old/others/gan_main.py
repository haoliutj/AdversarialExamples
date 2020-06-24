import data_loader, conv_models
import torch
from torch.autograd import Variable
import numpy as np



class Config:
    batch_size = 64
    test_batch_size = 1
    learning_rate = 0.00005
    epochs = 50
    input_seq_length = 256
    clip_value = 0.01       # lower and upper clip value for disc. weights
    n_discriminator = 5     # number of training steps for discriminator per iteration
    sample_interval = 400   # interval between input data samples


# obtain device placeholder
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# load data
path = 'test.csv'
train_loader,test_loader = data_loader.main(path,Config)
print('train_loader size: {}'.format(len(train_loader)))
print('test_loader size: {}'.format(len(test_loader)))

# build GAN
generator = conv_models.CausalConvGenerator(noise_size=256, output_size=1, n_layers=8, n_channel=10, kernel_size=8, dropout=0)
discriminator = conv_models.CausalConvDiscriminator(input_size=1, n_layers=8, n_channel=10, kernel_size=8, dropout=0)
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
        z = torch.Tensor(np.random.normal(0,1,(Config.batch_size,1,Config.input_seq_length)))   # mean=0,standard deviation=1 Gaussian distribution
        z = Variable(z.to(device))

        # generate a batch of data
        fake_data = generator(z).detach()   # return the results to fake_data (with grad False), in the mean time, remove the tensor from the computating graph

        # adversarial loss
        t1 = discriminator(fake_data)
        t2 = discriminator(real_data)
        loss_D = torch.mean(discriminator(fake_data)) - torch.mean(discriminator(real_data))

        loss_D.backward()
        optimizer_D.step()

        # clip weights of discriminator
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

        # save the sample
        # if batches_done % Config.sample_interval == 0:

torch.save(generator.state_dict(),'./model/generator.pth')
torch.save(discriminator.state_dict(),'./model/discriminator.pth')

