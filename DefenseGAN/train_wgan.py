import os
os.sys.path.append('..')
import numpy as np
from gan_model import Generator, Discriminator,Discriminator_0, generator, discriminator
from torch.autograd import Variable
import torch.optim as optim

import torch.autograd as autograd
import torch
import utils_DenGAN
import utils
import time
import copy
from sklearn.model_selection import train_test_split


"parameters init"
opt = utils_DenGAN.get_args()
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


"load data"
print('loading data...')
train_data = utils.load_data_main('../data/traffic_train.csv', opt.batch_size) # input_shape (batch_size,1,wide 256)
test_data = utils.load_data_main('../data/traffic_test.csv', opt.test_batch_size)
print('train_loader size: {} * {} = {}'.format(len(train_data),opt.batch_size,len(train_data)*opt.batch_size))
print('test_loader size: {} * {} = {}'.format(len(test_data),opt.test_batch_size,len(test_data)*opt.test_batch_size))


"partial borrow from https://github.com/sky4689524/DefenseGAN-Pytorch"
class train_wgan:

    def __init__(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.train_data, self.test_data = train_data, test_data
        # self.loss_criterion = nn.CrossEntropyLoss()
        self.netD, self.netG = Discriminator().to(self.device), Generator().to(self.device)


    def adjust_lr(self, optimizer, iteration, init_lr=1e-4, total_iteration=200000):

        gradient = (float(-init_lr) / total_iteration)

        lr = gradient * iteration + init_lr

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


    def compute_gradient_penalty(self,D, real_samples, fake_samples):
        "Calculates the gradient penalty loss for WGAN GP"

        # Random weight term for interpolation between real and fake samples
        alpha = Tensor(np.random.random((real_samples.size(0), 1, 1)))
        alpha = alpha.expand(real_samples.size(0), real_samples.size(1), real_samples.size(2))
        alpha = alpha.to(self.device)

        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = D(interpolates)
        fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
        fake.to(self.device)

        # Get gradient w.r.t. interpolates
        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        return gradient_penalty


    def train_GAN(self):

        # set parameters
        ITERS = opt.ITERS


        # set optimizer for generator and discriminator
        optimizerD = optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(0.5, 0.9))
        optimizerG = optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(0.5, 0.9))


        one = torch.tensor(1,dtype=torch.float)
        mone = one * -1

        one = one.to(self.device)
        mone = mone.to(self.device)


        # Training
        print('training start...')

        for iteration in range(opt.inital_epoch, ITERS, 1):

            print('training step {} ...',format(iteration))

            start_time = time.time()

            self.adjust_lr(optimizerD, iteration, init_lr=opt.lr, total_iteration=ITERS)
            self.adjust_lr(optimizerG, iteration, init_lr=opt.lr, total_iteration=ITERS)


            # for iter_d in range(CRITIC_ITERS):
            for i, (input_x, _) in enumerate(self.train_data):

                input_x = input_x.to(self.device)

                ############################
                # (1) Update D network
                ###########################
                for p in self.netD.parameters():  # reset requires_grad
                    p.requires_grad = True  # they are set to False below in netG update

                real_x = autograd.Variable(input_x)

                optimizerD.zero_grad()

                # Sample noise as generator input
                # z = autograd.Variable(torch.randn(input_x.size(0), opt.latent_dim))
                # z = z.to(self.device)
                z = Variable(Tensor(np.random.normal(0, 1, (input_x.size(0), opt.latent_dim))))

                # Generate a batch of x
                fake_x = self.netG(z).detach()
                fake_x = fake_x.to(self.device)

                # Real inputs
                real_validity = self.netD(real_x)
                d_loss_real = real_validity.mean()
                d_loss_real.backward(mone)

                # Fake inputs
                fake_validity = self.netD(fake_x)
                d_loss_fake = fake_validity.mean()
                d_loss_fake.backward(one)

                # Gradient penalty
                gradient_penalty = self.compute_gradient_penalty(self.netD, real_x.data, fake_x.data)
                gradient_penalty.backward()

                # Adversarial loss
                loss_D = d_loss_fake - d_loss_real + opt.lambda_gp * gradient_penalty

                # loss_D.backward(retain_graph=True)
                optimizerD.step()

                optimizerG.zero_grad()

                del real_validity, fake_validity, fake_x, gradient_penalty, real_x

                # Train the generator every n_critic iterations

                if (i + 1) % opt.n_critic == 0 or (i + 1) == len(self.train_data):

                    ############################
                    # (2) Update G network
                    ###########################
                    for p in self.netD.parameters():
                        p.requires_grad = False  # to avoid computation

                    # Generate a batch of images
                    fake_x = self.netG(z).cpu()
                    fake_x = fake_x.to(self.device)

                    # Loss measures generator's ability to fool the discriminator
                    # Train on fake images
                    fake_validity = self.netD(fake_x)
                    g_loss = fake_validity.mean()
                    g_loss.backward(mone)
                    loss_G = -g_loss

                    # loss_G.backward()
                    optimizerG.step()

                    del fake_validity


            "display results every interval_steps"
            if (iteration + 1) % opt.display_steps == 0 or (iteration + 1) == ITERS:
                print('batch {:>3}/{:>3}, D_cost {:.4f}, G_cost {:.4f}\r' \
                      .format(iteration + 1, ITERS, loss_D.item(), loss_G.item()))

            # save generator model after certain iteration
            if (iteration + 1) % 20 == 0:

                modelG_copy = copy.deepcopy(self.netG)
                torch.save(modelG_copy.state_dict(), '../model/DefenseGan_G.pth')

                modelD_copy = copy.deepcopy(self.netD)
                torch.save(modelD_copy.state_dict(), '../model/DefenseGan_D.pth')

                del modelG_copy, modelD_copy




    def test_process(self):

        costs_avg = 0.0
        disc_count = 0

        # test GAN model
        with torch.no_grad():
            for x, _ in test_data:
                x = x.to(self.device)

                D = self.netD(x)

                costs_avg += -D.mean().cpu().data.numpy()
                disc_count += 1

                del x

        costs_avg = costs_avg / disc_count

        print('test disc cost of D : {:.4f}'.format(costs_avg))



def train_main():
    training = train_wgan()
    training.train_GAN()
    training.test_process()


if __name__ == '__main__':
    train_main()




