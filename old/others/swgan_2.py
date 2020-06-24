"""
Different here:
    1. train D k steps, G 1 step
    2. change loss to wgan-gp

"""

import argparse
import os
import numpy as np
import math

# import torchvision.transforms as transforms
# from torchvision.utils import save_image

from torch.utils.data import DataLoader
# from torchvision import datasets
from torch.autograd import Variable
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch
import data_loader
import time

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the train batches")
parser.add_argument("--test_batch_size", type=int, default=1, help="size of the test batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--num_classes", type=int, default=101, help="number of classes for dataset")
parser.add_argument("--input_hight", type=int, default=1, help="size of each image dimension")
parser.add_argument("--input_wide", type=int, default=256, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
parser.add_argument("--critic_iters", type=int, default=5, help="interval for training G")
parser.add_argument("--display_steps", type=int, default=5, help="interval for display info")
parser.add_argument("--Lambda", type=int, default=10, help="gradient penalty hyperparameter")

opt = parser.parse_args()
print(opt)


cuda = True if torch.cuda.is_available() else False


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(opt.num_classes, opt.latent_dim)

        self.init_hight = int(math.ceil(opt.input_hight / 4))  # math.ceil(2.3) = 3; Initial size before upsampling
        self.init_wide = int(math.ceil(opt.input_wide / 4))  # Initial size before upsampling
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128 * self.init_hight * self.init_wide))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise):
        out = self.l1(noise)
        out = out.view(out.shape[0], 128, self.init_hight, self.init_wide)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            """Returns layers of each discriminator block"""
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.conv_blocks = nn.Sequential(
            *discriminator_block(opt.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_hight = int(math.ceil(opt.input_hight / 2 ** 4))
        ds_wide = int(math.ceil(opt.input_wide / 2 ** 4))

        # Output layers
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_hight * ds_wide, 1), nn.Sigmoid())
        self.aux_layer = nn.Sequential(nn.Linear(128 * ds_hight * ds_wide, opt.num_classes + 1), nn.Softmax())

    def forward(self, img):
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        label = self.aux_layer(out)

        return validity, label


# ---------------------------------------------------
# To Calculate the gradient penalty loss for WGAN GP
# ---------------------------------------------------

def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(torch.Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
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



# Loss functions
adversarial_loss = torch.nn.BCELoss()
auxiliary_loss = torch.nn.CrossEntropyLoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
generator.to(device)
discriminator.to(device)
adversarial_loss.to(device)
auxiliary_loss.to(device)


# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Configure data loader
# path = 'test.csv'
path = 'generic_class.csv'
train_loader,test_loader = data_loader.main(path,opt)
print('train_loader size: {}'.format(len(train_loader)))
print('test_loader size: {}'.format(len(test_loader)))

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

# ----------
#  Training
# ----------


for epoch in range(1,opt.n_epochs+1):

    start_time = time.time()

    for i, (imgs, labels) in enumerate(train_loader):

        # [batch_size,1,input_wide] to [batch_size,1,1,input_wide]
        imgs = imgs.unsqueeze(1)

        batch_size = imgs.shape[0]

        # Adversarial ground truths
        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)
        # all fake instances with labels (num_classes), the original labels for real is [0,num_classes)
        fake_aux_gt = Variable(LongTensor(batch_size).fill_(opt.num_classes), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(FloatTensor))
        labels = Variable(labels.type(LongTensor))

        # ---------------------
        #  Train Discriminator
        # ---------------------

        for p in discriminator.parameters():
            p.requires_grad = True

        optimizer_D.zero_grad()

        # Sample noise as generator input
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
        z = z.to(device)

        # generate a batch of traffics
        fake_imgs = generator(z)

        # Loss for real images
        real_validity, real_aux = discriminator(real_imgs)
        d_real_loss = real_validity.mean()
        d_real_loss.backward()
        # d_real_loss = (adversarial_loss(real_validity, valid) + auxiliary_loss(real_aux, labels)) / 2

        # Loss for fake images
        fake_validity, fake_aux = discriminator(fake_imgs.detach())
        d_fake_loss = fake_validity.mean()
        d_fake_loss.backward()
        # d_fake_loss = (adversarial_loss(fake_validity, fake) + auxiliary_loss(fake_aux, fake_aux_gt)) / 2

        # Gradient penalty
        gradient_penalty = compute_gradient_penalty(discriminator,real_imgs.data,fake_imgs.data)
        gradient_penalty.backward()

        # Total Discriminator loss
        d_loss = d_fake_loss - d_real_loss + gradient_penalty * opt.Lambda
        # d_loss = (d_real_loss + d_fake_loss) / 2

        # Calculate discriminator accuracy
        pred = np.concatenate([real_aux.data.cpu().numpy(), fake_aux.data.cpu().numpy()], axis=0)
        gt = np.concatenate([labels.data.cpu().numpy(), fake_aux_gt.data.cpu().numpy()], axis=0)
        d_acc = np.mean(np.argmax(pred, axis=1) == gt)

        optimizer_D.step()

        del real_validity,fake_validity,real_imgs,fake_imgs


        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        if (i+1) % opt.critic_iters == 0 or (i + 1) == len(train_loader):

            for p in discriminator.parameters():
                p.requires_grad = False         # to avoid computation

            # Generate a batch of images
            gen_imgs = generator(z)

            # Loss measures generator's ability to fool the discriminator
            fake_validity, _ = discriminator(gen_imgs)
            g_loss = -torch.mean(fake_validity)

            # g_loss = adversarial_loss(validity, valid)

            g_loss.backward()
            optimizer_G.step()

            del fake_validity

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %d%%] [G loss: %f]"
                % (epoch, opt.n_epochs, i, len(train_loader), d_loss.item(), 100 * d_acc, g_loss.item())
            )

        # batches_done = epoch * len(train_loader) + i
        # if batches_done % opt.sample_interval == 0:
        #     save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)


    # if (epoch+1) % opt.display_steps == 0:
    #     print(
    #         "[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %d%%] [G loss: %f]"
    #         % (epoch, opt.n_epochs, i, len(train_loader), d_loss.item(), 100 * d_acc,g_loss.item())
    #     )


torch.save(generator.state_dict(),'./model/s_generator.pth')
torch.save(discriminator.state_dict(),'./model/s_discriminator.pth')


# ---------------
#   testing
# ---------------
generator_trained = Generator()
discriminator_trained = Discriminator()
generator_trained.load_state_dict(torch.load('./model/s_generator.pth'))
discriminator_trained.load_state_dict(torch.load('./model/s_discriminator.pth'))
generator_trained.to(device)
discriminator_trained.to(device)
generator_trained.eval()
discriminator_trained.eval()
adversarial_loss.to(device)
auxiliary_loss.to(device)

correct_total = 0
fake_correct_total = 0
real_correct_total = 0
total_loss = 0


for i, (data,label) in enumerate(test_loader):

    print('testing {}'.format(i))

    # ----------------------
    # test on test dataset
    # ----------------------
    data = data.unsqueeze(1)
    data, label = Variable(data.to(device)), Variable(label.to(device))

    # build fake ground truth
    valid = Variable(FloatTensor(data.shape[0], 1).fill_(1.0), requires_grad=False)
    fake = Variable(FloatTensor(data.shape[0], 1).fill_(0.0), requires_grad=False)
    fake_aux_gt = Variable(LongTensor(data.shape[0]).fill_(opt.num_classes), requires_grad=False)

    # loss and acc on real test data
    real_preds, real_label_vec = discriminator_trained(data)
    real_loss = (adversarial_loss(real_preds, valid) + auxiliary_loss(real_label_vec, label)) / 2

    _, real_label = torch.max(real_label_vec, 1)
    real_correct = int(sum(real_label == label))


    # ---------------------------------------
    # test on samples generated by generator
    # ---------------------------------------
    z = Variable(FloatTensor(np.random.normal(0,1,(data.shape[0],opt.latent_dim))))
    gen_data = generator_trained(z)

    # loss and acc on generated data
    fake_preds, fake_label_vec = discriminator_trained(gen_data)
    fake_loss = (adversarial_loss(fake_preds, fake) + auxiliary_loss(fake_label_vec, fake_aux_gt)) / 2

    _, fake_label = torch.max(fake_label_vec,1)
    fake_correct = int(sum(fake_label == fake_aux_gt))

    # total loss and acc based on fake on real dataset
    fake_correct_total += fake_correct
    real_correct_total += real_correct
    total_loss += ((real_loss + fake_loss) * opt.test_batch_size)/ 2


# acc & loss
fake_acc = fake_correct_total / len(test_loader)
real_acc = real_correct_total / len(test_loader)
correct_total = real_correct_total + fake_correct_total
ave_acc = correct_total / (2 * len(test_loader))
ave_loss = total_loss / len(test_loader)
print('Loss: {:5.2f}, Accuracy: {:6.2%}'.format(ave_loss,ave_acc))
print('running time is {:0.2f} seconds.'.format(time.time() - start_time))


