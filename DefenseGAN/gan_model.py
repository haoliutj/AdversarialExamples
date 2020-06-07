import torch
import torch.nn as nn
import numpy as np
import utils_DenGAN




#***************************************************
"initial version of WGAN model"

opt = utils_DenGAN.get_args()
input_shape = (opt.height, opt.width)

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
            *block(opt.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(input_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *input_shape)  # img.shape[0]=channels (or batch_size), so input_shape=[H,W] instead of input_shape=[channels,H,W]
        return img


class Discriminator_0(nn.Module):
    def __init__(self):
        super(Discriminator_0, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(input_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity


"improved Discriminator with 3 layers CNN structure"
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()

        self.conv1 = nn.Sequential(  # 1D（height,wide)->>(1,256)
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
        )
        self.conv2 = nn.Sequential(  # (128,256)
            nn.Conv1d(64, 128, 3, 1, 1),
            nn.LeakyReLU(0.2),
        )
        self.conv3 = nn.Sequential(  # (256,256)
            nn.Conv1d(128, 256, 3, 1, 1),
            nn.LeakyReLU(0.2)
        )

        self.out = nn.Linear(256 * 256, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        vadility = self.out(x)
        return vadility




#**********************************************************************************
"similar structure in Defense_GAN paper"


DIM = 64  # This overfits substantially; you're probably better off with 64

class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()

        preprocess = nn.Sequential(
            nn.Linear(opt.latent_dim,256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
        )

        block1 = nn.Sequential(
            nn.ConvTranspose1d(1, 2 * DIM, 2, stride=2),
            nn.BatchNorm1d(2 * DIM),
            nn.ReLU(True),
        )
        block2 = nn.Sequential(
            nn.ConvTranspose1d(2 * DIM, DIM, 2, stride=2),
            nn.BatchNorm1d(DIM),
            nn.ReLU(True),
        )
        deconv_out = nn.ConvTranspose1d(DIM, int(np.prod(input_shape)), 2, stride=2)


        self.preprocess = preprocess
        self.block1 = block1
        self.block2 = block2
        self.deconv_out = deconv_out
        self.tanh = nn.Tanh()

    def forward(self, input):
        output = self.preprocess(input)
        output = output.view(-1, *input_shape)
        output = self.block1(output)
        output = self.block2(output)
        output = self.deconv_out(output)
        output = self.tanh(output)
        return output.view(input.shape[0], *input_shape)


class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        main = nn.Sequential(
            nn.Conv1d(1, DIM, 3, 2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(DIM, 2 * DIM, 3, 2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(2 * DIM, 4 * DIM, 3, 2, padding=1),
            nn.LeakyReLU(0.2),
        )

        self.main = main
        self.linear = nn.Linear(4 * 4 * 4 * DIM, 1)

    def forward(self, input):
        output = self.main(input)
        output = output.view(-1, 4 * 4 * 4 * DIM)
        output = self.linear(output)
        return output