import torch.nn as nn
import torch.nn.functional as F
import math



class target_model(nn.Module):
    def __init__(self):
        super(target_model, self).__init__()
        self.conv1 = nn.Sequential(         # 1D（height,wide)->>(1,256)
            nn.Conv1d(in_channels=1,out_channels=32,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool1d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(             #(32,128)
            nn.Conv1d(32,32,3,1,1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(2)
        )
        self.conv3 = nn.Sequential(             #(32,64)
            nn.Conv1d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(2)
        )
        self.conv4 = nn.Sequential(             #(64,32)
            nn.Conv1d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(2)
        )                                       #(64,16)

        self.out = nn.Linear(64*16,101)


    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0),-1)
        logits = self.out(x)
        return logits


class target_model_1(nn.Module):
    "imroved one with around 85% acc"

    def __init__(self,params):
        self.params = params
        super(target_model_1, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=self.params['conv1_input_channel'], out_channels=self.params['conv1_output_channel'],
                      kernel_size=self.params['kernel_size1'], stride=self.params['stride1'], padding=self.params['padding1']),
            nn.ReLU(),
            nn.Dropout(self.params['drop_rate1']),
            nn.MaxPool1d(kernel_size=self.params['pool1'])
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=self.params['conv2_input_channel'], out_channels=self.params['conv2_output_channel'],
                      kernel_size=self.params['kernel_size2'], stride=self.params['stride2'],
                      padding=self.params['padding2']),
            nn.ReLU(),
            nn.Dropout(self.params['drop_rate2']),
            nn.MaxPool1d(kernel_size=self.params['pool2'])
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=self.params['conv3_input_channel'], out_channels=self.params['conv3_output_channel'],
                      kernel_size=self.params['kernel_size3'], stride=self.params['stride3'],
                      padding=self.params['padding3']),
            nn.ReLU(),
            nn.Dropout(self.params['drop_rate3']),
            nn.MaxPool1d(kernel_size=self.params['pool3'])
        )

        self.conv4 = nn.Sequential(
            nn.Conv1d(in_channels=self.params['conv4_input_channel'], out_channels=self.params['conv4_output_channel'],
                      kernel_size=self.params['kernel_size4'], stride=self.params['stride4'],
                      padding=self.params['padding4']),
            nn.ReLU(),
            nn.Dropout(self.params['drop_rate4']),
            nn.MaxPool1d(kernel_size=self.params['pool4'])
        )

        self.out_param1 = math.ceil(math.ceil(math.ceil(math.ceil(self.params['input_size']/self.params['pool1'])/self.params['pool2'])/self.params['pool3'])/self.params['pool4'])
        self.out = nn.Linear(self.params['conv4_output_channel']*self.out_param1,self.params['num_classes'])


    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0),-1)
        logits = self.out(x)
        return logits


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(          #(1,248,1)
                in_channels=1,
                out_channels=128, # use 128 filters to extract features; equal to next conv input
                kernel_size =5,
                stride=1,
                padding=2, #if stride = 1, padding = (kernel_size-stride)/2
            ),  #-> (128,248,1) 128层的特征提取窗口, 采用padding，后两维同输入
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.ReLU(),  #-> (128,248,1)
            # pool default value = kernel_size
            nn.MaxPool1d(kernel_size=2),    #-> (128,124,1)
        )
        self.conv2 = nn.Sequential( #-> (128,125,1), in channnels = 输入第一维
            nn.Conv1d(128,256,5,1,2),    #-> (256,124,1)
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.MaxPool1d(2) #-> (256,62,1)
        )
        self.out = nn.Linear(256 * 62 * 1,101)      #392->98, 248->62

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)          # (batch,32,100,1)
        x = x.view(x.size(0),-1) #保留batch维度并展平 (x.size(0)保留batch维度， （batch, 32*100*1)
        output = self.out(x)
        return output


class Discriminator(nn.Module):
    def __init__(self, image_nc):
        super(Discriminator, self).__init__()
        # MNIST: 1*28*28
        model = [
            nn.Conv1d(image_nc, 8, kernel_size=4, stride=2, padding=0, bias=True),
            nn.LeakyReLU(0.2),
            # 8*13*13
            nn.Conv1d(8, 16, kernel_size=4, stride=2, padding=0, bias=True),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2),
            # 16*5*5
            nn.Conv1d(16, 32, kernel_size=4, stride=2, padding=0, bias=True),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Conv1d(32, 1, 1),
            nn.Sigmoid()
            # 32*1*1
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        output = self.model(x).squeeze()
        return output


class Generator(nn.Module):
    def __init__(self,
                 gen_input_nc,
                 image_nc,
                 ):
        super(Generator, self).__init__()

        encoder_lis = [
            # MNIST:1*28*28
            nn.Conv1d(gen_input_nc, 8, kernel_size=3, stride=1, padding=0, bias=True),
            nn.InstanceNorm1d(8),
            nn.ReLU(),
            # 8*26*26
            nn.Conv1d(8, 16, kernel_size=3, stride=2, padding=0, bias=True),
            nn.InstanceNorm1d(16),
            nn.ReLU(),
            # 16*12*12
            nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=0, bias=True),
            nn.InstanceNorm1d(32),
            nn.ReLU(),
            # 32*5*5
        ]

        bottle_neck_lis = [ResnetBlock(32),
                       ResnetBlock(32),
                       ResnetBlock(32),
                       ResnetBlock(32),]

        decoder_lis = [
            nn.ConvTranspose1d(32, 16, kernel_size=3, stride=2, padding=0, bias=False),
            nn.InstanceNorm1d(16),
            nn.ReLU(),
            # state size. 16 x 11 x 11
            nn.ConvTranspose1d(16, 8, kernel_size=3, stride=2, padding=0, bias=False),
            nn.InstanceNorm1d(8),
            nn.ReLU(),
            # state size. 8 x 23 x 23
            nn.ConvTranspose1d(8, image_nc, kernel_size=6, stride=1, padding=0, bias=False),
            nn.Tanh()
            # state size. image_nc x 28 x 28
        ]

        self.encoder = nn.Sequential(*encoder_lis)
        self.bottle_neck = nn.Sequential(*bottle_neck_lis)
        self.decoder = nn.Sequential(*decoder_lis)

    def forward(self, x):
        x = self.encoder(x)
        x = self.bottle_neck(x)
        x = self.decoder(x)
        return x


# Define a resnet block
# modified from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type='reflect', norm_layer=nn.BatchNorm1d, use_dropout=False, use_bias=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad1d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad1d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv1d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad1d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad1d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv1d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out
