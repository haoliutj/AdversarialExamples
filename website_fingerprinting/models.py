import torch.nn as nn
import math


class target_model(nn.Module):
    "imroved one with around 85% acc"

    def __init__(self,params):
        self.params = params
        super(target_model, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=self.params['conv1_input_channel'], out_channels=self.params['conv1_output_channel'],
                      kernel_size=self.params['kernel_size1'], stride=self.params['stride1'], padding=self.params['padding1']),
            nn.ReLU(),
            nn.Dropout(self.params['drop_rate1']),
            nn.BatchNorm1d(self.params['conv1_output_channel']),
            nn.MaxPool1d(kernel_size=self.params['pool1'])
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=self.params['conv2_input_channel'], out_channels=self.params['conv2_output_channel'],
                      kernel_size=self.params['kernel_size2'], stride=self.params['stride2'],
                      padding=self.params['padding2']),
            nn.ReLU(),
            nn.Dropout(self.params['drop_rate2']),
            nn.BatchNorm1d(self.params['conv2_output_channel']),
            nn.MaxPool1d(kernel_size=self.params['pool2'])
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=self.params['conv3_input_channel'], out_channels=self.params['conv3_output_channel'],
                      kernel_size=self.params['kernel_size3'], stride=self.params['stride3'],
                      padding=self.params['padding3']),
            nn.ReLU(),
            nn.Dropout(self.params['drop_rate3']),
            nn.BatchNorm1d(self.params['conv3_output_channel']),
            nn.MaxPool1d(kernel_size=self.params['pool3'])
        )

        self.conv4 = nn.Sequential(
            nn.Conv1d(in_channels=self.params['conv4_input_channel'], out_channels=self.params['conv4_output_channel'],
                      kernel_size=self.params['kernel_size4'], stride=self.params['stride4'],
                      padding=self.params['padding4']),
            nn.ReLU(),
            nn.Dropout(self.params['drop_rate4']),
            nn.BatchNorm1d(self.params['conv4_output_channel']),
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