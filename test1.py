import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import codecs
import time
import data
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import numpy as np

EMBEDDING_DIM = 50
HIDDEN_DIM = 100
BATCH_SIZE = 100
EPOCH_NUM = 20


class LSTMClassifier(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=1, dropout=0, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, 2)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.zeros(1, BATCH_SIZE, self.hidden_dim),
                torch.zeros(1, BATCH_SIZE, self.hidden_dim))

    def forward(self, sentence, length):
        embeds = self.word_embeddings(sentence)

        embeds = torch.nn.utils.rnn.pack_padded_sequence(embeds, length, batch_first=True)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)

        length = length - 1
        out = lstm_out[np.arange(lstm_out.shape[0]), length, :]

        out = F.relu(out)
        out = self.hidden2tag(out)

        tag_scores = torch.log_softmax(out, dim=1)

        return tag_scores




#***************************

  self.lstm = nn.LSTM(self.opts['input_size'],self.opts['hidden_dim'],num_layers=self.opts['num_layers'],
                            dropout=self.opts['dropout'],batch_first=self.opts['batch_first'])
        self.hidder2out = nn.Linear(self.opts['hidden_dim'],self.opts['num_classes'])


        def init_hidden(self):

            return(torch.zeros(1,self.opts['batch_size'],self.opts['hidden_dim']),
                   torch.zeros(1,self.opts['batch_size'],self.opts['hidden_dim']))


        def forward(self,x):
            lstm_out,self.hidden = self.lstm(x,self.hidden)
            out =