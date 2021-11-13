import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random


"""
Neural Networks model : Bidirection LSTM
"""


class BiLSTM(nn.Module):

    def __init__(self, input_dim,out_dim):
        super(BiLSTM, self).__init__()


        self.bilstm = nn.LSTM(input_dim, out_dim // 2, num_layers=1, batch_first=True, bidirectional=True,
                              bias=False)#out b*l*d
        print(self.bilstm)


    def forward(self, x,x_len,pad_maxlen=None):
        ##pad_maxlen不为None就Pad到padmaxlen长度，为None就pad到batch内非0的最大长度
        x_idx = x_len.argsort()[::-1]
        x=x[x_idx.tolist()]
        x_len1=x_len[x_idx]
        x_packed = torch.nn.utils.rnn.pack_padded_sequence(x, x_len1, batch_first=True)

        bilstm_out, _ = self.bilstm(x_packed)

        if(pad_maxlen):#pad_maxlen not none ,pad to the length
            x_paded, _ = torch.nn.utils.rnn.pad_packed_sequence(bilstm_out, batch_first=True,total_length=pad_maxlen)
        else:
            x_paded, _ = torch.nn.utils.rnn.pad_packed_sequence(bilstm_out, batch_first=True)

        r_idx = x_idx.argsort()
        x_paded=x_paded[r_idx]

        bilstm_out = F.relu(x_paded)
        return bilstm_out#b*l*c