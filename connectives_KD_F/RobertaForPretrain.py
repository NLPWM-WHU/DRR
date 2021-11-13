from torch.nn import CrossEntropyLoss, MSELoss
import torch
torch.cuda.current_device()
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
# from torchvision import datasets, transforms
# import torchvision.utils as vutils
import torch.nn.functional as F
import numpy as np
import bilstm
import nltk
from layers import *
import math
import random

# import torch
# from transformers import *
from transformers import RobertaTokenizer
from bertForMLM import BertForMaskedLM
import os

# UNCASED = '../BERT-BASE-UNCASED'
UNCASED = 'bert-base-uncased'
VOCAB = 'vocab.txt'
# os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda")
INF = 1e12
_INF = -1e12

_act_map = {"none": lambda x: x,
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "softmax": nn.Softmax(dim=-1),
            "sigmoid": nn.Sigmoid(),
            "leaky_relu": nn.LeakyReLU(),
            "prelu": nn.PReLU()}

def map_activation_str_to_layer(act_str):
    try:
        return _act_map[act_str]
    except:
        raise NotImplementedError("Error: %s activation fuction is not supported now." % (act_str))


class bertForPretrain(nn.Module):
    def __init__(self, class_num):
        self.roberta_size = 768
        super(bertForPretrain, self).__init__()
        self.bert_model = BertForMaskedLM.from_pretrained(UNCASED)
        self.set_finetune("full")

        self.num_labels = class_num
        # self.con_labels = conn_size

        self.dropout = torch.nn.Dropout(0.5)
        self.max_len = 200

        self.in_dim = self.roberta_size
        self.out_dim = self.roberta_size
        self.hidden_dim = 128
        self.num_perspectives = 16
        self.pos_middle_dim = 20
        self.out_pos_dim = self.out_dim
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.dropout1 = nn.Dropout(0.1)
        self.num_labels = class_num
        self.cls_classifier = nn.Linear(self.out_dim, self.num_labels)
        self.classifier = nn.Linear(self.out_dim, self.num_labels)
        # self.con_lass_lin = nn.Linear(self.out_dim, self.con_labels)
        self.cls_classifier_4 = nn.Linear(self.out_dim, 4)
        self.classifier_4 = nn.Linear(self.out_dim, 4)
        # transpose matrix
        self.trans_mat = torch.Tensor([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]]).float().cuda().transpose(0, 1)  # 11*4
        # self.trans_W = nn.Linear(4, 11, bias=False)
        # self.trans_W.weight.data.copy_(self.trans_mat)
        # self.trans_W = nn.Linear(4, self.out_dim, bias=False)

    def set_finetune(self, finetune):
        if finetune == "none":
            for param in self.bert_model.parameters():
                param.requires_grad = False
        elif finetune == "full":
            for param in self.bert_model.parameters():
                param.requires_grad = True
        elif finetune == "last":
            for param in self.bert_model.parameters():
                param.requires_grad = False
            for param in self.bert_model.encoder.layer[-1].parameters():
                param.requires_grad = True
        elif finetune == "type":
            for param in self.bert_model.parameters():
                param.requires_grad = False
            for param in self.bert_model.embeddings.token_type_embeddings.parameters():
                param.requires_grad = True

    def forward(self, x_tag, x_idx, data_name, epoch, y2=None, connect=None, test=False, alpha=0.1):
        # (x_id, x_seg, x_mask, x1_len) = x_idx
        # (hard_label, tag_4) = x_tag
        # outputs = self.bert_model(x_id, attention_mask=x_mask, token_type_ids=x_seg)
        # pooled_output = outputs[1]
        # last_hidden_state = outputs[0]
        # connect_emb = self.get_connective_emb(last_hidden_state, x1_len)
        # loss_fct = CrossEntropyLoss()
        #
        # cls_4_logit = self.cls_classifier_4(pooled_output)
        # con_4_logit = self.classifier_4(connect_emb)
        #
        # cls_4to_emb = self.trans_W(cls_4_logit)
        # con_4to_emb = self.trans_W(con_4_logit)
        # cls_11_emb = torch.cat((pooled_output, cls_4to_emb), dim=-1)
        # con_11_emb = torch.cat((connect_emb, con_4to_emb), dim=-1)
        # cls_logit = self.cls_classifier(cls_11_emb)
        # con_logit = self.classifier(con_11_emb)

        # loss1 = loss_fct(cls_logit.view(-1, self.num_labels), hard_label.view(-1))
        # loss2 = loss_fct(con_logit.view(-1, self.num_labels), hard_label.view(-1))
        # loss3 = loss_fct(cls_4_logit.view(-1, 4), tag_4.view(-1))
        # loss4 = loss_fct(con_4_logit.view(-1, 4), tag_4.view(-1))
        # con_class_loss = loss_fct(con_class_logit.view(-1, self.con_labels), conn_id.view(-1))
        # loss = 0.2 * con_class_loss + 0.8 * (loss1 + loss2)
        # loss = loss1 + loss2 + loss3 + loss4
        if not test:
            # bert_inputs = (X_input_ids, batch_segment, batch_mask, batch_X1_len, label_id, con_num)
            (x_id, x_seg, x_mask, x1_len, label_id, con_num) = x_idx
            (hard_label, flag, tag_4) = x_tag
            outputs = self.bert_model(x_id, attention_mask=x_mask, token_type_ids=x_seg)
            # predict_loss = outputs[0]
            prediction_scores = outputs[0]
            pooled_output = outputs[2]
            last_hidden_state = outputs[1]
            connect_emb = self.get_exp_con_emb(last_hidden_state, x1_len, con_num)

            cls_4_logit = self.cls_classifier_4(pooled_output)
            con_4_logit = self.classifier_4(connect_emb)
            cls_logit = self.cls_classifier(pooled_output)
            con_logit = self.classifier(connect_emb)

            a = alpha
            cls_11to4 = torch.matmul(cls_logit, self.trans_mat)
            con_11to4 = torch.matmul(con_logit, self.trans_mat)
            cls_4_logit = (1-a) * cls_4_logit + a * cls_11to4
            con_4_logit = (1-a) * con_4_logit + a * con_11to4
            # cls_4to11 = self.relu(self.trans_W(cls_4_logit))
            # con_4to11 = self.relu(self.trans_W(con_4_logit))
            # cls_logit = cls_logit + alpha * cls_4to11
            # con_logit = con_logit + alpha * con_4to11

            loss_fct = CrossEntropyLoss()
            loss1 = loss_fct(cls_logit.view(-1, self.num_labels), hard_label.view(-1))
            loss2 = loss_fct(con_logit.view(-1, self.num_labels), hard_label.view(-1))
            loss3 = loss_fct(cls_4_logit.view(-1, 4), tag_4.view(-1))
            loss4 = loss_fct(con_4_logit.view(-1, 4), tag_4.view(-1))
            # loss_fct = CrossEntropyLoss()
            sum_flag = torch.sum(flag) + 0.000001
            mlm_loss = 0
            for i in range(len(flag)):
                if flag[i] == 1:
                    score = prediction_scores[i]
                    mlm_loss += loss_fct(score.view(-1, self.bert_model.config.vocab_size), label_id[i].view(-1))
            mlm_loss = mlm_loss / sum_flag
            loss = 0.05 * mlm_loss + 0.95 * (loss1 + loss2 + loss3 + loss4)
            # mlm_loss = 0.0
            # loss = loss1 + loss2 + loss3 + loss4

            return loss, loss1, loss2, loss3, loss4, mlm_loss
        else:
            (x_id, x_seg, x_mask, x1_len) = x_idx
            (hard_label, tag_4) = x_tag
            outputs = self.bert_model(x_id, attention_mask=x_mask, token_type_ids=x_seg)
            pooled_output = outputs[2]
            last_hidden_state = outputs[1]
            connect_emb = self.get_connective_emb(last_hidden_state, x1_len)

            cls_4_logit = self.cls_classifier_4(pooled_output)
            con_4_logit = self.classifier_4(connect_emb)
            cls_logit = self.cls_classifier(pooled_output)
            con_logit = self.classifier(connect_emb)

            # alpha = 0.2
            #
            # cls_4to11 = self.relu(self.trans_W(cls_4_logit))
            # con_4to11 = self.relu(self.trans_W(con_4_logit))
            # cls_logit = cls_logit + alpha * cls_4to11
            # con_logit = con_logit + alpha * con_4to11
            a = alpha
            cls_11to4 = torch.matmul(cls_logit, self.trans_mat)
            con_11to4 = torch.matmul(con_logit, self.trans_mat)
            cls_4_logit = (1-a) * cls_4_logit + a * cls_11to4
            con_4_logit = (1-a) * con_4_logit + a * con_11to4

            loss_fct = CrossEntropyLoss()
            loss1 = loss_fct(cls_logit.view(-1, self.num_labels), hard_label.view(-1))
            loss2 = loss_fct(con_logit.view(-1, self.num_labels), hard_label.view(-1))
            loss3 = loss_fct(cls_4_logit.view(-1, 4), tag_4.view(-1))
            loss4 = loss_fct(con_4_logit.view(-1, 4), tag_4.view(-1))
            logit1 = 0.5 * (F.softmax(cls_logit, dim=-1) + F.softmax(con_logit, dim=-1))
            logit2 = 0.5 * (F.softmax(cls_4_logit, dim=-1) + F.softmax(con_4_logit, dim=-1))
            loss = loss1+loss2+loss3+loss4
            return loss, logit1, logit2

    def get_x1x2_bert_emb(self, last_hidden_state, x1_len, x2_len):
        'return： x1和x2的bert emb 并补齐到各自实际最大长度'
        x1_emb, x2_emb = [], []
        x1_max_len = max(x1_len)
        x2_max_len = max(x2_len)
        for i in range(len(x1_len)):
            x1_emb.append(
                torch.cat((last_hidden_state[i][0:x1_len[i]], torch.zeros(x1_max_len - x1_len[i], self.roberta_size).to(device)),
                          dim=0))
            x2_emb.append(torch.cat((last_hidden_state[i][x1_len[i]:x1_len[i] + x2_len[i]],
                                     torch.zeros(x2_max_len - x2_len[i], self.roberta_size).to(device)), dim=0))
        return torch.stack(x1_emb, dim=0), torch.stack(x2_emb, dim=0)

    def get_connective_emb(self, last_hidden_state, x1_len):
        connective_emb = []
        for i in range(len(x1_len)):
            connective_emb.append(last_hidden_state[i][x1_len[i]])
        return torch.stack(connective_emb, dim=0)

    def get_exp_con_emb(self, last_hidden_state, x1_len, con_num):
        connective_emb = []
        for i in range(len(x1_len)):
            temp_emb = last_hidden_state[i][x1_len[i]: x1_len[i] + con_num[i]].view(con_num[i], -1)
            connective_emb.append(torch.mean(temp_emb, dim=0).view(-1))
        return torch.stack(connective_emb, dim=0)
