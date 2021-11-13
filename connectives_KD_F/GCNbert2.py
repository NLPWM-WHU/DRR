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
from transformers import *
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


class GCNbert(nn.Module):
    def __init__(self, class_num):
        super(GCNbert, self).__init__()
        self.bert_size = 768
        self.bert_model = BertModel.from_pretrained(UNCASED)
        self.set_finetune("full")
        self.bert_model_e = BertModel.from_pretrained(UNCASED)
        self.set_finetune("none")

        self.num_labels = class_num
        if class_num == 4 or class_num == 2:
            self.con_labels = 93  # ji 4分类
        else:
            # self.con_labels = 94     # lin 11分类
            self.con_labels = 92  # ji 11分类
        self.max_len = 200
        self.in_dim = self.bert_size  # + pos_emb.shape[1] + ner_emb.shape[1] #+ 20
        self.out_dim = self.bert_size
        self.hidden_dim = 128
        self.num_perspectives = 16
        self.out_pos_dim = self.out_dim  # + self.pos_dim
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.dropout1 = nn.Dropout(0.2)

        self.fc1 = nn.Linear(self.out_dim * 2, self.out_dim, bias=True)
        self.fc_layer = FullyConnectedLayer(self.out_dim * 2, self.out_dim, self.num_labels)
        self.classifier = nn.Linear(self.out_dim, self.num_labels)
        self.cls_classifier = nn.Linear(self.out_dim, self.num_labels)
        self.cls_classifier11 = nn.Linear(self.out_dim, 11)
        self.classifier11 = nn.Linear(self.out_dim, 11)
        self.cls_classifier_e = nn.Linear(self.out_dim, self.num_labels)
        self.con_classifier_e = nn.Linear(self.out_dim, self.num_labels)
        for i in self.cls_classifier_e.parameters():
            i.requires_grad = False
        for i in self.con_classifier_e.parameters():
            i.requires_grad = False

        for i in self.cls_classifier11.parameters():
            i.requires_grad = False
        for i in self.classifier11.parameters():
            i.requires_grad = False
        # self.cls_classifier1 = nn.Linear(self.out_dim, self.num_labels)
        # self.exp_con_classifier = nn.Linear(self.out_dim, self.num_labels)
        # self.W_linear = nn.Linear(self.out_dim, self.out_dim, bias=True)

    def set_finetune(self, finetune):
        if finetune == "none":
            for param in self.bert_model_e.parameters():
                param.requires_grad = False
        elif finetune == "full":
            for param in self.bert_model.parameters():
                param.requires_grad = True
            # for param in self.bert_model_exp.parameters():
            #     param.requires_grad = True
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

    def forward(self, x1_info, x2_info, x_tag, x_idx, data_name, epoch, testing=False, y2=None, y3=None, y4=None,
                connect=None, exp_posi=None, alpha=0.0001):
        (x1_len, id1_map, id_len_1) = x1_info  # x1_len,x1_idx::numpyType
        (x2_len, id2_map, id_len_2) = x2_info

        # with torch.no_grad():
        (x_id, x_seg, x_mask, position_id) = x_idx
        # with torch.no_grad():
        outputs = self.bert_model(x_id, attention_mask=x_mask, token_type_ids=x_seg)
        # outputs = self.bert_model(x_id, attention_mask=x_mask)
        pooled_output = outputs[1]
        last_hidden_state = outputs[0]  # b*l*d
        connect_emb = self.get_connective_emb(last_hidden_state, x1_len)  # MASK位
        cls_emb = pooled_output
        logit = self.cls_classifier(cls_emb)
        connect_logit = self.classifier(connect_emb)

        if testing:
            logit_f = F.softmax(logit, dim=-1) + F.softmax(connect_logit, dim=-1)
            _, output_sense = torch.max(logit_f, 1)
            gold_sense = x_tag
            mask = (output_sense == y2)
            gold_sense[mask] = y2[mask]
            mask3 = (output_sense == y3)
            gold_sense[mask3] = y3[mask3]
            mask4 = (output_sense == y4)
            gold_sense[mask4] = y4[mask4]
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logit.view(-1, self.num_labels), gold_sense.view(-1))
            # logit = F.softmax(logit, dim=-1)
            return loss, logit_f  # logit
        else:
            '''
            # batch_posi = (X_exp_ids_p, exp_segment_p, exp_mask_p, batch_exp_Y_p, con_num_p, exp_X1_len_p)
            (X_exp_ids_p, exp_segment_p, exp_mask_p, batch_exp_Y_p, con_num_p, exp_X1_len_p) = exp_posi
            # (X_exp_ids_n, exp_segment_n, exp_mask_n, batch_exp_Y_n, con_num_n, exp_X1_len_n) = exp_neg
            outputs_exp_p = self.bert_model(X_exp_ids_p, attention_mask=exp_mask_p, token_type_ids=exp_segment_p)
            pooled_exp_p = outputs_exp_p[1]
            last_hidden_state_exp_p = outputs_exp_p[0]  # b*l*d
            exp_logit_p = self.cls_classifier1(pooled_exp_p)

            exp_con_emb_p = self.get_exp_con_emb(last_hidden_state_exp_p, exp_X1_len_p, con_num_p)
            exp_con_logit = self.exp_con_classifier(self.dropout1(exp_con_emb_p))

            exp_con_logit_p = self.classifier(exp_con_emb_p)
            imp_con_logit_p = connect_logit
            KL_loss = self.kl_categorical(imp_con_logit_p, exp_con_logit_p)
            '''
            outputs_e = self.bert_model_e(x_id, attention_mask=x_mask, token_type_ids=x_seg)
            pooled_output_e = outputs_e[1]
            last_hidden_state_e = outputs_e[0]  # b*l*d
            connect_emb_e = self.get_connective_emb(last_hidden_state_e, x1_len)  # MASK位

            cls_logit11 = self.cls_classifier11(pooled_output_e)
            con_logit11 = self.classifier11(connect_emb_e)

            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logit.view(-1, self.num_labels), x_tag.view(-1))
            loss_con = loss_fct(connect_logit.view(-1, self.num_labels), x_tag.view(-1))
            # loss_exp_con_p = loss_fct(exp_con_logit.view(-1, self.num_labels), batch_exp_Y_p.view(-1))
            # loss_exp_p = loss_fct(exp_logit_p.view(-1, self.num_labels), batch_exp_Y_p.view(-1))
            loss_imp = loss + loss_con
            # soft-label
            T = 2.0
            KL_loss_cls = self.kl_categorical(cls_logit11 / T, logit)
            KL_loss_con = self.kl_categorical(con_logit11 / T, connect_logit)
            KL_loss = (KL_loss_cls + KL_loss_con)
            # loss_exp = loss_exp_con_p + loss_exp_p
            # mask位对比学习
            cos_dis_con = torch.mean(1 - torch.cosine_similarity(connect_emb, connect_emb_e, dim=-1))
            cos_dis_cls = torch.mean(1 - torch.cosine_similarity(pooled_output, pooled_output_e, dim=-1))
            cos_dis = cos_dis_con + cos_dis_cls
            loss_ff = loss_imp + 1.0 * cos_dis + 1.0 * KL_loss
            # return loss_ff, loss_exp, KL_loss  # logit
            return loss_ff, loss_imp, KL_loss, cos_dis

    def kl_categorical(self, p_logit, q_logit):
        p = F.softmax(p_logit, dim=-1)
        _kl = torch.sum(p * (F.log_softmax(p_logit, dim=-1)
                             - F.log_softmax(q_logit, dim=-1)), 1)
        return torch.mean(_kl)

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
