import torch
from transformers import *
import nltk
import numpy as np
from time import strftime, localtime, time
from torch.nn import CrossEntropyLoss
import os
import sys
_INF = -1e12
data_dir = "../interim/l/"
pdtb_data = np.load(data_dir + 'pdtb2.npz')
exp_data = np.load(data_dir + 'pdtb2expcon.npz')
pdtb_train_Y = pdtb_data['sense_train_id']
exp_train_Y = exp_data['sense_train_id']
pdtb_train_arg1_text = exp_data['arg1_train']
pdtb_train_arg2_text = exp_data['arg2_train']
exp_con = exp_data['conn_train_list']

con_dict = ["accordingly", "as a result", "because", "by comparison", "by contrast", "consequently", "for example",
            "for instance", "furthermore", "in fact", "in other words", "in particular", "in short", "indeed",
            "previously", "rather", "so", "specifically", "therefore",
            "also", "although", "and", "as", "but", "however", "in addition", "instead", "meanwhile", "moreover", "rather", "since", "then", "thus", "while",
            "further", "in sum", "in the end", "overall", "similarly", "whereas"]

con_omit_dict = ["as long as", "if", "nor", "now that", "once", "otherwise", "unless", "until"]

exp_free_Y = []
exp_free_arg1 = []
exp_free_arg2 = []
exp_free_con = []
for i in range(len(exp_con)):
    if exp_con[i] not in con_omit_dict:
        exp_free_arg1.append(pdtb_train_arg1_text[i])
        exp_free_arg2.append(pdtb_train_arg2_text[i])
        exp_free_Y.append(exp_train_Y[i])
        exp_free_con.append(exp_con[i])

exp_free_Y = np.array(exp_free_Y)

exp_free_con = np.array(exp_free_con)
exp_free_arg1 = np.array(exp_free_arg1)
exp_free_arg2 = np.array(exp_free_arg2)
pre = '../interim/l/pdtb2exp_free.npz'
np.savez(pre, sense_train_id=exp_free_Y, arg1_train=exp_free_arg1, arg2_train=exp_free_arg2, conn_train_list=exp_free_con)
id_c, count = np.unique(exp_free_Y, return_counts=True)
print(count)
# id_class, count_imp = np.unique(pdtb_train_Y, return_counts=True)
# id_class2, count_exp = np.unique(exp_train_Y, return_counts=True)
# print(id_class)

# import logging
# logger = logging.getLogger()
# logger.setLevel(logging.INFO)
# logger.addHandler(logging.StreamHandler(sys.stdout))
# alpha = 0.1
# logger.info('Train: alpha: {:.2f}'.format(alpha))
# if not os.path.exists('./log'):
#     os.mkdir('./log')
# log_file = './log/{}-{}.log'.format(strftime("%y%m%d-%H%M%S", localtime()), alpha)
# logger.addHandler(logging.FileHandler(log_file))
# logger.info('> log file: {}'.format(log_file))
# loss_fct = CrossEntropyLoss()
# logit = torch.Tensor([[0.1, 0.9], [0.2, 0.8], [0.3, 0.7]])
# x_tag = torch.Tensor([1, 0, 0]).long()
# logit1 = torch.Tensor([[0.1, 0.9]])
# x_tag1 = torch.Tensor([1]).long()
# logit2 = torch.Tensor([[0.2, 0.8]])
# x_tag2 = torch.Tensor([0]).long()
# logit3 = torch.Tensor([[0.3, 0.7]])
# x_tag3 = torch.Tensor([0]).long()
# loss = loss_fct(logit.view(-1, 2), x_tag.view(-1))
# loss1 = loss_fct(logit1.view(-1, 2), x_tag1.view(-1))
# loss2 = loss_fct(logit2.view(-1, 2), x_tag2.view(-1))
# loss3 = loss_fct(logit3.view(-1, 2), x_tag3.view(-1))
# print(loss)
# Encode text
# input_ids = torch.tensor([tokenizer.encode("[CLS] Who was Jim ? [SEP] Jim was a puppeteer . [SEP]", add_special_tokens=True)])  # Add special tokens takes care of adding [CLS], [SEP], <s>... tokens in the right way for each model.
# with torch.no_grad():
#     last_hidden_states = model(input_ids)[0]  # Models outputs are now tuples
# MODELS = [(BertModel, BertTokenizer, 'bert-base-uncased'),
#           ]
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# tokenizerR = RobertaTokenizer.from_pretrained("roberta-base")
# one_hot = torch.zeros_like(logit).scatter_(1, x_tag.view(-1, 1), 1) * 2 - 1

# def subword_tokenize(tokens):
#     idx_map = []
#     split_tokens = []
#     for ix, token in enumerate(tokens):
#         sub_tokens = tokenizerR.tokenize(token.lower())
#         for jx, sub_token in enumerate(sub_tokens):
#             split_tokens.append(sub_token)
#             idx_map.append(ix)
#
#     return idx_map, split_tokens
#
# ss1 = 'He does this in at least three of his solo pieces.'
# ss2 = 'You might call it, a leitmotif or a virtuoso accomplishment'
# nltk_list1 = nltk.word_tokenize(ss1)
# nltk_list2 = nltk.word_tokenize(ss2)
# print(nltk.word_tokenize(ss1))
# # print(tokenizer.tokenize(sss))
# id1, tokens_1 = subword_tokenize(nltk_list1)
# # print(tokenizerR.tokenize(nltk_list1))
# print(nltk.word_tokenize(ss2))
# id2, tokens_2 = subword_tokenize(nltk_list2)
# print(tokenizerR.tokenize(nltk_list2))
# for i in range(len(nltk_list1)):
#     words = tokenizerR.tokenize(nltk_list1[i].lower())
#     id = tokenizerR.convert_tokens_to_ids(words)
#     c = tokenizerR.convert_tokens_to_ids('<s>')
#     s = tokenizerR.convert_tokens_to_ids('</s>')
#     e = tokenizerR.convert_tokens_to_ids('<mask>')
#     m = tokenizerR.convert_tokens_to_ids('<pad>')
#     u = tokenizerR.convert_tokens_to_ids('<unk>')
#     cc = tokenizerR.convert_tokens_to_ids('<cls>')
#     print(words)
# data_dir = "../interim/l/"
# pdtb_data = np.load(data_dir + "pdtb2exp.npz", allow_pickle=True)
# pnm_data = np.load(data_dir + "pdtb_pn.npz", allow_pickle=True)
# result_data = np.load("predict_dep.npz", allow_pickle=True)

# pdtb_train_arg1_text = pdtb_data['arg1_train']
# pdtb_train_arg2_text = pdtb_data['arg2_train']
# pdtb_test_arg1_text = pdtb_data['arg1_test']
# pdtb_test_arg2_text = pdtb_data['arg2_test']
# train_connect = pdtb_data['conn_train_list']
# test_connect = pdtb_data['conn_test_list']
#
# con_id, count1 = np.unique(train_connect, return_counts=True)
# con_id2, count2 = np.unique(test_connect, return_counts=True)
# print(con_id)
# print(con_id2)
# test_arg1_pos = pnm_data['test_arg1_pos']
# test_arg2_pos = pnm_data['test_arg2_pos']
# predict = result_data['predict']
# tag = result_data['target']

# for i in range(len(tag)):
#     print(pdtb_test_arg1_text[i])
#     print(test_arg1_pos[i])
#     print(pdtb_test_arg2_text[i])
#     print(test_arg2_pos[i])
#     print("predict: %d" % (predict[i]))
#     print("tag: %d" % tag[i])

# train_arg1_pos = pnm_data['train_arg1_pos']
# train_arg2_pos = pnm_data['train_arg2_pos']
# test_arg1_pos = pnm_data['test_arg1_pos']
# test_arg2_pos = pnm_data['test_arg2_pos']

# count1, train_arg1 = np.unique(train_arg1_pos, return_counts=True)
# count2, train_arg2 = np.unique(train_arg2_pos, return_counts=True)
# count3, test_arg1 = np.unique(test_arg1_pos, return_counts=True)
# count4, test_arg2 = np.unique(test_arg2_pos, return_counts=True)
'''
# train_arg1_pos[train_arg1_pos == 16] = 3
# train_arg1_pos[train_arg1_pos == 33] = 3
train_arg1_pos[train_arg1_pos == 8] = 7
train_arg1_pos[train_arg1_pos == 9] = 7
train_arg1_pos[train_arg1_pos == 13] = 12
train_arg1_pos[train_arg1_pos == 14] = 12
train_arg1_pos[train_arg1_pos == 15] = 12
# train_arg1_pos[train_arg1_pos == 19] = 18
train_arg1_pos[train_arg1_pos == 21] = 20
train_arg1_pos[train_arg1_pos == 22] = 20
train_arg1_pos[train_arg1_pos == 28] = 27
train_arg1_pos[train_arg1_pos == 29] = 27
train_arg1_pos[train_arg1_pos == 30] = 27
train_arg1_pos[train_arg1_pos == 31] = 27
train_arg1_pos[train_arg1_pos == 32] = 27
train_arg1_pos[train_arg1_pos == 35] = 34
# train_arg1_pos[train_arg1_pos == 36] = 34

# train_arg2_pos[train_arg2_pos == 16] = 3
# train_arg2_pos[train_arg2_pos == 33] = 3
train_arg2_pos[train_arg2_pos == 8] = 7
train_arg2_pos[train_arg2_pos == 9] = 7
train_arg2_pos[train_arg2_pos == 13] = 12
train_arg2_pos[train_arg2_pos == 14] = 12
train_arg2_pos[train_arg2_pos == 15] = 12
# train_arg2_pos[train_arg2_pos == 19] = 18
train_arg2_pos[train_arg2_pos == 21] = 20
train_arg2_pos[train_arg2_pos == 22] = 20
train_arg2_pos[train_arg2_pos == 28] = 27
train_arg2_pos[train_arg2_pos == 29] = 27
train_arg2_pos[train_arg2_pos == 30] = 27
train_arg2_pos[train_arg2_pos == 31] = 27
train_arg2_pos[train_arg2_pos == 32] = 27
train_arg2_pos[train_arg2_pos == 35] = 34
# train_arg2_pos[train_arg2_pos == 36] = 34

# test_arg1_pos[test_arg1_pos == 16] = 3
# test_arg1_pos[test_arg1_pos == 33] = 3
test_arg1_pos[test_arg1_pos == 8] = 7
test_arg1_pos[test_arg1_pos == 9] = 7
test_arg1_pos[test_arg1_pos == 13] = 12
test_arg1_pos[test_arg1_pos == 14] = 12
test_arg1_pos[test_arg1_pos == 15] = 12
# test_arg1_pos[test_arg1_pos == 19] = 18
test_arg1_pos[test_arg1_pos == 21] = 20
test_arg1_pos[test_arg1_pos == 22] = 20
test_arg1_pos[test_arg1_pos == 28] = 27
test_arg1_pos[test_arg1_pos == 29] = 27
test_arg1_pos[test_arg1_pos == 30] = 27
test_arg1_pos[test_arg1_pos == 31] = 27
test_arg1_pos[test_arg1_pos == 32] = 27
test_arg1_pos[test_arg1_pos == 35] = 34
# test_arg1_pos[test_arg1_pos == 36] = 34

# test_arg2_pos[test_arg2_pos == 16] = 3
# test_arg2_pos[test_arg2_pos == 33] = 3
test_arg2_pos[test_arg2_pos == 8] = 7
test_arg2_pos[test_arg2_pos == 9] = 7
test_arg2_pos[test_arg2_pos == 13] = 12
test_arg2_pos[test_arg2_pos == 14] = 12
test_arg2_pos[test_arg2_pos == 15] = 12
# test_arg2_pos[test_arg2_pos == 19] = 18
test_arg2_pos[test_arg2_pos == 21] = 20
test_arg2_pos[test_arg2_pos == 22] = 20
test_arg2_pos[test_arg2_pos == 28] = 27
test_arg2_pos[test_arg2_pos == 29] = 27
test_arg2_pos[test_arg2_pos == 30] = 27
test_arg2_pos[test_arg2_pos == 31] = 27
test_arg2_pos[test_arg2_pos == 32] = 27
test_arg2_pos[test_arg2_pos == 35] = 34
# test_arg2_pos[test_arg2_pos == 36] = 34
file6_path = data_dir + "pdtb_pn3.npz"
np.savez(file6_path, train_arg1_pos=train_arg1_pos, train_arg2_pos=train_arg2_pos, test_arg1_pos=test_arg1_pos, test_arg2_pos=test_arg2_pos)

'''