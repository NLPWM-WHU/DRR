import argparse
import torch
import time
import json
import numpy as np
import math
import random
import time
import pickle
# import myevaluation
import os

def get_data_bert(data_dir, binary_cls=-1):
    pdtb_data = np.load(data_dir + "pre_train_data.npz", allow_pickle=True)

    pdtb_train_Y = pdtb_data['arg_y']
    pdtb_train_arg1_text = pdtb_data['arg1_text']
    pdtb_train_arg2_text = pdtb_data['arg2_text']
    pdtb_train_connect = pdtb_data['arg_con']

    dev_num = int(len(pdtb_train_Y) * 0.95)
    pdtb_dev_Y = pdtb_train_Y[dev_num:]
    pdtb_dev_arg1 = pdtb_train_arg1_text[dev_num:]
    pdtb_dev_arg2 = pdtb_train_arg2_text[dev_num:]
    pdtb_dev_connect = pdtb_train_connect[dev_num:]

    pdtb_train_Y = pdtb_train_Y[0:dev_num]
    pdtb_train_arg1_text = pdtb_train_arg1_text[0:dev_num]
    pdtb_train_arg2_text = pdtb_train_arg2_text[0:dev_num]
    pdtb_train_connect = pdtb_train_connect[0:dev_num]

    '''
    ##处理成2分类，binary_cls数字为n，就是第n个类和其他类one vs others ['Temporal','Comparison','Contingency','Expansion']
    if(binary_cls!=-1):
        pdtb_train_Y_binary=np.zeros(len(pdtb_train_Y))
        for i in range(len(pdtb_train_Y)):
            if(pdtb_train_Y[i]==binary_cls):
                pdtb_train_Y_binary[i] = 0
                # if(np.sum(pdtb_train_Y[i])>1):
                #     pdtb_train_Y_binary[i][1] = 1
            else:
                pdtb_train_Y_binary[i] = 1
        pdtb_train_Y=pdtb_train_Y_binary.astype(int)
    '''
    pdtb_train_data = (pdtb_train_arg1_text, pdtb_train_arg2_text, pdtb_train_Y, pdtb_train_connect)
    pdtb_dev_data = (pdtb_dev_arg1, pdtb_dev_arg2, pdtb_dev_Y, pdtb_dev_connect)
    return pdtb_train_data, pdtb_dev_data


def get_data_pre(data_dir, binary_cls=-1):
    pdtb_data_imp = np.load(data_dir + "pdtb2.npz", allow_pickle=True)
    dev_arg1_text = pdtb_data_imp['arg1_dev']
    dev_arg2_text = pdtb_data_imp['arg2_dev']
    dev_Y = pdtb_data_imp['sense1_dev_id']
    train_arg1_text = pdtb_data_imp['arg1_train']
    train_arg2_text = pdtb_data_imp['arg2_train']
    train_Y = pdtb_data_imp['sense_train_id']
    train_connect = np.asarray(['NONE'] * train_Y.size)
    imp_flag = np.asarray([0] * train_Y.size)

    exp_data = np.load(data_dir + 'pdtb2expcon.npz')
    exp_train_Y = exp_data['sense_train_id']
    exp_train_arg1_text = exp_data['arg1_train']
    exp_train_arg2_text = exp_data['arg2_train']
    exp_train_connect = exp_data['conn_train_list']
    exp_train_connect = np.char.lower(exp_train_connect)
    exp_flag = np.asarray([1] * exp_train_Y.size)

    arg1_text = np.append(train_arg1_text, exp_train_arg1_text)
    arg2_text = np.append(train_arg2_text, exp_train_arg2_text)
    Y = np.append(train_Y, exp_train_Y)
    connect = np.append(train_connect, exp_train_connect)
    flag = np.append(imp_flag, exp_flag)

    shuffle_idx = np.random.permutation(len(arg1_text))
    arg1_text = arg1_text[shuffle_idx]
    arg2_text = arg2_text[shuffle_idx]
    Y = Y[shuffle_idx]
    connect = connect[shuffle_idx]
    flag = flag[shuffle_idx]

    Y_4 = Y.copy()
    Y_4[np.where(Y == 0)] = 0
    Y_4[np.where(Y == 1)] = 0
    Y_4[np.where(Y == 2)] = 1
    Y_4[np.where(Y == 3)] = 1
    Y_4[np.where(Y == 4)] = 2
    Y_4[np.where(Y == 5)] = 2
    Y_4[np.where(Y > 5)] = 3

    # conn2i = {}
    # conn_size = 0
    # for conn in exp_train_connect:
    #     if conn not in conn2i:
    #         conn2i[conn] = conn_size
    #         conn_size += 1
    # conn_train_id = []
    # for i in range(len(exp_train_connect)):
    #     conn_train_id.append(conn2i[exp_train_connect[i]])
    # conn_train_id = np.asarray(conn_train_id)
    # exp_train_Y_4 = exp_train_Y.copy()
    # exp_train_Y_4[np.where(exp_train_Y == 0)] = 0
    # exp_train_Y_4[np.where(exp_train_Y == 1)] = 0
    # exp_train_Y_4[np.where(exp_train_Y == 2)] = 1
    # exp_train_Y_4[np.where(exp_train_Y == 3)] = 1
    # exp_train_Y_4[np.where(exp_train_Y == 4)] = 2
    # exp_train_Y_4[np.where(exp_train_Y == 5)] = 2
    # exp_train_Y_4[np.where(exp_train_Y > 5)] = 3
    imp_dev_Y_4 = dev_Y.copy()
    imp_dev_Y_4[np.where(dev_Y == 0)] = 0
    imp_dev_Y_4[np.where(dev_Y == 1)] = 0
    imp_dev_Y_4[np.where(dev_Y == 2)] = 1
    imp_dev_Y_4[np.where(dev_Y == 3)] = 1
    imp_dev_Y_4[np.where(dev_Y == 4)] = 2
    imp_dev_Y_4[np.where(dev_Y == 5)] = 2
    imp_dev_Y_4[np.where(dev_Y > 5)] = 3
    train_data = (arg1_text, arg2_text, Y, connect, Y_4, flag)
    imp_dev_data = (dev_arg1_text, dev_arg2_text, dev_Y, imp_dev_Y_4)

    return train_data, imp_dev_data


def get_unlabel_data():
    # pdtb_data = np.load(data_dir + "pre_train_data.npz", allow_pickle=True)
    pdtb_data = np.load("../unlabel_data/DisSent_data/unlabel_expcon_FIVE.npz")
    pdtb_train_arg1_text = pdtb_data['arg1_train']
    pdtb_train_arg2_text = pdtb_data['arg2_train']
    pdtb_train_connect = pdtb_data['conn_train_list']
    train_arg1_text = []
    train_arg2_text = []
    train_connect = []
    train_Y = []
    for i in range(len(pdtb_train_connect)):
        train_arg1_text.append(pdtb_train_arg1_text[i])
        train_arg2_text.append(pdtb_train_arg2_text[i])
        train_connect.append(pdtb_train_connect[i])
        train_Y.append(1)

        neg_i = np.random.randint(0, len(pdtb_train_connect))
        while neg_i == i:
            neg_i = np.random.randint(0, len(pdtb_train_connect))
        train_arg1_text.append(pdtb_train_arg1_text[i])
        train_arg2_text.append(pdtb_train_arg2_text[neg_i])
        train_connect.append(pdtb_train_connect[neg_i])
        train_Y.append(0)

    pdtb_dev_arg1 = pdtb_data['arg1_dev'][0:5000]
    pdtb_dev_arg2 = pdtb_data['arg2_dev'][0:5000]
    pdtb_dev_connect = pdtb_data['conn_dev_list'][0:5000]

    dev_arg1_text = []
    dev_arg2_text = []
    dev_connect = []
    dev_Y = []
    for i in range(len(pdtb_dev_connect)):
        dev_arg1_text.append(pdtb_dev_arg1[i])
        dev_arg2_text.append(pdtb_dev_arg2[i])
        dev_connect.append(pdtb_dev_connect[i])
        dev_Y.append(1)

        neg_i = np.random.randint(0, len(pdtb_dev_connect))
        while neg_i == i:
            neg_i = np.random.randint(0, len(pdtb_dev_connect))
        dev_arg1_text.append(pdtb_dev_arg1[i])
        dev_arg2_text.append(pdtb_dev_arg2[neg_i])
        dev_connect.append(pdtb_dev_connect[neg_i])
        dev_Y.append(0)

    pdtb_train_data = (np.asarray(train_arg1_text), np.asarray(train_arg2_text), np.asarray(train_connect), np.asarray(train_Y))
    pdtb_dev_data = (np.asarray(dev_arg1_text), np.asarray(dev_arg2_text), np.asarray(dev_connect), np.asarray(dev_Y))
    return pdtb_train_data, pdtb_dev_data


def get_data_exp(data_dir):
    exp_data = np.load(data_dir + 'pdtb2expcon.npz')
    exp_train_Y = exp_data['sense_train_id']
    exp_train_arg1_text = exp_data['arg1_train']
    exp_train_arg2_text = exp_data['arg2_train']
    exp_train_connect = exp_data['conn_train_list']

    dev_len = int(len(exp_train_Y) * 0.95)
    pdtb_dev_Y = exp_train_Y[dev_len:]
    pdtb_dev_arg1 = exp_train_arg1_text[dev_len:]
    pdtb_dev_arg2 = exp_train_arg2_text[dev_len:]
    pdtb_dev_connect = exp_train_connect[dev_len:]
    pdtb_train_Y = exp_train_Y[0:dev_len]
    pdtb_train_arg1 = exp_train_arg1_text[0:dev_len]
    pdtb_train_arg2 = exp_train_arg2_text[0:dev_len]
    pdtb_train_connect = exp_train_connect[0:dev_len]

    pdtb_train_data = (pdtb_train_arg1, pdtb_train_arg2, pdtb_train_Y, pdtb_train_connect)
    pdtb_dev_data = (pdtb_dev_arg1, pdtb_dev_arg2, pdtb_dev_Y, pdtb_dev_connect)
    return pdtb_train_data, pdtb_dev_data
