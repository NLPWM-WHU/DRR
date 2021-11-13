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

    # pdtb_data = np.load(data_dir + "pdtb2.npz",allow_pickle=True)
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

    dev_num = int(len(pdtb_dev_Y))
    dev_arg11, dev_arg12, dev_con1, dev_arg21, dev_arg22, dev_con2, dev_arg31, dev_arg32, dev_con3 = [], [], [], [], [], [], [], [], []
    for i in range(dev_num):
        dev_arg11.append(pdtb_dev_arg1[i])
        dev_arg12.append(pdtb_dev_arg2[i])
        dev_con1.append(pdtb_dev_connect[i])
        pos_i = np.random.randint(0, dev_num)
        while pdtb_dev_Y[pos_i] != pdtb_dev_Y[i] or pos_i == i:
            pos_i = np.random.randint(0, dev_num)
        dev_arg21.append(pdtb_dev_arg1[pos_i])
        dev_arg22.append(pdtb_dev_arg2[pos_i])
        dev_con2.append(pdtb_dev_connect[pos_i])
        neg_i = np.random.randint(0, dev_num)
        while pdtb_dev_Y[neg_i] == pdtb_dev_Y[i]:
            neg_i = np.random.randint(0, dev_num)
        dev_arg31.append(pdtb_dev_arg1[neg_i])
        dev_arg32.append(pdtb_dev_arg2[neg_i])
        dev_con3.append(pdtb_dev_connect[neg_i])

    train_num = int(len(pdtb_train_Y))
    train_arg11, train_arg12, train_con1, train_arg21, train_arg22, train_con2, train_arg31, train_arg32, train_con3 = [], [], [], [], [], [], [], [], []
    for i in range(train_num):
        train_arg11.append(pdtb_train_arg1[i])
        train_arg12.append(pdtb_train_arg2[i])
        train_con1.append(pdtb_train_connect[i])
        pos_i = np.random.randint(0, train_num)
        while pdtb_train_Y[pos_i] != pdtb_train_Y[i] or pos_i == i:
            pos_i = np.random.randint(0, train_num)
        train_arg21.append(pdtb_train_arg1[pos_i])
        train_arg22.append(pdtb_train_arg2[pos_i])
        train_con2.append(pdtb_train_connect[pos_i])

        neg_i = np.random.randint(0, train_num)
        while pdtb_train_Y[neg_i] == pdtb_train_Y[i]:
            neg_i = np.random.randint(0, train_num)
        train_arg31.append(pdtb_train_arg1[neg_i])
        train_arg32.append(pdtb_train_arg2[neg_i])
        train_con3.append(pdtb_train_connect[neg_i])

    pdtb_train_data = (np.asarray(train_arg11), np.asarray(train_arg12), np.asarray(train_con1), np.asarray(train_arg21), np.asarray(train_arg22), np.asarray(train_con2), np.asarray(train_arg31), np.asarray(train_arg32), np.asarray(train_con3))
    pdtb_dev_data = (np.asarray(dev_arg11), np.asarray(dev_arg12), np.asarray(dev_con1), np.asarray(dev_arg21), np.asarray(dev_arg22), np.asarray(dev_con2), np.asarray(dev_arg31), np.asarray(dev_arg32), np.asarray(dev_con3))

    return pdtb_train_data, pdtb_dev_data

exp_data = get_data_bert("../interim/l/")
print(exp_data)


def get_data_bert_WSJ(data_dir,binary_cls=-1):

    pdtb_data = np.load(data_dir + "pdtb.npz",allow_pickle=True)

    pdtb_embedding=([],[],[])

    pdtb_dev_arg1_X=pdtb_data['dev_arg1_X']
    pdtb_dev_arg2_X=pdtb_data['dev_arg2_X']
    pdtb_dev_Y=pdtb_data['dev_Y']
    pdtb_dev_arg1_arg2_pos_ner_list=pdtb_data['dev_arg1_arg2_pos_ner_list']


    pdtb_train_arg1_X = pdtb_data['train_arg1_X']
    pdtb_train_arg2_X = pdtb_data['train_arg2_X']
    pdtb_train_Y = pdtb_data['train_Y']
    pdtb_train_pos_adjm = pdtb_data['train_pos_adjm']
    pdtb_train_arg1_arg2_pos_ner_list = pdtb_data['train_arg1_arg2_pos_ner_list']
    pdtb_train_arg1_adj_matrix = pdtb_data["train_arg1_adj_matrix"]
    pdtb_train_arg2_adj_matrix = pdtb_data["train_arg2_adj_matrix"]
    pdtb_train_arg1_text = pdtb_data['train_arg1_text']
    pdtb_train_arg2_text = pdtb_data['train_arg2_text']


    ###处理成2分类，binary_cls数字为n，就是第n个类和其他类one vs others ['Temporal','Comparison','Contingency','Expansion']
    if(binary_cls!=-1):
        pdtb_train_Y_binary=np.zeros((len(pdtb_train_Y),2))
        for i in range(len(pdtb_train_Y)):
            if(pdtb_train_Y[i][binary_cls]!=0):
                pdtb_train_Y_binary[i][0] = 1
                if(np.sum(pdtb_train_Y[i])>1):
                    pdtb_train_Y_binary[i][1] = 1
            else:
                pdtb_train_Y_binary[i][0] = 0
                pdtb_train_Y_binary[i][1] = 1
        pdtb_train_Y=pdtb_train_Y_binary

    train_arg1_pos, train_arg1_ner, train_arg2_pos, train_arg2_ner=pdtb_train_arg1_arg2_pos_ner_list[0],pdtb_train_arg1_arg2_pos_ner_list[1],pdtb_train_arg1_arg2_pos_ner_list[2],pdtb_train_arg1_arg2_pos_ner_list[3]
    pdtb_train_data =(pdtb_train_arg1_X,pdtb_train_arg2_X,pdtb_train_Y,pdtb_train_pos_adjm,train_arg1_pos, train_arg1_ner,\
                      pdtb_train_arg1_adj_matrix,train_arg2_pos, train_arg2_ner,pdtb_train_arg2_adj_matrix,pdtb_train_arg1_text,pdtb_train_arg2_text)

    pdtb_test_arg1_X = pdtb_data['test_arg1_X']
    pdtb_test_arg2_X = pdtb_data['test_arg2_X']
    pdtb_test_Y = pdtb_data['test_Y']
    pdtb_test_pos_adjm = pdtb_data['test_pos_adjm']
    pdtb_test_arg1_arg2_pos_ner_list = pdtb_data['test_arg1_arg2_pos_ner_list']
    pdtb_test_arg1_adj_matrix = pdtb_data["test_arg1_adj_matrix"]
    pdtb_test_arg2_adj_matrix = pdtb_data["test_arg2_adj_matrix"]
    pdtb_test_arg1_text = pdtb_data['test_arg1_text']
    pdtb_test_arg2_text = pdtb_data['test_arg2_text']
    test_arg1_pos, test_arg1_ner, test_arg2_pos, test_arg2_ner = pdtb_test_arg1_arg2_pos_ner_list[0], \
                                                                     pdtb_test_arg1_arg2_pos_ner_list[1], \
                                                                     pdtb_test_arg1_arg2_pos_ner_list[2], \
                                                                     pdtb_test_arg1_arg2_pos_ner_list[3]

    if(binary_cls!=-1):
        pdtb_test_Y_binary=np.zeros((len(pdtb_test_Y),2))
        for i in range(len(pdtb_test_Y)):
            if(pdtb_test_Y[i][binary_cls]!=0):
                pdtb_test_Y_binary[i][0] = 1
                if(np.sum(pdtb_test_Y[i])>1):
                    pdtb_test_Y_binary[i][1] = 1
            else:
                pdtb_test_Y_binary[i][0] = 0
                pdtb_test_Y_binary[i][1] = 1
        pdtb_test_Y=pdtb_test_Y_binary


    pdtb_test_data=(pdtb_test_arg1_X,pdtb_test_arg2_X,pdtb_test_Y,pdtb_test_pos_adjm,test_arg1_pos, test_arg1_ner,pdtb_test_arg1_adj_matrix,\
                    test_arg2_pos, test_arg2_ner,pdtb_test_arg2_adj_matrix, pdtb_test_arg1_text, pdtb_test_arg2_text)

    pdtb_all=(pdtb_embedding,pdtb_train_data,pdtb_test_data)


    wiki_all = ([],[],[])

    return pdtb_all,wiki_all
def get_data_bert_store(data_dir):
    pdtb_emb_file = np.load(data_dir + "pdtb.embedding.vec.npz",allow_pickle=True)
    pdtb_data = np.load(data_dir + "pdtb.npz",allow_pickle=True)

    pdtb_word_embedding=pdtb_emb_file['word_embedding']
    pdtb_pos_embedding=pdtb_emb_file['pos_embedding']
    pdtb_ner_embedding=pdtb_emb_file['ner_embedding']

    pdtb_embedding=(pdtb_word_embedding,pdtb_pos_embedding,pdtb_ner_embedding)

    pdtb_dev_arg1_X=pdtb_data['dev_arg1_X']
    pdtb_dev_arg2_X=pdtb_data['dev_arg2_X']
    pdtb_dev_Y=pdtb_data['dev_Y']
    pdtb_dev_arg1_arg2_pos_ner_list=pdtb_data['dev_arg1_arg2_pos_ner_list']


    pdtb_train_arg1_X = pdtb_data['train_arg1_X']
    pdtb_train_arg2_X = pdtb_data['train_arg2_X']
    pdtb_train_Y = pdtb_data['train_Y']
    pdtb_train_pos_adjm = pdtb_data['train_pos_adjm']
    pdtb_train_arg1_arg2_pos_ner_list = pdtb_data['train_arg1_arg2_pos_ner_list']
    pdtb_train_arg1_adj_matrix = pdtb_data["train_arg1_adj_matrix"]
    pdtb_train_arg2_adj_matrix = pdtb_data["train_arg2_adj_matrix"]
    pdtb_train_arg1_text = pdtb_data['train_arg1_text']
    pdtb_train_arg2_text = pdtb_data['train_arg2_text']

    train_arg1_pos, train_arg1_ner, train_arg2_pos, train_arg2_ner=pdtb_train_arg1_arg2_pos_ner_list[0],pdtb_train_arg1_arg2_pos_ner_list[1],pdtb_train_arg1_arg2_pos_ner_list[2],pdtb_train_arg1_arg2_pos_ner_list[3]
    pdtb_train_data =(pdtb_train_arg1_X,pdtb_train_arg2_X,pdtb_train_Y,pdtb_train_pos_adjm,train_arg1_pos, train_arg1_ner,\
                      pdtb_train_arg1_adj_matrix,train_arg2_pos, train_arg2_ner,pdtb_train_arg2_adj_matrix,pdtb_train_arg1_text,pdtb_train_arg2_text)

    pdtb_test_arg1_X = pdtb_data['test_arg1_X']
    pdtb_test_arg2_X = pdtb_data['test_arg2_X']
    pdtb_test_Y = pdtb_data['test_Y']
    pdtb_test_pos_adjm = pdtb_data['test_pos_adjm']
    pdtb_test_arg1_arg2_pos_ner_list = pdtb_data['test_arg1_arg2_pos_ner_list']
    pdtb_test_arg1_adj_matrix = pdtb_data["test_arg1_adj_matrix"]
    pdtb_test_arg2_adj_matrix = pdtb_data["test_arg2_adj_matrix"]
    pdtb_test_arg1_text = pdtb_data['test_arg1_text']
    pdtb_test_arg2_text = pdtb_data['test_arg2_text']
    test_arg1_pos, test_arg1_ner, test_arg2_pos, test_arg2_ner = pdtb_test_arg1_arg2_pos_ner_list[0], \
                                                                     pdtb_test_arg1_arg2_pos_ner_list[1], \
                                                                     pdtb_test_arg1_arg2_pos_ner_list[2], \
                                                                     pdtb_test_arg1_arg2_pos_ner_list[3]

    pdtb_test_data=(pdtb_test_arg1_X,pdtb_test_arg2_X,pdtb_test_Y,pdtb_test_pos_adjm,test_arg1_pos, test_arg1_ner,pdtb_test_arg1_adj_matrix,\
                    test_arg2_pos, test_arg2_ner,pdtb_test_arg2_adj_matrix, pdtb_test_arg1_text, pdtb_test_arg2_text)

    pdtb_all=(pdtb_embedding,pdtb_train_data,pdtb_test_data)

    #############wiki#####################
    wiki_emb_file = np.load(data_dir + "wiki.embedding.vec.npz")
    wiki_data = np.load(data_dir + "wiki.npz")

    wiki_word_embedding = wiki_emb_file['wiki_word_embedding']
    wiki_pdtb_pos_embedding = wiki_emb_file['wiki_pos_embedding']
    wiki_pdtb_ner_embedding = wiki_emb_file['wiki_ner_embedding']

    wiki_embedding = (wiki_word_embedding,wiki_pdtb_pos_embedding, wiki_pdtb_ner_embedding)

    wiki_arg1_X = wiki_data['wiki_arg1_X']
    wiki_arg2_X = wiki_data['wiki_arg2_X']
    wiki_Y = wiki_data['wiki_Y']
    wiki_arg1_arg2_pos_ner_list = wiki_data['wiki_arg1_arg2_pos_ner_list']
    #加了pos特征则删除
    wiki_arg1_arg2_pos_ner_list=np.zeros((40000,1))

    wiki_all=list(zip(wiki_arg1_X,wiki_arg2_X,wiki_Y,wiki_arg1_arg2_pos_ner_list))#zip 后划分测试集和打乱
    each_relation_num=10000
    train_rate=0.8
    wiki_train_data=wiki_all[0:8000]+wiki_all[10000:18000]+wiki_all[20000:28000]+wiki_all[30000:38000]
    wiki_test_data=wiki_all[8000:10000]+wiki_all[18000:20000]+wiki_all[28000:30000]+wiki_all[38000:40000]

    random.shuffle(wiki_train_data)
    random.shuffle(wiki_test_data)

    a,b,c,d=zip(*wiki_train_data)#wiki_arg1_X,wiki_arg2_X,wiki_Y,wiki_arg1_arg2_pos_ner_list
    wiki_train_data=(np.array(a),np.array(b),np.array(c),np.array(d))
    a, b, c, d =zip(*wiki_test_data)
    wiki_test_data=(np.array(a),np.array(b),np.array(c),np.array(d))

    wiki_all=(wiki_embedding,wiki_train_data,wiki_test_data)

    return pdtb_all,wiki_all