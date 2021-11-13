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
def balance_data(pdtb_train_data):
    #train_arg1_arg2_pos_ner_list_balance 未做balance
    (train_arg1_X, train_arg2_X, train_Y, train_arg1_arg2_pos_ner_list) = pdtb_train_data
    train_arg1_X_balance, train_arg2_X_balance, train_Y_balance, train_arg1_arg2_pos_ner_list_balance=[],[],[],[]
    for i in range(len(train_Y)):
        if(train_Y[i][0]==1):
            train_arg1_X_balance.extend([train_arg1_X[i]]*10)
            train_arg2_X_balance.extend([train_arg2_X[i]]*10)
            train_Y_balance.extend([train_Y[i]] * 10)
            #train_arg1_arg2_pos_ner_list_balance.extend([train_arg1_arg2_pos_ner_list[i]] * 10)
        if(train_Y[i][1]==1):
            train_arg1_X_balance.extend([train_arg1_X[i]] * 3)
            train_arg2_X_balance.extend([train_arg2_X[i]] * 3)
            train_Y_balance.extend([train_Y[i]] * 3)
            #train_arg1_arg2_pos_ner_list_balance.extend([train_arg1_arg2_pos_ner_list[i]] * 3)
        if(train_Y[i][2]==1):
            train_arg1_X_balance.extend([train_arg1_X[i]] * 2)
            train_arg2_X_balance.extend([train_arg2_X[i]] * 2)
            train_Y_balance.extend([train_Y[i]] * 2)
            #train_arg1_arg2_pos_ner_list_balance.extend([train_arg1_arg2_pos_ner_list[i]] * 2)
        if(train_Y[i][3]==1):
            train_arg1_X_balance.extend([train_arg1_X[i]] )
            train_arg2_X_balance.extend([train_arg2_X[i]] )
            train_Y_balance.extend([train_Y[i]])
            #train_arg1_arg2_pos_ner_list_balance.extend([train_arg1_arg2_pos_ner_list[i]])

    train_arg1_X_balance, train_arg2_X_balance, train_Y_balance, train_arg1_arg2_pos_ner_list_balance=np.array(train_arg1_X_balance),np.array(train_arg2_X_balance),np.array(train_Y_balance),np.array(train_arg1_arg2_pos_ner_list_balance)

    shuffle_idx = np.random.permutation(len(train_arg1_X_balance))
    train_arg1_X_balance = train_arg1_X_balance[shuffle_idx]
    train_arg2_X_balance = train_arg2_X_balance[shuffle_idx]
    train_Y_balance = train_Y_balance[shuffle_idx]
    #train_arg1_arg2_pos_ner_list_balance= train_arg1_arg2_pos_ner_list_balance[shuffle_idx]
    pdtb_train_data_balance=(train_arg1_X_balance, train_arg2_X_balance, train_Y_balance, train_arg1_arg2_pos_ner_list_balance)

    return pdtb_train_data_balance

def get_data(data_dir):
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

    train_arg1_pos, train_arg1_ner, train_arg2_pos, train_arg2_ner=pdtb_train_arg1_arg2_pos_ner_list[0],pdtb_train_arg1_arg2_pos_ner_list[1],pdtb_train_arg1_arg2_pos_ner_list[2],pdtb_train_arg1_arg2_pos_ner_list[3]
    pdtb_train_data =(pdtb_train_arg1_X,pdtb_train_arg2_X,pdtb_train_Y,pdtb_train_pos_adjm,train_arg1_pos, train_arg1_ner,pdtb_train_arg1_adj_matrix,train_arg2_pos, train_arg2_ner,pdtb_train_arg2_adj_matrix)

    pdtb_test_arg1_X = pdtb_data['test_arg1_X']
    pdtb_test_arg2_X = pdtb_data['test_arg2_X']
    pdtb_test_Y = pdtb_data['test_Y']
    pdtb_test_pos_adjm = pdtb_data['test_pos_adjm']
    pdtb_test_arg1_arg2_pos_ner_list = pdtb_data['test_arg1_arg2_pos_ner_list']
    pdtb_test_arg1_adj_matrix = pdtb_data["test_arg1_adj_matrix"]
    pdtb_test_arg2_adj_matrix = pdtb_data["test_arg2_adj_matrix"]
    test_arg1_pos, test_arg1_ner, test_arg2_pos, test_arg2_ner = pdtb_test_arg1_arg2_pos_ner_list[0], \
                                                                     pdtb_test_arg1_arg2_pos_ner_list[1], \
                                                                     pdtb_test_arg1_arg2_pos_ner_list[2], \
                                                                     pdtb_test_arg1_arg2_pos_ner_list[3]

    pdtb_test_data=(pdtb_test_arg1_X,pdtb_test_arg2_X,pdtb_test_Y,pdtb_test_pos_adjm,test_arg1_pos, test_arg1_ner,pdtb_test_arg1_adj_matrix, test_arg2_pos, test_arg2_ner,pdtb_test_arg2_adj_matrix)

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
def DCT(a,k):
    batch_size=a.shape[0]
    N = a.shape[1]
    ck=torch.zeros((k,N))#k,4
    for i in range(k):
        ci = np.zeros((N,))
        for j in range(N):
            if(i==0):
                d=np.power(1.0/N, 0.5)
                ci[j]=d
            else:
                ci[j] = np.power(2.0 / N, 0.5)*np.cos(np.pi*(j+1/2)*i/N)
        ck[i]=torch.from_numpy(ci)
    ck=ck.cuda()
    ans=torch.matmul(ck,a)
    return  ans.view(batch_size,-1)

def get_data_bert(data_dir, binary_cls=-1):

    # pdtb_data = np.load(data_dir + "pdtb.npz",allow_pickle=True)
    # emb_file = np.load(data_dir + "pdtb_embedding.npz")
    # index_data = np.load(data_dir + "pdtb_index.npz")
    pdtb_data = np.load(data_dir + 'pdtb2.npz')
    # pnm_data = np.load(data_dir + 'pdtb_pn.npz')
    exp_data = np.load(data_dir + 'pdtb2expcon.npz')
    # dep_data = np.load(data_dir + 'pdtb_dep.npz')
    # train_arg1_e = np.load(data_dir + 'pdtb_elements_train_arg1_drop2.npz')
    # train_arg2_e = np.load(data_dir + 'pdtb_elements_train_arg2_drop2.npz')
    # test_e = np.load(data_dir + 'pdtb_elements_test_drop2.npz')
    # pos_mat = np.load(data_dir + 'pos_mat.npz')

    # word_embedding = emb_file['word_embedding']
    # pos_embedding = emb_file['pos_embedding']
    # ner_embedding = emb_file['ner_embedding']
    # pos_embedding = []
    # ner_embedding = []

    pdtb_train_Y = pdtb_data['sense_train_id']
    pdtb_train_arg1_text = pdtb_data['arg1_train']
    pdtb_train_arg2_text = pdtb_data['arg2_train']
    exp_train_Y = exp_data['sense_train_id']
    exp_train_arg1_text = exp_data['arg1_train']
    exp_train_arg2_text = exp_data['arg2_train']
    exp_train_connect = exp_data['conn_train_list']

    # train_ele_arg1 = train_arg1_e['train_arg1_ele']
    # train_ele_arg2 = train_arg2_e['train_arg2_ele']
    # test_ele_arg1 = test_e['test_arg1_ele']
    # test_ele_arg2 = test_e['test_arg2_ele']
    # train_ele_arg1 = dep_data['train_arg1_dep']
    # train_ele_arg2 = dep_data['train_arg2_dep']
    # test_ele_arg1 = dep_data['test_arg1_dep']
    # test_ele_arg2 = dep_data['test_arg2_dep']

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
        exp_train_Y_binary = np.zeros(len(exp_train_Y))
        for i in range(len(exp_train_Y)):
            if (exp_train_Y[i] == binary_cls):
                exp_train_Y_binary[i] = 0
                # if(np.sum(pdtb_train_Y[i])>1):
                #     pdtb_train_Y_binary[i][1] = 1
            else:
                exp_train_Y_binary[i] = 1
        exp_train_Y = exp_train_Y_binary.astype(int)

    pdtb_exp_data = (exp_train_arg1_text, exp_train_arg2_text, exp_train_Y, exp_train_connect)
    # pdtb_exp_test = (exp_test_arg1, exp_test_arg2, exp_test_Y, exp_test_connect)
    # train_arg1_X = index_data['train_arg1_index']
    # train_arg2_X = index_data['train_arg2_index']
    # train_arg1_pos = pnm_data['train_arg1_pos']
    # train_arg2_pos = pnm_data['train_arg2_pos']
    # # train_arg1_ner = pnm_data['train_arg1_ner']
    # # train_arg2_ner = pnm_data['train_arg2_ner']
    # train_arg1_mat = pnm_data['train_arg1_mat']
    # train_arg2_mat = pnm_data['train_arg2_mat']
    # train_pos_mat = pos_mat['pos_mat_train']

    # train_arg1_pos, train_arg1_ner, train_arg2_pos, train_arg2_ner=pdtb_train_arg1_arg2_pos_ner_list[0],pdtb_train_arg1_arg2_pos_ner_list[1],pdtb_train_arg1_arg2_pos_ner_list[2],pdtb_train_arg1_arg2_pos_ner_list[3]
    # pdtb_train_data =(pdtb_train_arg1_X,pdtb_train_arg2_X,pdtb_train_Y,pdtb_train_pos_adjm,train_arg1_pos, train_arg1_ner,\
    #                   pdtb_train_arg1_adj_matrix,train_arg2_pos, train_arg2_ner,pdtb_train_arg2_adj_matrix,pdtb_train_arg1_text,pdtb_train_arg2_text)

    # pdtb_train_data = (pdtb_train_arg1_text,pdtb_train_arg2_text, pdtb_train_Y, pdtb_train_connect,
    #                    train_arg1_X,train_arg2_X,train_arg1_pos,train_arg2_pos,train_arg1_ner,train_arg2_ner,train_arg1_mat,train_arg2_mat,train_pos_mat)
    pdtb_train_data = (pdtb_train_arg1_text, pdtb_train_arg2_text, pdtb_train_Y)
    # pdtb_test_arg1_X = pdtb_data['test_arg1_X']
    # pdtb_test_arg2_X = pdtb_data['test_arg2_X']
    pdtb_test_Y = pdtb_data['sense1_test_id']
    pdtb_test_Y2 = pdtb_data['sense2_test_id']
    pdtb_test_Y3 = pdtb_data['sense3_test_id']
    pdtb_test_Y4 = pdtb_data['sense4_test_id']
    # pdtb_test_pos_adjm = pdtb_data['test_pos_adjm']
    # pdtb_test_arg1_arg2_pos_ner_list = pdtb_data['test_arg1_arg2_pos_ner_list']
    # pdtb_test_arg1_adj_matrix = pdtb_data["test_arg1_adj_matrix"]
    # pdtb_test_arg2_adj_matrix = pdtb_data["test_arg2_adj_matrix"]
    pdtb_test_arg1_text = pdtb_data['arg1_test']
    pdtb_test_arg2_text = pdtb_data['arg2_test']
    # test_arg1_X = index_data['test_arg1_index']
    # test_arg2_X = index_data['test_arg2_index']
    # test_arg1_pos = pnm_data['test_arg1_pos']
    # test_arg2_pos = pnm_data['test_arg2_pos']
    # # test_arg1_ner = pnm_data['test_arg1_ner']
    # # test_arg2_ner = pnm_data['test_arg2_ner']
    # test_arg1_mat = pnm_data['test_arg1_mat']
    # test_arg2_mat = pnm_data['test_arg2_mat']
    # test_pos_mat = pos_mat['pos_mat_test']
    # test_label, test_count = np.unique(pdtb_test_Y, return_counts=True)
    # test2_label, test2_count = np.unique(pdtb_test_Y2, return_counts=True)
    # test_arg1_pos, test_arg1_ner, test_arg2_pos, test_arg2_ner = pdtb_test_arg1_arg2_pos_ner_list[0], \
    #                                                                  pdtb_test_arg1_arg2_pos_ner_list[1], \
    #                                                                  pdtb_test_arg1_arg2_pos_ner_list[2], \
    #                                                                  pdtb_test_arg1_arg2_pos_ner_list[3]

    if(binary_cls!=-1):
        pdtb_test_Y_binary=np.zeros(len(pdtb_test_Y))
        for i in range(len(pdtb_test_Y)):
            if(pdtb_test_Y[i]==binary_cls):
                pdtb_test_Y_binary[i] = 0
            else:
                pdtb_test_Y_binary[i] = 1
        pdtb_test_Y=pdtb_test_Y_binary.astype(int)
        # test_label, test_count = np.unique(pdtb_test_Y, return_counts=True)
        # pdtb_test_Y_binary2 = np.zeros(len(pdtb_test_Y2))
        for i in range(len(pdtb_test_Y2)):
            if(pdtb_test_Y2[i]==binary_cls):
                pdtb_test_Y2[i] = 0
            elif(pdtb_test_Y2[i] != -1):
                pdtb_test_Y2[i] = 1
        pdtb_test_Y2 = pdtb_test_Y2.astype(int)
        for i in range(len(pdtb_test_Y3)):
            if(pdtb_test_Y3[i]==binary_cls):
                pdtb_test_Y3[i] = 0
            elif(pdtb_test_Y3[i] != -1):
                pdtb_test_Y3[i] = 1
        pdtb_test_Y3 = pdtb_test_Y3.astype(int)
        for i in range(len(pdtb_test_Y4)):
            if(pdtb_test_Y4[i]==binary_cls):
                pdtb_test_Y4[i] = 0
            elif(pdtb_test_Y4[i] != -1):
                pdtb_test_Y4[i] = 1
        pdtb_test_Y4 = pdtb_test_Y4.astype(int)
        # test2_label, test2_count = np.unique(pdtb_test_Y2, return_counts=True)

    # pdtb_test_data=(pdtb_test_arg1_X,pdtb_test_arg2_X,pdtb_test_Y,pdtb_test_pos_adjm,test_arg1_pos, test_arg1_ner,pdtb_test_arg1_adj_matrix,\
    #                 test_arg2_pos, test_arg2_ner,pdtb_test_arg2_adj_matrix, pdtb_test_arg1_text, pdtb_test_arg2_text)

    # pdtb_test_data = (pdtb_test_arg1_text, pdtb_test_arg2_text, pdtb_test_Y, pdtb_test_Y2,
    #                   test_arg1_X,test_arg2_X,test_arg1_pos,test_arg2_pos,test_arg1_ner,test_arg2_ner,test_arg1_mat,test_arg2_mat,test_pos_mat)
    pdtb_test_data = (pdtb_test_arg1_text, pdtb_test_arg2_text, pdtb_test_Y, pdtb_test_Y2, pdtb_test_Y3, pdtb_test_Y4)
    # embedding_data = (pos_embedding, ner_embedding)
    pdtb_all = (pdtb_train_data, pdtb_test_data, pdtb_exp_data)

    return pdtb_all


def get_data_exp(data_dir, binary_cls=-1):

    exp_data = np.load(data_dir + 'pdtb2expcon.npz')
    exp_train_Y = exp_data['sense_train_id']
    exp_train_arg1_text = exp_data['arg1_train']
    exp_train_arg2_text = exp_data['arg2_train']
    exp_train_connect = exp_data['conn_train_list']
    exp_test_Y = exp_data['sense1_test_id']
    exp_test_arg1 = exp_data['arg1_test']
    exp_test_arg2 = exp_data['arg2_test']
    exp_test_connect = exp_data['conn_test_list']

    pdtb_exp_data = (exp_train_arg1_text, exp_train_arg2_text, exp_train_Y, exp_train_connect)
    pdtb_exp_test = (exp_test_arg1, exp_test_arg2, exp_test_Y, exp_test_connect)

    pdtb_all = (pdtb_exp_data, pdtb_exp_test)

    return pdtb_all

# exp_data = get_data_bert("../interim/l/")
# print(exp_data)

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