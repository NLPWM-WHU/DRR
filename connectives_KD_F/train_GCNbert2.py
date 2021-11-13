import argparse
import torch
torch.cuda.current_device()
import time
import json
import numpy as np
import math
import random
from time import strftime, localtime, time
import nltk
import GCNbert_myevaluation
#from lstm_att_network import  NNMA
import os
from utils import *
# from GCNbert import GCNbert
from GCNbert2 import GCNbert
import sys
import pickle
# torch.set_printoptions(threshold=10000000)
# UNCASED = '../BERT-BASE-UNCASED'
UNCASED = 'bert-base-uncased'
VOCAB = 'vocab.txt'

# os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cuda")

import torch
from transformers import *
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))
# Encode text
# input_ids = torch.tensor([tokenizer.encode("[CLS] Who was Jim ? [SEP] Jim was a puppeteer . [SEP]", add_special_tokens=True)])  # Add special tokens takes care of adding [CLS], [SEP], <s>... tokens in the right way for each model.
# with torch.no_grad():
#     last_hidden_states = model(input_ids)[0]  # Models outputs are now tuples
# MODELS = [(BertModel, BertTokenizer, 'bert-base-uncased'),
#           ]
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# tokenizerR = RobertaTokenizer.from_pretrained("roberta-base")
tokenizerR = BertTokenizer.from_pretrained(UNCASED)
pad_max = 100
pad_max_2 = 100


def evaluate2(model, X1, X2, Y,data_name,epoch,batch_size,test=False, y2=None,y3=None,y4=None, connect=None):
    model.eval()
    predict1=[]
    target = []
    target2 = []
    target3 = []
    target4 = []
    loss_all=0.0
    for batch in batch_generator(X1, X2, Y, batch_size, y2=y2,y3=y3, y4=y4, train_connect=connect):
        batch_X1, batch_X2, batch_y_info, batch_x = batch
        if test == False:
            (batch_y) = batch_y_info
            loss, logit1 = model(batch_X1, batch_X2, batch_y, batch_x, data_name, epoch, testing=test)
        else:
            (batch_y, batch_y2, batch_y3, batch_y4) = batch_y_info
            loss, logit1 = model(batch_X1, batch_X2, batch_y, batch_x, data_name, epoch, testing=test, y2=batch_y2, y3=batch_y3, y4=batch_y4)

        predict1.extend(logit1.data.cpu().numpy())
        target.extend(batch_y)
        target2.extend(batch_y2)
        target3.extend(batch_y3)
        target4.extend(batch_y4)
        # predict2.extend(logit2.data.cpu().numpy())
        # predict3.extend(logit3.data.cpu().numpy())
        loss_all+=loss.data

    model.train()
    # target = Y  # numpy
    predict1=np.array(predict1)
    # predict2 = np.array(predict2)
    # predict3=np.array(predict3)

    predict1, target1 = GCNbert_myevaluation.process_label(predict1, target, target2, target3, target4)
    # predict2, target2 = GCNbert_myevaluation.process_label(predict2, target)
    # predict3, target3 = GCNbert_myevaluation.process_label(predict3, target)

    predict = np.argmax(predict1, axis=1)
    target = np.argmax(target1, axis=1)
    # predict2 = np.argmax(predict2, axis=1)
    # target2 = np.argmax(target2, axis=1)
    # predict3 = np.argmax(predict3, axis=1)
    # target3 = np.argmax(target3, axis=1)

    acc, f1_1, each_f1 = GCNbert_myevaluation.print_evaluation_result((predict, target, loss_all),-1)
    # f1_2, CM2 = GCNbert_myevaluation.evaluation_result((predict2, target2, loss_all))
    # f1_3, CM3 = GCNbert_myevaluation.evaluation_result((predict3, target3, loss_all))
    # (X1_x, X1_pos,X1__adj_matrix, X1_ner,_)=X1
    # for binary_class
    # f1_1 = each_f1[0]
    return loss_all/len(X1),acc,f1_1,each_f1, predict1, target1#,f1_2,CM2 #,f1_3,CM3

def subword_tokenize(tokens):
    idx_map = []
    split_tokens = []
    for ix, token in enumerate(tokens):
        sub_tokens = tokenizerR.tokenize(token.lower())
        for jx, sub_token in enumerate(sub_tokens):
            split_tokens.append(sub_token)
            idx_map.append(ix)

    return idx_map, split_tokens


def train_pos_neg_instance(imp_Y, exp_data):
    """instance train loader for one training epoch"""
    (exp_train_arg1_text, exp_train_arg2_text, exp_train_Y, exp_train_connect) = exp_data
    posi_arg1, posi_arg2, posi_y, posi_con = [], [], [], []
    # neg_arg1, neg_arg2, neg_y, neg_con = [], [], [], []
    # num_negatives = 1
    item_num = len(exp_train_Y)
    for i, ylist in enumerate(imp_Y):
        # if i % 2 == 0:
        pos_i = np.random.randint(0, item_num)
        while exp_train_Y[pos_i] != ylist:
            pos_i = np.random.randint(0, item_num)
        posi_arg1.append(exp_train_arg1_text[pos_i])
        posi_arg2.append(exp_train_arg2_text[pos_i])
        posi_y.append(exp_train_Y[pos_i])
        posi_con.append(exp_train_connect[pos_i])
        # else:
        #     pos_i = np.random.randint(0, item_num)
        #     while exp_train_Y[pos_i] == ylist:
        #         pos_i = np.random.randint(0, item_num)
        #     posi_arg1.append(exp_train_arg1_text[pos_i])
        #     posi_arg2.append(exp_train_arg2_text[pos_i])
        #     posi_y.append(exp_train_Y[pos_i])
        #     posi_con.append(exp_train_connect[pos_i])
        # neg_i = np.random.randint(0, item_num)
        # while exp_train_Y[neg_i] == ylist:
        #     neg_i = np.random.randint(0, item_num)
        # neg_arg1.append(exp_train_arg1_text[neg_i])
        # neg_arg2.append(exp_train_arg2_text[neg_i])
        # neg_y.append(exp_train_Y[neg_i])
        # neg_con.append(exp_train_connect[neg_i])
    pos_data = (np.array(posi_arg1), np.array(posi_arg2), np.array(posi_y), np.array(posi_con))
    # neg_data = (neg_arg1, neg_arg2, neg_y, neg_con)
    return pos_data  #, neg_data


def batch_generator(X1_info,X2_info, y, batch_size, y2=None,y3=None,y4=None,train_connect=None, posi_data=None, neg_data=None):
    X1_text = X1_info
    X2_text = X2_info
    for offset in range(0, X1_text.shape[0], batch_size):
        batch_y = y[offset:offset + batch_size]
        # batch_connective=connective[offset:offset+batch_size]

        # batch_y_tmp=[0]*len(batch_y)
        # for i in range(len(batch_y)):
        #     for j in range(len(batch_y[i])):
        #         if(batch_y[i][j]!=0):
        #             batch_y_tmp[i]=j
        #             continue
        # batch_y=np.array(batch_y_tmp)

        # batch_pos_adjm=pos_adjm[offset:offset + batch_size]

        batch_X1_text = X1_text[offset:offset + batch_size]
        batch_X2_text = X2_text[offset:offset + batch_size]
        X1_input_ids,X2_input_ids = [], []
        batch_X1_len,batch_X2_len = [],[]
        idx1_map_list, idx2_map_list = [],[]
        id_len_1, id_len_2 = [], []
        for i in range(len(batch_X1_text)):
            word_list = nltk.word_tokenize(batch_X1_text[i])
            l = len(word_list)
            id_l = min(l, pad_max)
            id_len_1.append(id_l)
            word_list = word_list[0: id_l]
            idx_map, split_tokens = subword_tokenize(word_list)
            idx1_map_list.append(idx_map)
            token_l = len(idx_map) + 2
            batch_X1_len.append(token_l)
            tokens = ['[CLS]'] + split_tokens + ['[SEP]']
            tokens_id = tokenizerR.convert_tokens_to_ids(tokens)
            X1_input_ids.append(tokens_id)
        max_len = 0
        for i in range(len(batch_X2_text)):
            word_list2 = nltk.word_tokenize(batch_X2_text[i])
            l2 = len(word_list2)
            id_l2 = min(l2, pad_max)
            id_len_2.append(id_l2)
            word_list2 = word_list2[0: id_l2]
            idx_map2, split_tokens2 = subword_tokenize(word_list2)
            idx2_map_list.append(idx_map2)
            token_l2 = len(idx_map2) + 2
            batch_X2_len.append(token_l2)
            if max_len < batch_X1_len[i] + token_l2:
                max_len = batch_X1_len[i] + token_l2
            tokens2 = ['[MASK]'] + split_tokens2 + ['[SEP]']
            # tokens2 = ['<mask>'] + ['</s>'] + split_tokens2 + ['</s>']
            tokens_id2 = tokenizerR.convert_tokens_to_ids(tokens2)
            X2_input_ids.append(tokens_id2)
        arg1_maxl, arg2_maxl = max(batch_X1_len), max(batch_X2_len)

        segment = []
        mask = []
        X_input_ids = []
        position_id = []
        for i in range(len(batch_X1_len)):
            segment.append([0] * batch_X1_len[i] + [1] * (max_len - batch_X1_len[i]))
            mask_t = [1] * (batch_X1_len[i]) + [1] * (batch_X2_len[i]) + [0] * (
                        max_len - batch_X2_len[i] - batch_X1_len[i])
            X_input_ids.append(X1_input_ids[i] + X2_input_ids[i] + [0] * (max_len - batch_X1_len[i] - batch_X2_len[i]))
            mask.append(mask_t)

        # position_id = torch.stack(tuple(position_id), dim=0).long().to(device)
        X_input_ids = torch.tensor(X_input_ids).to(device)
        batch_segment = torch.tensor(segment).to(device)
        batch_mask = torch.tensor(mask).to(device)
        bert_inputs = (X_input_ids, batch_segment, batch_mask, position_id)


        batch_X1_info = (batch_X1_len, idx1_map_list, id_len_1)
        batch_X2_info = (batch_X2_len, idx2_map_list, id_len_2)

        batch_y = torch.autograd.Variable(torch.from_numpy(batch_y).long().to(device))
        # batch_connective = torch.autograd.Variable(torch.from_numpy(batch_connective).long().to(device))
        # batch_pos_adjm=torch.autograd.Variable(torch.from_numpy(batch_pos_adjm).float().to(device))
        if y2 is None:
            batch_y_info = (batch_y)
        elif y2 is not None:
            batch_y2 = y2[offset: offset + batch_size]
            batch_y2 = torch.autograd.Variable(torch.from_numpy(batch_y2).long().to(device))
            batch_y3 = y3[offset: offset + batch_size]
            batch_y3 = torch.autograd.Variable(torch.from_numpy(batch_y3).long().to(device))
            batch_y4 = y4[offset: offset + batch_size]
            batch_y4 = torch.autograd.Variable(torch.from_numpy(batch_y4).long().to(device))
            batch_y_info = (batch_y, batch_y2, batch_y3, batch_y4)
        else:
            batch_connect = train_connect[offset:offset+batch_size]
            batch_connect = torch.autograd.Variable(torch.from_numpy(batch_connect).long().to(device))
            batch_y_info = (batch_y, batch_connect)
        if len(batch_y) > 1:
            yield (batch_X1_info, batch_X2_info, batch_y_info, bert_inputs)

def __print(str1,data_name,op='a'):
    # with open("picture/%s_result.txt"%data_name, op) as f:
    #     f.write(str1 + '\n')
    print(str1)


def train(train_data,test_data,pdtb_exp_data,data_name,model, model_fn,
          optimizer, parameters, scheduler,batch_size,epochs=200, data_dir='', alpha=0.0001):
    # (train_arg1_text, train_arg2_text, train_arg1_pos, train_arg2_pos, train_arg1_mat,train_arg2_mat,train_ele_arg1,train_ele_arg2, train_Y, train_connect) = train_data
    # train_arg1_info = (train_arg1_text, train_arg1_pos, train_arg1_mat, train_ele_arg1)
    # train_arg2_info = (train_arg2_text, train_arg2_pos, train_arg2_mat, train_ele_arg2)
    # (test_arg1_text, test_arg2_text, test_arg1_pos, test_arg2_pos, test_arg1_mat, test_arg2_mat,test_ele_arg1,test_ele_arg2, test_Y, test_Y2, test_Y3, test_Y4) = test_data
    # test_arg1_info = (test_arg1_text, test_arg1_pos, test_arg1_mat, test_ele_arg1)
    # test_arg2_info = (test_arg2_text, test_arg2_pos, test_arg2_mat, test_ele_arg2)
    (train_arg1_text, train_arg2_text, train_Y) = train_data
    train_arg1_info = (train_arg1_text)
    train_arg2_info = (train_arg2_text)
    (test_arg1_text, test_arg2_text, test_Y, test_Y2, test_Y3, test_Y4) = test_data
    test_arg1_info = (test_arg1_text)
    test_arg2_info = (test_arg2_text)
    # (exp_train_arg1_text, exp_train_arg2_text, exp_train_Y, exp_train_connect) = pdtb_exp_data

    __print(str("dataset: %s\n"%data_name), data_name, 'w')
    __print(str("start:"+time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))+'\n',data_name,'w')
    best_loss = float("inf")
    best_acc = -1.0
    best_f1 = -1.0
    best_train_CM=""
    best_test_CM=''
    best_each_f1 = [-1.0,-1.0,-1.0,-1.0]
    test_loss_history=[]
    test_f1_history=[]
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_arg1_text))
    logger.info("  Batch size = %d", batch_size)
    logger.info("case study")

    step = 0
    for epoch in range(1, 1+epochs):
        shuffle_idx = np.random.permutation(len(train_arg1_text))
        # shuffle_idx2 = np.random.permutation(len(exp_train_arg1_text))
        (train_arg1_text) = train_arg1_info
        (train_arg2_text) = train_arg2_info
        # exp_train_data = (exp_train_arg1_text, exp_train_arg2_text, exp_train_Y, exp_train_connect)
        train_Y = train_Y[shuffle_idx]
        # train_connect = train_connect[shuffle_idx]
        train_arg1_text = train_arg1_text[shuffle_idx]
        train_arg2_text = train_arg2_text[shuffle_idx]
        # train_arg1_X = train_arg1_X[shuffle_idx]
        # train_arg2_X = train_arg2_X[shuffle_idx]
        train_arg1_info = (train_arg1_text)
        train_arg2_info = (train_arg2_text)
        # exp_train_data = (exp_train_arg1_text, exp_train_arg2_text, exp_train_Y, exp_train_connect)


        for batch in batch_generator(train_arg1_info,train_arg2_info,train_Y,batch_size, train_connect=None):
            batch_train_X1_info,batch_train_X2_info, batch_train_y,batch_X_idx = batch
            (batch_y) = batch_train_y
            # exp_data = (X_exp_ids, exp_segment, exp_mask, batch_exp_Y, con_num, exp_X1_len, con_mask)
            loss, imp_loss, KL_loss, cos_dis = model(batch_train_X1_info,batch_train_X2_info, batch_y,batch_X_idx,data_name, epoch, alpha=alpha)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(parameters, 2.0)
            optimizer.step()
            # test_loss, test_f1, test_CM1 = evaluate2(model, test_arg1_info, test_arg2_info, test_Y, data_name, epoch, batch_size, test=True, y2=test_Y2)
            # print(test_f1)
            # test_f1 = max(test_f11, test_f12, test_f13)
            step+=1
            if(step%50==0):
                logger.info('\n{:-^80}'.format('Iter' + str(epoch)))
                # logger.info(
                #     'Train: final loss={:.6f}, aspect loss={:.6f}, opinion loss={:.6f}, sentiment loss={:.6f}, reg loss={:.6f}, step={}'.
                #     format(train_loss, train_aspect_loss, train_opinion_loss, train_sentiment_loss, train_reg_loss,
                #            global_step))
                # print("Epoch %d train_loss:%0.4f, margin_loss:%0.4f" % (epoch, loss, mar_loss))
                logger.info('Train: final loss={:.6f}, imp loss={:.6f}, KL loss={:.6f}, cos_dis={:.6f}'.format(loss, imp_loss, KL_loss, cos_dis))
                # train_loss,acc, train_f1,train_CM = evaluate2(model, train_arg1_info,train_arg2_info,train_Y,data_name,epoch,batch_size, connect=train_connect)
                test_loss,test_acc, test_f1,test_each_f1,predict1,target1 = evaluate2(model, test_arg1_info,test_arg2_info,test_Y,data_name,epoch,batch_size,test=True, y2=test_Y2, y3=test_Y3, y4=test_Y4)
                # test_f1 = max(test_f11, test_f12, test_f13)
                # test_f1 = test_each_f1[0]
                # train_loss_history.append(train_loss), train_f1_history.append(train_f1)
                test_loss_history.append(test_loss), test_f1_history.append(test_f1)
                # print("Epoch %d loss: test:%0.4f" % (epoch,  test_loss))
                logger.info('Test loss={:.6f}'.format(test_loss))
                # print("Epoch %d f1: test:acc %0.2f, f1 %0.2f " % (epoch,  test_acc * 100, test_f1 * 100))
                logger.info('Test: acc={:.6f},f1={:.6f}'.format(test_acc * 100, test_f1 * 100))
                if test_f1 > best_f1:
                    best_f1 = test_f1
                    # best_train_CM=train_CM
                    best_each_f1 = test_each_f1
                    # params = {
                    #     # 'bert_model': model.bert_model.state_dict(),
                    #     # 'cls_classifier': model.cls_classifier.state_dict(),
                    #     'classifier': model.classifier.state_dict(),
                    #     # 'cls_classifier1': model.cls_classifier1.state_dict(),
                    #     # 'exp_con_classifier': model.exp_con_classifier.state_dict()
                    # }
                    # try:
                    #     torch.save(params, 'model/best_model_madeup_stable.pt')
                    #     print("model saved.")
                    # except BaseException:
                    #     print("[Warning: Saving failed... continuing anyway.]")
                    # np.savez(data_dir + "only_student_predict_dep11.npz", predict=predict1, target=target1)
                if test_acc > best_acc:
                    best_acc = test_acc
                    #GCNbert_myevaluation.test_sentence2file(model, test_arg1_info, test_arg2_info, test_Y, test_pos_adjm,data_name,epoch,batch_size,data_dir)
        if (epoch % 1 == 0):
            # print('Epoch %d bestacc: %0.2f bestf1: %0.2f ' % (epoch, best_acc * 100, best_f1 * 100))
            logger.info('best: acc={:.6f},f1={:.6f}'.format(best_acc * 100, best_f1 * 100))
            # print(best_each_f1)
            logger.info('best_each_f1:')
            logger.info(best_each_f1)

        # 权重衰减
        if (epoch % 1 == 0):
            scheduler.step()
    __print('best_test_CM: \n %s \nbest test_acc:\n %0.2f \nbestf1: %0.2f  ' % (best_test_CM, best_acc * 100,(best_f1 * 100)),data_name,'a')
    __print(str("end:" + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))) + '\n',data_name, 'a')
    return best_acc, best_f1


def run(domain, data_dir, model_dir, valid_split, runs, epochs, lr, dropout, class_num, batch_size=16, alpha=0.0001):

    pdtb_all = get_data_bert(data_dir)
    (pdtb_train_data, pdtb_test_data, pdtb_exp_data) = pdtb_all

    acc_mean = 0
    f1_mean = 0

    for r in range(runs):
        filename = 'model_exp/Pre-trained_2tasks_2datasets_11_bert_base.pt'
        # print(filename)
        logger.info(filename)
        # To use TensorFlow 2.0 versions of the models, simply prefix the class names with 'TF', e.g. `TFRobertaModel` is the TF 2.0 counterpart of the PyTorch model `RobertaModel`
        # pretrained_weights = 'bert-base-uncased'
        checkpoint = torch.load(filename)
        # Let's encode some text in a sequence of hidden-states using each model:
        model = GCNbert(class_num)
        # device = torch.device("cuda")
        # model.bert_model.load_state_dict(checkpoint['bert_model'], strict=False)
        model.bert_model_e.load_state_dict(checkpoint['bert'])
        model.cls_classifier11.load_state_dict(checkpoint['cls_class'])
        model.classifier11.load_state_dict(checkpoint['con_class'])
        # model.cuda()#模型中的模型会被放入CUDA，但是自己创建的tensor不会被放入，应该是没有继承nn.module的原因
        model.to(device)

        # param_optimizer = [(k, v) for k, v in model.named_parameters() if v.requires_grad == True]
        #
        # no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        # optimizer_grouped_parameters = [
        #     {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        #     {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        # ]
        # optimizer = AdamW(
        #     optimizer_grouped_parameters, lr=lr
        # )

        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(parameters, lr=lr)

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)#每过1轮，学习率乘以0.9
        acc, f1 = train(pdtb_train_data,pdtb_test_data,pdtb_exp_data,"pdtb",model, model_dir + domain + str(r), optimizer, parameters, scheduler,
                                             batch_size,epochs=epochs,data_dir=data_dir, alpha=alpha)
        acc_mean = acc_mean + acc
        f1_mean = f1_mean + f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default="../model/")
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--domain', type=str, default="restaurant")
    parser.add_argument('--data_dir', type=str, default="../interim/")
    parser.add_argument('--valid', type=int, default=150)  # number of validation data.
    parser.add_argument('--lr', type=float, default=0.00002)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--class_num', type=int, default=4)
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--alpha', type=float, default=0.0001)
    # parser.add_argument('--parse_tool', type=str, default="PDTB_LIN/")
    parser.add_argument('--parse_tool', type=str, default="l/")
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    if not os.path.exists('./log'):
        os.mkdir('./log')
    log_file = './log/Pre-trained_2tasks_2datasets_KL_loss_alpha_bert_base-{}-{}-{}-{}-{}-{}.log'.format(args.class_num, args.parse_tool.strip('/'),strftime("%y%m%d-%H%M%S", localtime()), args.seed, args.batch_size, args.alpha)
    logger.addHandler(logging.FileHandler(log_file))
    logger.info('> log file: {}'.format(log_file))

    run(args.domain, args.data_dir+args.parse_tool,args.model_dir, args.valid, args.runs, args.epochs, args.lr, args.dropout, args.class_num,
        args.batch_size, args.alpha)

