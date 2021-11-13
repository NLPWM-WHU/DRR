import argparse
import torch
import time
import json
import numpy as np
import math
import random
import time
import sys
import GCNbert_myevaluation

import os
from utils4pre import *
# from GCNbert import GCNbert
from RobertaForPretrain import bertForPretrain
import os
import nltk
import pickle
# UNCASED = '../BERT-BASE-UNCASED'
UNCASED = 'bert-base-uncased'
VOCAB = 'vocab.txt'


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda")

import torch
from transformers import *
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))
tokenizerR = BertTokenizer.from_pretrained(UNCASED)
# tokenizerR = RobertaTokenizer.from_pretrained('BERT-BASE-UNCASED')
pad_max = 100


def evaluate2(model, X1, X2, Y, data_name, epoch, batch_size, y_4=None, alpha=0.1):
    model.eval()
    predict1 = []
    target1 = []
    predict2 = []
    target2 = []
    loss_all = 0.0
    batch_l = 0.0
    for batch in batch_generator(X1, X2, Y, batch_size, y_4=y_4):
        batch_y, batch_x = batch
        (batch_y_11, batch_y_4) = batch_y
        loss, logit1, logit2 = model(batch_y, batch_x, data_name, epoch, test=True, alpha=alpha)
        loss_all += loss.data
        batch_l += 1.0
        predict1.extend(logit1.data.cpu().numpy())
        target1.extend(batch_y_11)
        predict2.extend(logit2.data.cpu().numpy())
        target2.extend(batch_y_4)
    model.train()
    predict1 = np.array(predict1)
    predict1, target1 = GCNbert_myevaluation.process_label(predict1, target1)
    predict1 = np.argmax(predict1, axis=1)
    target1 = np.argmax(target1, axis=1)
    acc_11, f1_11, each_f1_11 = GCNbert_myevaluation.print_evaluation_result((predict1, target1, loss_all),-1)

    predict2 = np.array(predict2)
    predict2, target2 = GCNbert_myevaluation.process_label(predict2, target2)
    predict2 = np.argmax(predict2, axis=1)
    target2 = np.argmax(target2, axis=1)
    acc_4, f1_4, each_f1_4 = GCNbert_myevaluation.print_evaluation_result((predict2, target2, loss_all), -1)
    return loss_all/batch_l, acc_11, f1_11, each_f1_11, acc_4, f1_4, each_f1_4

def subword_tokenize(tokens):
    idx_map = []
    split_tokens = []
    for ix, token in enumerate(tokens):
        sub_tokens = tokenizerR.tokenize(token.lower())
        for jx, sub_token in enumerate(sub_tokens):
            split_tokens.append(sub_token)
            idx_map.append(ix)

    return idx_map, split_tokens

def batch_generator(X1_info, X2_info, y, batch_size, y2=None, train_connect=None, y_4=None, flag=None):
    X1_text = X1_info
    X2_text = X2_info
    for offset in range(0, X1_text.shape[0], batch_size):
        batch_y = y[offset:offset + batch_size]

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
        batch_X2_len = []
        X2_input_ids = []
        if train_connect is not None:
            con_num = []
            con_label_id = []
            batch_connect = train_connect[offset:offset+batch_size]

            for i in range(len(batch_X2_text)):
                word_list2 = nltk.word_tokenize(batch_X2_text[i])
                con_list = nltk.word_tokenize(batch_connect[i])
                l2 = len(word_list2)
                id_l2 = min(l2, pad_max)
                word_list2 = word_list2[0: id_l2]
                idx_map2, split_tokens2 = subword_tokenize(word_list2)
                idx_con, split_tokens_con = subword_tokenize(con_list)
                # idx2_map_list.append(idx_map2)
                con_num.append(len(idx_con))
                token_l2 = len(idx_map2) + 1 + len(idx_con)
                batch_X2_len.append(token_l2)
                if max_len < batch_X1_len[i] + token_l2:
                    max_len = batch_X1_len[i] + token_l2
                tokens2 = ['[MASK]'] * len(idx_con) + split_tokens2 + ['[SEP]']
                tokens_id2 = tokenizerR.convert_tokens_to_ids(tokens2)
                con_label_temp = split_tokens_con
                con_label_id.append(tokenizerR.convert_tokens_to_ids(con_label_temp))
                X2_input_ids.append(tokens_id2)

            segment = []
            mask = []
            X_input_ids = []
            label_id = []
            for i in range(len(batch_X1_len)):
                segment.append([0] * batch_X1_len[i] + [1] * (max_len - batch_X1_len[i]))
                mask_t = [1] * (batch_X1_len[i]) + [1] * (batch_X2_len[i]) + [0] * (
                            max_len - batch_X2_len[i] - batch_X1_len[i])
                X_input_ids.append(X1_input_ids[i] + X2_input_ids[i] + [0] * (max_len - batch_X1_len[i] - batch_X2_len[i]))
                label_id.append(
                    [-100] * batch_X1_len[i] + con_label_id[i] + [-100] * (max_len - batch_X1_len[i] - con_num[i]))
                mask.append(mask_t)

            X_input_ids = torch.tensor(X_input_ids).to(device)
            batch_segment = torch.tensor(segment).to(device)
            batch_mask = torch.tensor(mask).to(device)
            label_id = torch.tensor(label_id).to(device)
            bert_inputs = (X_input_ids, batch_segment, batch_mask, batch_X1_len, label_id, con_num)
        else:
            for i in range(len(batch_X2_text)):
                word_list2 = nltk.word_tokenize(batch_X2_text[i])
                l2 = len(word_list2)
                id_l2 = min(l2, pad_max)
                word_list2 = word_list2[0: id_l2]
                idx_map2, split_tokens2 = subword_tokenize(word_list2)
                token_l2 = len(idx_map2) + 2
                batch_X2_len.append(token_l2)
                if max_len < batch_X1_len[i] + token_l2:
                    max_len = batch_X1_len[i] + token_l2
                tokens2 = ['[MASK]'] + split_tokens2 + ['[SEP]']
                tokens_id2 = tokenizerR.convert_tokens_to_ids(tokens2)
                X2_input_ids.append(tokens_id2)

            segment = []
            mask = []
            X_input_ids = []
            for i in range(len(batch_X1_len)):
                segment.append([0] * batch_X1_len[i] + [1] * (max_len - batch_X1_len[i]))
                mask_t = [1] * (batch_X1_len[i]) + [1] * (batch_X2_len[i]) + [0] * (
                        max_len - batch_X2_len[i] - batch_X1_len[i])
                X_input_ids.append(X1_input_ids[i] + X2_input_ids[i] + [0] * (max_len - batch_X1_len[i] - batch_X2_len[i]))
                mask.append(mask_t)

            X_input_ids = torch.tensor(X_input_ids).to(device)
            batch_segment = torch.tensor(segment).to(device)
            batch_mask = torch.tensor(mask).to(device)
            bert_inputs = (X_input_ids, batch_segment, batch_mask, batch_X1_len)
        batch_y = torch.autograd.Variable(torch.from_numpy(batch_y).long().to(device))


        if flag is not None and y_4 is not None:
            train_flag = flag[offset:offset + batch_size]
            train_flag = torch.autograd.Variable(torch.from_numpy(train_flag).long().to(device))
            batch_y_4 = y_4[offset:offset + batch_size]
            batch_y_4 = torch.autograd.Variable(torch.from_numpy(batch_y_4).long().to(device))
            y_info = (batch_y, train_flag, batch_y_4)
        elif y_4 is not None:
            batch_y_4 = y_4[offset:offset + batch_size]
            batch_y_4 = torch.autograd.Variable(torch.from_numpy(batch_y_4).long().to(device))
            y_info = (batch_y, batch_y_4)
        else:
            y_info = batch_y

        yield (y_info, bert_inputs)


def __print(str1,data_name,op='a'):
    # with open("picture/%s_result.txt"%data_name, op) as f:
    #     f.write(str1 + '\n')
    print(str1)


def train(train_data, dev_data, data_name, model, model_fn, optimizer, parameters, scheduler, batch_size, epochs=20, data_dir='', crf=False, alpha=0.1):
    (train_arg1_text, train_arg2_text, train_Y, train_connect, train_Y_4, flag) = train_data
    train_arg1_info = (train_arg1_text)
    train_arg2_info = (train_arg2_text)
    (dev_arg1_text, dev_arg2_text, dev_Y, dev_Y_4) = dev_data
    __print(str("dataset: %s\n"%data_name), data_name, 'w')
    __print(str("start:"+time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))+'\n',data_name,'w')
    best_loss = float("inf")
    logger.info("***** Running pre-training *****")
    logger.info("  Num examples = %d", len(train_arg1_text))
    logger.info("  Batch size = %d", batch_size)
    logger.info(" pre-train 2_tasks and 2_datasets BERT-base")

    best_acc_11 = -1.0
    best_f1_11 = -1.0
    best_each_f1_11 = [-1.0,-1.0,-1.0,-1.0]
    best_acc_4 = -1.0
    best_f1_4 = -1.0
    best_each_f1_4 = [-1.0,-1.0,-1.0,-1.0]

    step = 0
    for epoch in range(1, 1+epochs):
        shuffle_idx = np.random.permutation(len(train_arg1_text))
        (train_arg1_text) = train_arg1_info
        (train_arg2_text) = train_arg2_info
        train_Y = train_Y[shuffle_idx]
        train_Y_4 = train_Y_4[shuffle_idx]
        train_connect = train_connect[shuffle_idx]
        flag = flag[shuffle_idx]
        train_arg1_text = train_arg1_text[shuffle_idx]
        train_arg2_text = train_arg2_text[shuffle_idx]
        train_arg1_info = (train_arg1_text)
        train_arg2_info = (train_arg2_text)
        for batch in batch_generator(train_arg1_info,train_arg2_info,train_Y,batch_size, train_connect=train_connect, y_4=train_Y_4, flag=flag):
            batch_y, batch_X = batch
            loss, loss_1, loss_2, loss_3, loss_4, mlm_loss = model(batch_y, batch_X, data_name, epoch, alpha=alpha)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(parameters, 2.0)
            optimizer.step()
            step += 1
            if(step%50==0):
                logger.info('\n{:-^80}'.format('Iter' + str(epoch)))
                logger.info('Train: final loss={:.6f}, loss_1={:.6f}, loss_2={:.6f}, loss_3={:.6f}, loss_4={:.6f}, mlm_loss={:.6f}'.format(loss, loss_1, loss_2, loss_3, loss_4, mlm_loss))
                dev_loss, acc_11, f1_11, each_f1_11, acc_4, f1_4, each_f1_4 = evaluate2(model, dev_arg1_text, dev_arg2_text, dev_Y, data_name, epoch, batch_size, y_4=dev_Y_4, alpha=alpha)
                logger.info('Dev loss={:.6f}, acc_11={:.6f}, f1_11={:.6f}, acc_4={:.6f}, f1_4={:.6f}'.format(dev_loss, acc_11, f1_11, acc_4, f1_4))
                if acc_11 > best_acc_11:
                    best_acc_11 = acc_11
                    params = {
                        'bert': model.bert_model.bert.state_dict(),
                        'lm_head': model.bert_model.cls.state_dict(),
                        'cls_class': model.cls_classifier.state_dict(),
                        'con_class': model.classifier.state_dict(),
                        'cls_class_4': model.cls_classifier_4.state_dict(),
                        'con_class_4': model.classifier_4.state_dict(),
                        # 'trans_W': model.trans_W.state_dict()
                    }
                    try:
                        torch.save(params, 'model_exp/Pre-trained_2tasks_2datasets_11_bert_base.pt')
                        print("model 11 saved.")
                    except BaseException:
                        print("[Warning: Saving failed... continuing anyway.]")
                if best_f1_11 < f1_11:
                    best_f1_11 = f1_11
                    best_each_f1_11 = each_f1_11

                if acc_4 > best_acc_4:
                    best_acc_4 = acc_4
                    params = {
                        'bert': model.bert_model.bert.state_dict(),
                        'lm_head': model.bert_model.cls.state_dict(),
                        'cls_class': model.cls_classifier.state_dict(),
                        'con_class': model.classifier.state_dict(),
                        'cls_class_4': model.cls_classifier_4.state_dict(),
                        'con_class_4': model.classifier_4.state_dict(),
                        # 'trans_W': model.trans_W.state_dict()
                    }
                    try:
                        torch.save(params, 'model_exp/Pre-trained_2tasks_2datasets_4_bert_base.pt')
                        print("model 4 saved.")
                    except BaseException:
                        print("[Warning: Saving failed... continuing anyway.]")
                if best_f1_4 < f1_4:
                    best_f1_4 = f1_4
                    best_each_f1_4 = each_f1_4

                if best_loss > dev_loss:
                    best_loss = dev_loss

        if (epoch % 1 == 0):
            logger.info('best: loss={:.6f}'.format(best_loss))
            logger.info('best: acc_11={:.6f}, f1_11={:.6f}'.format(best_acc_11, best_f1_11))
            logger.info(best_each_f1_11)
            logger.info('best: acc_4={:.6f}, f1_4={:.6f}'.format(best_acc_4, best_f1_4))
            logger.info(best_each_f1_4)

        # 权重衰减
        if (epoch % 1 == 0):
            scheduler.step()
    return best_loss


def run(domain, data_dir, model_dir, valid_split, runs, epochs, lr, dropout,class_num,batch_size=128,alpha=0.1):

    train_data, imp_dev_data = get_data_pre(data_dir)

    for r in range(runs):
        # filename = 'model_exp/best_model.pt'
        # checkpoint = torch.load(filename)
        # To use TensorFlow 2.0 versions of the models, simply prefix the class names with 'TF', e.g. `TFRobertaModel` is the TF 2.0 counterpart of the PyTorch model `RobertaModel`
        # pretrained_weights = 'bert-base-uncased'
        # Let's encode some text in a sequence of hidden-states using each model:
        model = bertForPretrain(class_num)
        model.to(device)

        n_trainable_params, n_nontrainable_params = 0, 0
        for p in model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        logger.info(
            'n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(parameters, lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)#每过1轮，学习率乘以0.9
        start_time = time.time()
        # def train(train_data, data_name, model, model_fn, optimizer, parameters, batch_size, epochs=20, data_dir='', crf=False):
        loss = train(train_data, imp_dev_data, "pdtb", model, model_dir + domain + str(r), optimizer, parameters, scheduler, batch_size, epochs=epochs, data_dir=data_dir, crf=False, alpha=alpha)
        end_time = time.time()
        logger.info('run time={:.6f}'.format(end_time - start_time))
    # print('mean acc: %0.2f mean f1: %0.2f ' % (acc_f * 100, f1_f * 100))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default="../model/")
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--domain', type=str, default="restaurant")
    parser.add_argument('--data_dir', type=str, default="../interim/")
    parser.add_argument('--valid', type=int, default=150)  # number of validation data.
    parser.add_argument('--lr', type=float, default=0.00002)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--class_num', type=int, default=11)
    parser.add_argument('--parse_tool', type=str, default="ji/")
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--alpha', type=float, default=0.09)

    args = parser.parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    if not os.path.exists('./log'):
        os.mkdir('./log')
    log_file = './log/Pre-trained_2tasks_2datasets_BERT_base_teacher-{}-{}-{}-{}-{}.log'.format(args.class_num, args.batch_size, args.seed, args.lr, args.alpha)
    logger.addHandler(logging.FileHandler(log_file))
    logger.info('> log file: {}'.format(log_file))

    run(args.domain, args.data_dir+args.parse_tool,args.model_dir, args.valid, args.runs, args.epochs, args.lr, args.dropout,args.class_num,
        args.batch_size,args.alpha)
