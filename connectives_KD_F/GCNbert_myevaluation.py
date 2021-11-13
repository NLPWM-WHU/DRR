import argparse
import torch
import time
import json
import numpy as np
import math
import random
# import train_bert_fineturn
# import train_GCNbert
from sklearn import metrics
# import matplotlib.pyplot as plt


np.random.seed(1337)
random.seed(1337)
torch.manual_seed(1337)
torch.cuda.manual_seed(1337)

def process_label(predict_Y,target_Y,y2=None, y3=None, y4=None):
    assert len(predict_Y) == len(target_Y)
    list_1 = []
    list_2 = []

    for i in range(predict_Y.shape[0]):
        real_label = np.zeros(predict_Y.shape[1])
        real_label[target_Y[i]] = 1
        if y2 is not None:
            if y2[i] > -1:
                real_label[y2[i]] = 1
        if y3 is not None:
            if y3[i] > -1:
                real_label[y3[i]] = 1
        if y4 is not None:
            if y4[i] > -1:
                real_label[y4[i]] = 1
        predict_label = predict_Y[i,:]

        #handle non-label case
        if np.sum(real_label) <= 0:
            continue

        #handle multilabel case
        if np.sum(real_label) >= 2:
            # predict one of the correct label
            if real_label[np.argmax(predict_label)] > 0:
                real_label = np.zeros(predict_Y.shape[1])
                real_label[np.argmax(predict_label)] = 1

        list_1.append(real_label)
        list_2.append(predict_label)

    if len(list_1) > 0:
        real_Y = np.stack(list_1)
        predict_Y = np.stack(list_2)

        return predict_Y,real_Y
    else:
        return None,None


def my_plot(train_loss_history, train_f1_history, test_loss_history, test_f1_history,dataname):
    x = range(len(train_loss_history))

    ##loss
    fig = plt.figure()

    maxloss=max(max(train_loss_history),max(test_loss_history))
    ax1 = fig.add_subplot(111)
    ax1.plot(x, train_loss_history,label = "train_loss")
    ax1.set_ylim([0.0, maxloss.cpu()])
    ax1.set_ylabel('train_loss_history')
    ax1.set_title("LOSS")

    ax2 = ax1.twinx()  # this is the important function
    ax2.plot(x, test_loss_history, 'r',label = "test_loss")
    ax2.set_ylim([0.0, maxloss.cpu()])
    ax2.set_ylabel('test_loss_history')
    ax2.set_xlabel('epochs')
    fig.savefig("picture/%s_loss.jpg"%dataname)

    ###f1
    fig2 = plt.figure()

    ax1 = fig2.add_subplot(111)
    ax1.plot(x, train_f1_history,label = "train_f1")
    ax1.set_ylim([0.0, 1.0])
    ax1.set_ylabel('train_f1_history')
    ax1.set_title("f1")

    ax2 = ax1.twinx()  # this is the important function
    ax2.plot(x, test_f1_history, 'r',label = "test_f1")
    # ax2.set_xlim([0, np.e])
    ax2.set_ylim([0.0, 1.0])
    ax2.set_ylabel('test_f1_history')
    ax2.set_xlabel('epochs')
    fig2.savefig("picture/%s_f1.jpg"%dataname)


def print_evaluation_result(result, binary_cls):
    predict_Y,target_Y,loss = result[0],result[1],result[2]

    print ('Confusion Metric')
    CM=str(metrics.confusion_matrix(target_Y, predict_Y))
    print (CM)
    print ('Accuracy')
    acc = metrics.accuracy_score(target_Y, predict_Y)
    print (acc)
    print ('loss')
    print (loss)
    print ('Micro Precision/Recall/F-score')
    print (metrics.precision_recall_fscore_support(target_Y, predict_Y, average='micro') )
    print ('Macro Precision/Recall/F-score')
    print (metrics.precision_recall_fscore_support(target_Y, predict_Y, average='macro') )
    print ('Each-class Precision/Recall/F-score')
    each_macrof1 = metrics.precision_recall_fscore_support(target_Y, predict_Y, average=None)[2]
    print (each_macrof1)
    f1 = metrics.precision_recall_fscore_support(target_Y, predict_Y, average='macro')[2]
    if(binary_cls!=-1):
        print('Binary F1 pos_label='+str(binary_cls))
        f1 = metrics.f1_score(target_Y, predict_Y,pos_label=0, average='binary')
        print(f1)
    return acc, f1, each_macrof1
def evaluation_result(result):
    predict_Y, target_Y, loss = result[0], result[1], result[2]
    CM = str(metrics.confusion_matrix(target_Y, predict_Y))
    return metrics.precision_recall_fscore_support(target_Y, predict_Y, average='macro')[2], CM

def evaluate(model, X1,X2, Y,pos_adjm,data_name,epoch,batch_size,discourse='implicit'):
    model.eval()
    predict1=[]
    predict2 = []
    predict3 = []
    loss_all=0.0
    for batch in train_bert_fineturn.batch_generator(X1,X2, Y,pos_adjm,batch_size):
        batch_X1, batch_X2, batch_y,batch_pos_adjm,batch_x= batch#test
        loss,logit1,logit2,logit3 = model(batch_X1, batch_X2, batch_y,batch_pos_adjm,batch_x,data_name,epoch,testing=False)
        predict1.extend(logit1.data.cpu().numpy())
        predict2.extend(logit2.data.cpu().numpy())
        predict3.extend(logit3.data.cpu().numpy())
        loss_all+=loss.data

    model.train()
    target = Y#numpy
    predict1=np.array(predict1)
    predict2 = np.array(predict2)
    predict3=np.array(predict3)

    predict1, target1 = process_label(predict1, target)
    predict2, target2 = process_label(predict2, target)
    predict3, target3 = process_label(predict3, target)

    predict1 = np.argmax(predict1, axis=1)
    target1 = np.argmax(target1, axis=1)

    predict2 = np.argmax(predict2, axis=1)
    target2 = np.argmax(target2, axis=1)

    predict3 = np.argmax(predict3, axis=1)
    target3 = np.argmax(target3, axis=1)

    f1_1,CM1=print_evaluation_result((predict1, target1, loss_all))
    f1_2, CM2 = evaluation_result((predict2, target2, loss_all))
    f1_3, CM3 = evaluation_result((predict3, target3, loss_all))

    (X1_x, X1_pos,X1__adj_matrix, X1_ner,_)=X1
    return loss_all/len(X1_x),f1_1,CM1,f1_2,CM2,f1_3,CM3

def evaluate2(model, X1,X2, Y,pos_adjm,data_name,epoch,batch_size,discourse='implicit'):
    model.eval()
    predict1=[]
    predict2 = []
    predict3 = []
    loss_all=0.0
    for batch in train_GCNbert.batch_generator(X1,X2, Y,pos_adjm,batch_size):
        batch_X1, batch_X2, batch_y,batch_pos_adjm,batch_x= batch#test
        loss,logit1,logit2,logit3 = model(batch_X1, batch_X2, batch_y,batch_pos_adjm,batch_x,data_name,epoch,testing=False)
        predict1.extend(logit1.data.cpu().numpy())
        predict2.extend(logit2.data.cpu().numpy())
        predict3.extend(logit3.data.cpu().numpy())
        loss_all+=loss.data

    model.train()
    target = Y#numpy
    predict1=np.array(predict1)
    predict2 = np.array(predict2)
    predict3=np.array(predict3)

    predict1, target1 = process_label(predict1, target)
    predict2, target2 = process_label(predict2, target)
    predict3, target3 = process_label(predict3, target)

    predict1 = np.argmax(predict1, axis=1)
    target1 = np.argmax(target1, axis=1)

    predict2 = np.argmax(predict2, axis=1)
    target2 = np.argmax(target2, axis=1)

    predict3 = np.argmax(predict3, axis=1)
    target3 = np.argmax(target3, axis=1)

    f1_1,CM1=print_evaluation_result((predict1, target1, loss_all))
    f1_2, CM2 = evaluation_result((predict2, target2, loss_all))
    f1_3, CM3 = evaluation_result((predict3, target3, loss_all))

    (X1_x, X1_pos,X1__adj_matrix, X1_ner,_)=X1
    return loss_all/len(X1_x),f1_1,CM1,f1_2,CM2,f1_3,CM3

def test_sentence2file(model, X1,X2, Y,pos_adjm,data_name,epoch,batch_size,data_dir,discourse='implicit'):
    discourse_sense_list = ['Temporal', 'Comparison', 'Contingency', 'Expansion']
    if data_name=="pdtb":
        model.eval()
        predict = []
        loss_all = 0.0
        for batch in train_GCNbert.batch_generator(X1, X2, Y,pos_adjm,batch_size):
            batch_X1, batch_X2, batch_y, batch_pos_adjm, batch_x = batch  # test
            loss,logit = model(batch_X1, batch_X2, batch_y,batch_pos_adjm,batch_x,data_name,epoch,testing=False)
            predict.extend(logit.data.cpu().numpy())
            loss_all += loss.data

        model.train()
        target=Y
        predict = np.array(predict)
        predict, target = process_label(predict, target)

        predict = np.argmax(predict, axis=1)
        target = np.argmax(target, axis=1)
        predict_relation=[discourse_sense_list[rel_index] for rel_index in predict]
        target_relation=[discourse_sense_list[rel_index] for rel_index in target]

        file = open(data_dir+'test_pdtb_sentence.json', 'r', encoding='utf-8')
        test_info = json.load(file)
        file2 = open(data_dir+'pdtb_word2id.json', 'r', encoding='utf-8')
        word2id = json.load(file2)
        ###判断test顺序能对得上#####
        a1=word2id[test_info[1][0].split()[1]]
        a2=X1[0][1][1]
        assert a1==a2

        ###("arg1", "arg2", "label_relation","target_relation", "predict_relation", "iscorrect")
        out_list=[]
        for i,discourse_relation in enumerate(test_info):
            arg1, arg2, _, relation, source_file = discourse_relation
            iscorrect=False
            if(predict[i]==target[i]):
                iscorrect=True
            out_list.append((arg1, arg2,relation,target_relation[i],predict_relation[i],iscorrect))
        def my_cmp(x,y):
            if x[5]!=y[5]:
                if(x[5]):
                    return 1
                else:
                    return -1
            else:
                return discourse_sense_list.index(x[3])-discourse_sense_list.index(y[3])
        from functools import cmp_to_key
        key = cmp_to_key(lambda x, y:my_cmp(x, y))
        out_list=sorted(out_list,key=key)
        j_test = json.dumps(out_list, indent=6, ensure_ascii=False)
        with open("picture/test_sentence_predict_result.json",
                  'w')  as f:  # 设置以utf-8解码模式读取文件，encoding参数必须设置，否则默认以gbk模式读取文件，当文件中包含中文时，会报错
            f.write(j_test)


if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--data_dir', type=str, default="data/prep_data/")
    parser.add_argument('--model_dir', type=str, default="model/")
    parser.add_argument('--domain', type=str, default="laptop")

    args = parser.parse_args()
    evaluate(args.runs, args.data_dir, args.model_dir, args.domain)
