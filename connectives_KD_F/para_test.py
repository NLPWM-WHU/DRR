from transformers import *
import numpy as np
# data_dir4 = "../interim/l/"
data_dir = "../interim/ji/"
pdtb_data_imp = np.load(data_dir + "pdtb2.npz", allow_pickle=True)
test_arg1_text = pdtb_data_imp['arg1_test']
test_arg2_text = pdtb_data_imp['arg2_test']
M21_data = np.load(data_dir + "predict_dep_M21.npz")
MTMD_data11 = np.load(data_dir + "MTMD_predict_dep11.npz", allow_pickle=True)
MTL_data = np.load(data_dir + "MTL_predict_dep11.npz", allow_pickle=True)

T11_data = np.load(data_dir + "predict_dep_MTMT_teacher11.npz")
S11_data = np.load(data_dir + "predict_dep_MTMT_11.npz")
otmt_teacher_data = np.load(data_dir + "OTMT_teacher_predict11.npz", allow_pickle=True)
OTMT_labels_data = np.load(data_dir + "OTMT_predict_dep11.npz", allow_pickle=True)
MTL_label = MTL_data['predict']
MTMD_label11 = MTMD_data11['predict']
T11_label = T11_data['predict']
S11_label = S11_data['predict']
M21_label = M21_data['predict']
otmt_teacher_label = otmt_teacher_data['predict']
OTMT_predict = OTMT_labels_data['predict']
true_label_11 = OTMT_labels_data['target']
# true_label_4 = S4_data['target']
for i in range(len(OTMT_predict)):
    max_label_S11 = np.argmax(S11_label[i])
    max_m21_label_11 = np.argmax(M21_label[i])
    max_mtl_label_11 = np.argmax(MTL_label[i])
    max_true_label_11 = np.argmax(true_label_11[i])
    max_otmt_label_11 = np.argmax(OTMT_predict[i])
    max_MTMD_label_11 = np.argmax(MTMD_label11[i])
    if max_label_S11 != max_true_label_11 and max_otmt_label_11 == max_true_label_11 and max_MTMD_label_11 != max_true_label_11 \
            and max_mtl_label_11 != max_true_label_11 and max_m21_label_11 != max_true_label_11:
        print(test_arg1_text[i])
        print(test_arg2_text[i])
        print("M21 label:")
        print(M21_label[i])
        print("MTMT_teacher:")
        print(T11_label[i])
        print('MTMT_S11:')
        print(S11_label[i])
        print('MTMD_label:')
        print(MTMD_label11[i])
        print('OTMT-teacher-data:')
        print(otmt_teacher_label[i])
        print('OTMT_label:')
        print(OTMT_predict[i])
        print(true_label_11[i])
print('OK')
# data_dir = "../interim/ji/"
# exp_data = np.load(data_dir + 'pdtb2expcon.npz')
# Y = exp_data['sense_train_id']
# exp_train_arg1_text = exp_data['arg1_train']
# exp_train_arg2_text = exp_data['arg2_train']
# id_11, count_11 = np.unique(Y, return_counts=True)
# Y_4 = Y.copy()
# Y_4[np.where(Y == 0)] = 0
# Y_4[np.where(Y == 1)] = 0
# Y_4[np.where(Y == 2)] = 1
# Y_4[np.where(Y == 3)] = 1
# Y_4[np.where(Y == 4)] = 2
# Y_4[np.where(Y == 5)] = 2
# Y_4[np.where(Y > 5)] = 3
# id_4, count_4 = np.unique(Y_4, return_counts=True)
# print('ok')