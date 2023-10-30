from custom_tools import *
import time
import torch
import torch.nn as nn
import numpy as np
from nn_model.FusionNet import FusionNet
from load_batch import BatchDataLoadImgWavText, split_train_test
from torch.utils.data import DataLoader
import os
from sklearn.metrics import recall_score, f1_score, precision_score, accuracy_score
import warnings
warnings.filterwarnings("ignore")

np.random.seed(3407)
torch.manual_seed(3407)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(3407)
    torch.backends.cudnn.deterministic = True


class Trainer(object):
    def __init__(self,
                 learn_rate,
                 batch_size,
                 thresh):

        super(Trainer, self).__init__()

        self.class_number = 7
        self.batch_size = batch_size
        self.lr = learn_rate
        self.clip = -1
        self.pre_train = 'bert-base-uncased'
        self.thresh = thresh

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion1 = nn.CrossEntropyLoss().to(self.device)
        self.criterion2 = nn.CrossEntropyLoss().to(self.device)
        self.criterion3 = nn.CrossEntropyLoss().to(self.device)

        self.model = FusionNet(bert_path=self.pre_train)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.model = self.model.to(self.device)

    def train(self, epoch, train_loader, train_len):

        self.model.train()
        correct = 0
        train_loss, train_acc, = 0, 0
        loss_1, loss_2, loss_3 = 0, 0, 0
        for batch_idx, batch in enumerate(train_loader):

            batch = [i.to(self.device) for i in batch]
            img_feature, mfcc_feature, input_ids, mask, target = batch[0], batch[1], batch[2], batch[3], batch[4]
            text_encode = {"input_ids": input_ids, "attention_mask": mask}
            k1, k2, k3 = self.model(text_encode, img_feature, mfcc_feature)

            loss1 = self.criterion1(k1, target.long())
            loss2 = self.criterion2(k2, target.long())
            loss3 = self.criterion3(k3, target.long())

            loss = 0.7 * loss1 + 0.2 * loss2 + 0.1 * loss3
            self.optimizer.zero_grad()
            if self.clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

            loss.backward()
            self.optimizer.step()
            train_loss += loss * target.size(0)
            loss_1 += loss1 * target.size(0)
            loss_2 += loss2 * target.size(0)
            loss_3 += loss3 * target.size(0)

            argmax = torch.argmax(k1, 1)
            train_acc += (argmax == target).sum()
            pred = k1.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

            if batch_idx % 20 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * self.batch_size,
                    train_len, 100. * batch_idx * self.batch_size / train_len,
                    loss.item()))
        train_loss = torch.true_divide(train_loss, train_len)
        train_loss1 = torch.true_divide(loss_1, train_len)
        train_loss2 = torch.true_divide(loss_2, train_len)
        train_loss3 = torch.true_divide(loss_3, train_len)
        train_acc = torch.true_divide(train_acc, train_len)
        print('Train set: Average loss: {:.6f}, Accuracy: {}/{} ({:.5f}%)'.format(
            train_loss, correct, train_len, 100. * correct / train_len))
        return train_loss, train_acc, train_loss1, train_loss2, train_loss3

    def evaluate(self, epoch, test_loader):
        global best_acc
        self.model.eval()
        with torch.no_grad():
            k1_out, k2_out, k3_out = torch.zeros((1, 7)), torch.zeros((1, 7)), torch.zeros((1, 7))
            tar = []
            out = np.zeros(shape=(1, 7))
            for test_idx, test_batch in enumerate(test_loader):
                batch = [i.to(self.device) for i in test_batch]
                img_feature, mfcc_feature, input_ids, mask, target = batch[0], batch[1], batch[2], batch[3], batch[4]
                text_encode = {"input_ids": input_ids, "attention_mask": mask}

                k1, k2, k3 = self.model(text_encode, img_feature, mfcc_feature)

                k1_out = torch.vstack((k1_out, k1.cpu()))
                k2_out = torch.vstack((k2_out, k2.cpu()))
                k3_out = torch.vstack((k3_out, k3.cpu()))
                out = np.vstack((out, k1.cpu().numpy()))
                tar.extend(target.cpu().numpy())

            k1_out, k2_out, k3_out = k1_out[1:], k2_out[1:], k3_out[1:]
            arg, label = self.score_layer(layer_out=[k1_out, k2_out, k3_out],
                                          target=tar,
                                          thresh=self.thresh)

            test_acc = accuracy_score(y_true=label, y_pred=arg)
            print('\ntest set: Accuracy: ({:.5f}%), Best_Accuracy({:.5f})'.format(
                test_acc * 100, best_acc))
            if test_acc > best_acc:
                best_acc = test_acc
                print('The effect becomes better ...')
                visualization(output=out[1:],
                              target=tar,
                              labels=["anger", "contempt", "disgust",
                                      "fear", "happy", "sadness", "surprise"],
                              name="FusionNet")
                p = precision_score(label, arg, average='macro')
                recall = recall_score(label, arg, average='macro')
                f1 = f1_score(label, arg, average='macro')

                cm = confusion_matrix(label, arg)
                plot_confusion_matrix_red(cm=cm,
                                          savename=r"result/FusionNet/Confusion-Matrix-FusionNet.png",
                                          title=r"Confusion Matrix FusionNet",
                                          classes=["anger", "contempt", "disgust", "fear", "happy", "sadness",
                                                   "surprise"])

                plot_confusion_matrix_sim(cm=cm,
                                          savename=r"result/FusionNet/Confusion-Matrix-FusionNet-Sim.png",
                                          title=r"Confusion Matrix FusionNet SIM",
                                          classes=["anger", "contempt", "disgust", "fear", "happy", "sadness",
                                                   "surprise"])

                plot_confusion_matrix(cm=cm,
                                      savename=r"result/FusionNet/Confusion-Matrix.png",
                                      classes=["anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"])

                result_text = r'result/FusionNet/FusionNet_Record.txt'
                file_handle = open(result_text, mode='a+')
                file_handle.write('epoch:{},test_acc:{}, precision:{}, recall:{},f1_score:{}\n'.format(
                    epoch, test_acc, p, recall, f1
                ))
                file_handle.close()
            return test_acc

    @staticmethod
    def score_layer(layer_out, target, thresh):

        def bool_max(input_data, threshold=None):
            values, indices = torch.max(input_data, dim=1)
            values = values.tolist()
            if threshold:
                bool_value = [(value > float(threshold)) + 0 for value in values]
            else:
                bool_value = [1] * len(values)
            k1_argmax = indices.tolist()
            return bool_value, k1_argmax

        def score(bools, arg, labels, in_data):

            true_list = [index for index, value in enumerate(bools) if value == 1]
            arg = [arg[index] for index in true_list]
            label = [labels[index] for index in true_list]

            false_list = [index for index, value in enumerate(bools) if value == 0]
            last_label = [labels[index] for index in false_list]
            last_data = [in_data[index].tolist() for index in false_list]
            last_data = torch.tensor(last_data, dtype=torch.float32)
            return arg, label, last_label, last_data

        k1, k2, k3 = layer_out[0], layer_out[1], layer_out[2]
        k1_bool, argmax_k1 = bool_max(k1, threshold=thresh)
        arg_k1, labels_k1, last_label_k1, last_data_k1 = score(k1_bool, argmax_k1, target, k1)

        k2_bool, argmax_k2 = bool_max(last_data_k1, threshold=thresh)
        arg_k2, labels_k2, last_label_k2, last_data_k2 = score(k2_bool, argmax_k2, last_label_k1, last_data_k1)

        k3_bool, argmax_k3 = bool_max(last_data_k2, threshold=None)
        arg_k3, labels_k3, _, _ = score(k3_bool, argmax_k3, last_label_k2, last_data_k2)

        args = arg_k1 + arg_k2 + arg_k3
        las = labels_k1 + labels_k2 + labels_k3
        return args, las


def mian(learn_rate, batch_size, patience, mfcc_norm_way="max", thresh=0.7):
    if not os.path.exists(f"result/FusionNet"):
        os.makedirs(f"result/FusionNet")

    train_set, test_set = split_train_test(pkl_path='data/processed_pkl/feature.pkl')

    data_train_loader = BatchDataLoadImgWavText(batch_data=train_set,
                                                expression_file='data/dataset/CK+',
                                                wave_file='data/dataset/RAVDESS',
                                                state='train',
                                                mfcc_norm_way=mfcc_norm_way)

    data_test_loader = BatchDataLoadImgWavText(batch_data=test_set,
                                               expression_file='data/dataset/CK+',
                                               wave_file='data/dataset/RAVDESS',
                                               state='test',
                                               mfcc_norm_way=mfcc_norm_way)

    train_loader = DataLoader(data_train_loader, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(data_test_loader, batch_size=batch_size, shuffle=True)

    train_len, test_len = len(train_set["expression"]), len(test_set["expression"])

    loss_train, acc_train, acc_test = [], [], []
    train_loss1, train_loss2, train_loss3 = [], [], []
    epoch_list = []
    start = time.time()

    T = Trainer(learn_rate, batch_size, thresh=thresh)
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    for epoch in range(10000):
        epoch_list.append(epoch)
        train_loss, train_acc, loss1, loss2, loss3 = T.train(epoch=epoch,
                                                             train_loader=train_loader,
                                                             train_len=train_len)

        test_acc = T.evaluate(epoch=epoch,
                              test_loader=test_loader)

        if torch.cuda.is_available():
            loss_train.append(train_loss.cuda().data.cpu().numpy())
            acc_train.append(train_acc.cuda().data.cpu().numpy())
            acc_test.append(test_acc)
            train_loss1.append(loss1.cuda().data.cpu().numpy())
            train_loss2.append(loss2.cuda().data.cpu().numpy())
            train_loss3.append(loss3.cuda().data.cpu().numpy())

        else:
            loss_train.append(train_loss.detach().numpy())
            acc_train.append(train_acc.detach().numpy())
            acc_test.append(test_acc)
            train_loss1.append(loss1.detach().numpy())
            train_loss2.append(loss2.detach().numpy())
            train_loss3.append(loss3.detach().numpy())

        early_stopping(test_acc)
        if early_stopping.early_stop:
            print(" === > Early stopping ! ! ! ! ! ")
            break
        print("....................... . Next . .......................")

    end = time.time()
    train_time = end - start
    print("训练时间长度为  ==== > {} s".format(train_time))

    plot_curve_train_acc(epoch_list, acc_train,
                         savename=f"result/FusionNet/train-acc-FusionNet",
                         title=f"train acc FusionNet")

    plot_curve_train_loss(epoch_list, loss_train,
                          savename=f"result/FusionNet/train-loss-FusionNet",
                          title=f"train loss FusionNet")

    plot_curve_test_acc(epoch_list, acc_test,
                        savename=f"result/FusionNet/test-acc-FusionNet",
                        title=f"test acc FusionNet")

    plot_curve_train_test_acc(epoch_list, acc_train, acc_test,
                              savename=f"result/FusionNet/train-and-test-acc-FusionNet",
                              title=f"train and test acc FusionNet")

    plot_curve_train_loss1_loss2_loss3(epoch_list, train_loss1, train_loss2, train_loss3,
                                       savename=f"result/FusionNet/different-layer-train-loss-FusionNet",
                                       title=f"different-layer-train-loss-FusionNet")


if __name__ == '__main__':
    best_acc = 0
    import os

    print(f'cuda is use: {torch.cuda.is_available()}')
    if torch.cuda.is_available() is True:
        print(f'use {torch.cuda.get_device_name()} train ')
    print('start train ......')

    mian(learn_rate=2e-4, batch_size=16,  patience=6)
