import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns
import torch
import random
from itertools import cycle
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_score, f1_score, confusion_matrix, accuracy_score
import re
import pickle


# def main_creditcard(output, target, name):
#
#     df, predictions = output, target
#
#     Xtsne = TSNE(n_components=2).fit_transform(df)
#
#     predictions_list = []
#     for x in predictions:
#         predictions_list.append(int(x))
#
#     dftsne = pd.DataFrame(data=Xtsne, columns=['dimX', 'dimY'])
#     dftsne['cluster'] = predictions_list
#     plt.figure(figsize=(6, 5))
#     sns.scatterplot(data=dftsne, x='dimX', y='dimY', hue='cluster', legend="full", palette='tab10', alpha=0.5)
#     plt.title(f'{name} features visualization')
#     plt.savefig(f'result/{name}/{name} features visualization', dpi=300)
#     plt.close()


def visualization(output, target, labels, name):
    data, predictions = output, target

    tsne = TSNE(n_components=2, random_state=42)
    reduced_data = tsne.fit_transform(data)

    plt.figure(figsize=(10, 8))
    colors = {0: 'b', 1: 'g', 2: 'r', 3: 'c', 4: 'm', 5: 'y', 6: 'k'}  # 定义每个类别的颜色

    # for i in range(len(labels)):
    #     plt.scatter(reduced_data[i, 0], reduced_data[i, 1], c=colors[i], label=labels[i])

    l = []
    for m, p in zip(reduced_data, predictions):
        p = int(p)
        if p in l:
            plt.scatter(m[0], m[1], c=colors[p])
        else:
            plt.scatter(m[0], m[1], c=colors[p], label=labels[p])
        l.append(p)

    plt.legend()
    # sns.scatterplot(data=dftsne, x='dimX', y='dimY', hue='cluster', legend="full", palette='tab10', alpha=0.5)
    plt.title(f'{name} features visualization')
    plt.savefig(f'result/{name}/{name} features visualization.png', dpi=300)
    plt.close()


def plot_confusion_matrix(cm, savename, classes):  # 画混淆矩阵

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    fig.colorbar(cax)

    ax.set_xticklabels([""] + classes, rotation=45)
    ax.set_yticklabels([""] + classes)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.savefig(savename)
    plt.close()


def plot_confusion_matrix_sim(cm, savename, title, classes):  # 画混淆矩阵
    plt.figure(figsize=(12, 8), dpi=100)
    np.set_printoptions(precision=2)
    # 在混淆矩阵中每格的概率值
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    ind_array = np.arange(len(classes))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val]
        plt.text(x_val, y_val, "%0.3f" % (c,), color='red', fontsize=15, va='center', ha='center')
        # plt.text(x_val, y_val, c, color='red', fontsize=15, va='center', ha='center')

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.binary)
    plt.title(title, fontsize=20)
    plt.colorbar()
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, classes, rotation=30, fontsize=15)
    plt.yticks(xlocations, classes, fontsize=15)
    plt.ylabel('Actual label', fontsize=15)
    plt.xlabel('Predict label', fontsize=15)

    # offset the tick
    tick_marks = np.array(range(len(classes))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)

    # show confusion matrix
    plt.savefig(savename)
    plt.cla()
    plt.close()


def plot_confusion_matrix_red(cm, savename, title, classes):  # 画混淆矩阵
    plt.figure(figsize=(12, 8), dpi=100)
    np.set_printoptions(precision=2)
    # 在混淆矩阵中每格的概率值
    # cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    ind_array = np.arange(len(classes))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val]
        # plt.text(x_val, y_val, "%0.3f" % (c,), color='red', fontsize=15, va='center', ha='center')
        plt.text(x_val, y_val, c, color='red', fontsize=15, va='center', ha='center')

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.binary)
    plt.title(title, fontsize=20)
    plt.colorbar()
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, classes, rotation=30, fontsize=15)
    plt.yticks(xlocations, classes, fontsize=15)
    plt.ylabel('Actual label', fontsize=15)
    plt.xlabel('Predict label', fontsize=15)

    # offset the tick
    tick_marks = np.array(range(len(classes))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)

    # show confusion matrix
    plt.savefig(savename)
    plt.cla()
    plt.close()


def plot_curve_train_acc(epoch_list, train_acc, savename, title):
    epoch = epoch_list
    fig, ax = plt.subplots()
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.plot(epoch, train_acc, label="train-acc")
    plt.title('{}'.format(title))
    plt.ylabel('Accuracy')
    plt.legend(loc='best')
    plt.xlabel('Epoch')
    plt.legend(loc='best')
    plt.savefig('{}.png'.format(savename))
    plt.close()


def plot_curve_train_loss1_loss2_loss3(epoch_list, loss1, loss2, loss3, savename, title):
    epoch = epoch_list
    fig, ax = plt.subplots()
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.plot(epoch, loss1, label="train-loss1")
    plt.plot(epoch, loss2, label="train-loss2")
    plt.plot(epoch, loss3, label="train-loss3")
    plt.title('{}'.format(title))
    plt.ylabel('Loss')
    plt.legend(loc='best')
    plt.xlabel('Epoch')
    plt.legend(loc='best')
    plt.savefig('{}.png'.format(savename))
    plt.close()


def plot_curve_train_loss(epoch_list, train_loss, savename, title):
    epoch = epoch_list
    fig, ax = plt.subplots()
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.plot(epoch, train_loss, label="train-loss")
    plt.title('{}'.format(title))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='best')
    plt.savefig('{}.png'.format(savename))
    plt.close()


def plot_curve_test_acc(epoch_list, train_acc, savename, title):
    epoch = epoch_list
    fig, ax = plt.subplots()
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.plot(epoch, train_acc, label="test-acc")
    plt.title('{}'.format(title))
    plt.ylabel('Accuracy')
    plt.legend(loc='best')
    plt.xlabel('Epoch')
    plt.legend(loc='best')
    plt.savefig('{}.png'.format(savename))
    plt.close()


def plot_curve_train_test_acc(epoch_list, train_acc, test_acc, savename, title):
    epoch = epoch_list
    fig, ax = plt.subplots()
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.plot(epoch, train_acc, label="train-acc")
    plt.plot(epoch, test_acc, label="test-acc")
    plt.title('{}'.format(title))
    plt.ylabel('Accuracy')
    plt.legend(loc='best')
    plt.xlabel('Epoch')
    plt.legend(loc='best')
    plt.savefig('{}.png'.format(savename))
    plt.close()


class EarlyStopping():  # 提前停止
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=12, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.test_acc_min = np.Inf
        self.delta = delta

    def __call__(self, test_acc):

        score = test_acc

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(test_acc)
        elif score <= self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter ----- > : {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(test_acc)
            self.counter = 0

    def save_checkpoint(self, test_acc):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'test acc ----> ({self.test_acc_min:.6f} --> {test_acc:.6f}).  Saving model ...')
        self.test_acc_min = test_acc
