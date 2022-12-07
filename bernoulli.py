# -*- coding: utf-8 -*-
'''

'''
# @Time : 2022/12/7 10:44
# @Author : LINYANZHEN
# @File : bernoulli.py
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import seaborn as sns


# 训练模型函数
def model_fit(x_train, y_train, x_test, y_test):
    clf = BernoulliNB()
    clf.fit(x_train, y_train)  # 对训练集进行拟合
    # print(x_test)
    # print(y_test)
    # print(clf.score(x_train, y_train))
    # print(clf.score(x_test, y_test))
    pred = clf.predict(x_test)
    print(pred)
    cm = confusion_matrix(y_test, pred)
    print(cm)
    return cm


# 混淆矩阵可视化
def matplotlib_show(cm):
    plt.figure(dpi=100)  # 设置窗口大小（分辨率）
    plt.title("BernoulliNB")
    labels = ['a', 'b', 'c', 'd']
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)
    sns.heatmap(cm, cmap=sns.color_palette("Blues"), annot=True, fmt='d')
    plt.ylabel('real_type')  # x坐标为实际类别
    plt.xlabel('pred_type')  # y坐标为预测类别
    plt.show()


if __name__ == '__main__':
    cancer = load_breast_cancer()
    x, y = cancer.data, cancer.target
    print(cancer)
    # print(x.shape)
    # print(y.shape)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=30)
    # cm = model_fit(x_train, y_train, x_test, y_test)
    # matplotlib_show(cm)
