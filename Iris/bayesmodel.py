# -*- coding: utf-8 -*-
'''

'''
import numpy as np
from pgmpy.estimators import BayesianEstimator
# @Time : 2022/12/7 14:45
# @Author : LINYANZHEN
# @File : bayesmodel.py
from pgmpy.models import BayesianModel, BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
import pandas as pd
from graphviz import Digraph
import os  # 以下这两行是手动进行环境变量配置，防止在本机的环境变量部署失败
import seaborn as sns
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt

os.environ['PATH'] = os.pathsep + r'C:/Program Files/Graphviz/bin'


def showBN(model, save=False):
    '''传入BayesianModel对象，调用graphviz绘制结构图，jupyter中可直接显示'''

    node_attr = dict(
        style='filled',
        shape='box',
        align='left',
        fontsize='12',
        ranksep='0.1',
        height='0.2'
    )
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()
    edges = model.edges()
    for a, b in edges:
        cpds = model.get_cpds(a).values
        dot.node(a, label="{%s|{0|%.2f}|{1|%.2f}|{2|%.2f}}" % (a, cpds[0], cpds[1], cpds[2]),
                 _attributes=dict(shape="record"))
        dot.edge(a, b)

    if save:
        dot.view(cleanup=True)
    return dot


model = BayesianNetwork([('SepalLengthCm', 'Species'),
                         ('SepalWidthCm', 'Species'),
                         ('PetalLengthCm', 'Species'),
                         ('PetalWidthCm', 'Species')])

train_data = pd.read_csv("Iris-train.csv")
test_data = pd.read_csv("Iris-test-gt.csv")
test_x = test_data.drop(columns=["Species"])
model.fit(train_data)
showBN(model, True)
# 在测试集上测试模型准度
pred = model.predict((test_x))

# 输出模型在测试集的精度
accuracy = (pred["Species"] == test_data["Species"]).sum() / len(pred)
print(accuracy)
# 画出混淆矩阵
labels = ["Iris-versicolor", "Iris-setosa", "Iris-virginica"]
cm = confusion_matrix(test_data["Species"], pred["Species"],
                      labels=labels)
print(cm)
plt.figure(dpi=400)
plt.title("Iris Bayes Network confusion matrix\naccuracy={:.3f}%".format(accuracy * 100))
sns.heatmap(cm, cmap=sns.color_palette("Blues"), annot=True, fmt='d', xticklabels=labels, yticklabels=labels)
plt.ylabel('real_type')  # y坐标为实际类别
plt.xlabel('pred_type')  # x坐标为预测类别
plt.savefig("confusionmatrix.png")
plt.show()
