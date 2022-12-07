# -*- coding: utf-8 -*-
'''
朴素贝叶斯实践，使用泰坦尼克号数据集
'''
# @Time : 2022/12/7 22:39
# @Author : LINYANZHEN
# @File : NaiveBayes.py
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix


def nb_fit(X, y):
    classes = y.unique()
    class_count = y.value_counts()
    class_prior = class_count / len(y)
    prior = dict()
    for col in X.columns:
        for j in classes:
            p_x_y = X[(y == j).values][col].value_counts()
            for i in p_x_y.index:
                prior[(col, i, j)] = p_x_y[i] / class_count[j]
    return classes, class_prior, prior


def predict(X_test):
    result = dict()
    for c in classes:
        p_y = class_prior[c]
        # 各条件概率相乘
        p_x_y = 1
        for i in X_test.items():
            p_x_y *= prior[tuple(list(i) + [c])]
        result[c] = p_y * p_x_y
    return result


train_data = pd.read_csv("train.csv")
# 只取 客舱等级、性别、兄弟数量、父母数量这4类特征，其他特征数据均有缺漏
x = train_data.loc[:, ["Pclass", "Sex", "SibSp", "Parch"]]
y = train_data["Survived"]
classes, class_prior, prior = nb_fit(x, y)
# 一共几类
print(classes)
# 类别的概率
print(class_prior)
# 各类条件概率
print(prior)

# 测试
test_data = pd.read_csv("test.csv")
y_true = pd.read_csv("gender_submission.csv")["Survived"]
test_data = test_data.loc[:, ["Pclass", "Sex", "SibSp", "Parch"]]
# 因为训练数据中这2列没有超过3的，将大于3的数据修改为3
test_data.loc[test_data['SibSp'] > 3, 'SibSp'] = 3
test_data.loc[test_data['Parch'] > 3, 'Parch'] = 3
pred = []
for i in range(len(test_data)):
    result = predict(test_data.iloc[i])
    pred.append(max(result, key=result.get))
pred = pd.Series(pred)
# 计算准确率，并画出混淆矩阵
accuracy = (pred == y_true).sum() / len(pred)
print(accuracy)
# # 画出混淆矩阵
labels = ["Death", "Survived"]
cm = confusion_matrix(y_true, pred)
print(cm)
plt.figure(dpi=400)
plt.title("Titanic Naive Bayes confusion matrix\naccuracy={:.3f}%".format(accuracy * 100))
sns.heatmap(cm, cmap=sns.color_palette("Blues"), annot=True, fmt='d', xticklabels=labels, yticklabels=labels)
plt.ylabel('real_type')  # y坐标为实际类别
plt.xlabel('pred_type')  # x坐标为预测类别
plt.savefig("confusionmatrix.png")
plt.show()
