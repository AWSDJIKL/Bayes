# -*- coding: utf-8 -*-
'''
鸢尾花数据集数据预处理，将连续型数据离散化
'''
# @Time : 2022/12/7 12:08
# @Author : LINYANZHEN
# @File : data-preprocess.py

'''
Id：编号，无用属性
SepalLengthCm：萼片长度，使用聚类分为3类
SepalWidthCm：萼片宽度，使用聚类分为3类
PetalLengthCm：花瓣长度，使用聚类分为3类
PetalWidthCm：花瓣宽度，使用聚类分为3类
Species：鸢尾花种类，一共三分类
'''
from sklearn.cluster import KMeans
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

# 读取数据
data = pd.read_csv("Iris.csv")
# 画出数据分布，解释为何离散化是聚为3类
plt.figure(dpi=400)
sns.FacetGrid(data, hue="Species").map(sns.distplot, "PetalLengthCm").add_legend()
plt.savefig("PetalLengthCm.png")
sns.FacetGrid(data, hue="Species").map(sns.distplot, "PetalWidthCm").add_legend()
plt.savefig("PetalWidthCm.png")
sns.FacetGrid(data, hue="Species").map(sns.distplot, "SepalLengthCm").add_legend()
plt.savefig("SepalLengthCm.png")
sns.FacetGrid(data, hue="Species").map(sns.distplot, "SepalWidthCm").add_legend()
plt.savefig("SepalWidthCm.png")
plt.show()

for column in ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]:
    c_data = data[column].values.reshape((-1, 1))
    km = KMeans(n_clusters=3).fit(c_data)
    c_data = km.fit_predict(c_data)
    data[column] = c_data
# print(data)
data.to_csv("Iris-discretization.csv", index=False)
# 抽样出训练集和测试集,训练集100条，测试集50条
data = data.sample(frac=1)
print(data)
train_data = data.iloc[:100]
test_data = data.iloc[100:]
print(train_data)
print(test_data)
train_data.to_csv("Iris-train.csv", index=False)
test_data.to_csv("Iris-test.csv", index=False)
