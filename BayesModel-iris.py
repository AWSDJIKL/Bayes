# -*- coding: utf-8 -*-
'''
贝叶斯网络的实践-鸢尾花数据集
'''
# @Time : 2022/12/7 12:03
# @Author : LINYANZHEN
# @File : BayesModel-iris.py
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD

model = BayesianModel([('SepalLengthCm', 'Species'),
                       ('SepalWidthCm', 'Species'),
                       ('PetalLengthCm', 'Species'),
                       ('PetalWidthCm', 'Species')])
