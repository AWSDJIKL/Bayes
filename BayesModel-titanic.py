# -*- coding: utf-8 -*-
'''
贝叶斯网络的实践-泰坦尼克号数据集
'''
# @Time : 2022/12/7 11:15
# @Author : LINYANZHEN
# @File : BayesModel-titanic.py
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD

model = BayesianModel([('Age', 'Survived'),
                       ('Sex', 'Survived'),
                       ('Fare','Pclass'),
                       ('Pclass','Survived'),
                       ('Cabin','Survived')])

