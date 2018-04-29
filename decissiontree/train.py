# !/usr/bin/env python
# -*-coding:utf-8-*-

"""
@author: xhades
@Date: 2018/4/3

"""

from sklearn import tree

X = [[0, 0], [1, 1]]
Y = [0, 1]
clf = tree.DecisionTreeClassifier()
clf.fit(X, Y)
pred = clf.predict([[2., 2.]])
print(pred)