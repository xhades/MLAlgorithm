# !/usr/bin/env python
# -*-coding:utf-8-*-

"""
@author: xhades
@Date: 2018/3/26

"""


from sklearn import linear_model
import numpy as np

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit([[0, 0], [1, 1], [2, 2]], [0, 1, 2])

# Predict
regr.predict(np.array([1, 1]).reshape(1, -1))