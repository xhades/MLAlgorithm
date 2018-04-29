# !/usr/bin/env python
# -*-coding:utf-8-*-

"""
@author: xhades
@Date: 2018/4/21

"""
from numpy import *

"""
《机器学习实战》轮子
代码理解和注释
"""


def loadsimpData():
    """
    加载数据
    :return:
    """
    dataMat=matrix([[1,2.1],
                    [2,1.1],
                    [1.3,1],
                    [1,1],
                    [2,1]])
    classLabels=[1,1,-1,-1,1]
    return dataMat,classLabels


def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    """

    :param dataMatrix:
    :param dimen: 特征
    :param threshVal: 特征阈值
    :param threshIneq: 大于还是小于
    :return:
    """
    retArray = ones((shape(dataMatrix)[0],1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0

    return retArray


# D是权重向量
def buildStump(dataArr,classLabels,D):
    dataMatrix=mat(dataArr);labelMat=mat(classLabels).T
    m,n=shape(dataMatrix)
    numSteps=10.0
    bestStump={}
    bestClassEst=mat(zeros((m,1)))
    # 最小值初始化为无穷大
    minError=inf
    # 对每一个特征
    for i in range(n):
        # 找到最大值和最小值
        rangeMin=dataMatrix[:,i].min()
        rangeMax=dataMatrix[:,i].max()
        # 确定步长stepSize
        stepSize=(rangeMax-rangeMin)/numSteps
        for j in range(-1,int(numSteps)+1):
            for inequal in ['lt','gt']:
                # 得到阈值
                threshVal=(rangeMin+float(j)*stepSize)
                # 调用函数，并得到分类列表
                predictedVals=stumpClassify(dataMatrix,i,threshVal,inequal)
                # 初始化errArr
                errArr=mat(ones((m,1)))
                # 将errArr中分类正确的置为0
                errArr[predictedVals==labelMat]=0
                # 计算加权错误率
                weightedError=D.T*errArr
                print("split:dim %d,thresh %.2f,thresh inequal:"
                       "%s,the weighted error is %.3f"%(i,threshVal,
                         inequal,weightedError))
                # 如果错误率比之前的小
                if(weightedError<minError):
                    minError=weightedError
                    # bestClassEst中是错误最小的分类类别
                    bestClassEst=predictedVals.copy()
                    bestStump['dim']=i
                    bestStump['thresh']=threshVal
                    bestStump['ineq']=inequal
    return bestStump,minError,bestClassEst


if __name__ == '__main__':
    datMat, classLabels = loadsimpData()
    print(datMat, classLabels)
    D = mat(ones((5,1))/5)
    buildStump(datMat, classLabels, D)