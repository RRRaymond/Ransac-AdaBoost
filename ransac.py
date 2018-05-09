"""
@author: raymondchen
@date: 2018/5/7
Description:
"""


import numpy as np
import pandas as pd

class Ransac(object):
    def __init__(self, data, threshold_t, threshold_T, N):
        """
        :type data: pd.DataFrame
        :type threshold_T: int
        :type threshold_t: int
        :type N: int
        """
        self.__data = data
        self.__t = threshold_t
        self.__T = threshold_T
        self.__N = N
        self.__weight = []
        self.__inlier = pd.DataFrame()

    def train(self):
        inlier = pd.DataFrame(columns=list("xy"))
        for i in range(self.__N):
            sample = self.__data.sample(n=2)
            k = (sample.iloc[0, 1]-sample.iloc[1, 1])/(sample.iloc[0, 0]-sample.iloc[1, 0])
            b = sample.iloc[0, 1] - sample.iloc[0, 0] * k
            temp_inlier = pd.DataFrame(columns=list("xy"))
            for j in range(len(self.__data)):
                if abs(self.__data.iloc[j, 0] * k + b - self.__data.iloc[j, 1]) < self.__t:
                    temp_inlier.loc[len(temp_inlier)] = self.__data.iloc[j]
            if len(temp_inlier) > len(inlier):
                inlier = temp_inlier
            if len(temp_inlier) > self.__T:
                break
        x_data = pd.DataFrame(inlier["x"])
        x_data['1'] = 1
        x_data = np.array(x_data)
        y_data = pd.DataFrame(inlier["y"])
        y_data = np.array(y_data)
        self.__weight = np.dot(np.dot(np.mat(np.dot(x_data.T, x_data)).I,x_data.T),y_data)
        self.__inlier = inlier

    def getweight(self):
        return self.__weight

    def getinlier(self):
        return self.__inlier