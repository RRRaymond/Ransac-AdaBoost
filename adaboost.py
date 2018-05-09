"""
@author: raymondchen
@date: 2018/5/8
Description:
    This is the class of the adaboost classifier.
    AdaBoost is combined with several weak classifier and has a strong ability.
"""
import numpy as np
import pandas as pd
from classifier import WeakClassifier
from math import log
from math import e
from matplotlib import pyplot as plt


class AdaBoost(object):
    def __init__(self, data, iterations):
        """
        :type data:pd.DataFrame
        """
        self.__data = data
        self.__iterations = iterations
        self.__weight = np.array([1/len(data)] * len(data))
        self.__classifier = []

    def train(self):
        strongclassifier = []
        for i in range(self.__iterations):
            best_weakclassifier = WeakClassifier()
            error_rate = 1
            for j in range(0, 400, 10):
                # vertical
                m_weakclasssifier = WeakClassifier("vertical", True, j)
                m_error = 0
                for index, row in self.__data.iterrows():
                    m_prediction = m_weakclasssifier.predict(row)
                    m_error += self.__weight[index] if m_prediction != row[2] else 0
                if m_error < error_rate:
                    best_weakclassifier = m_weakclasssifier
                    error_rate = m_error
                elif 1 - m_error < error_rate:
                    best_weakclassifier = m_weakclasssifier.flip()
                    error_rate = 1 - m_error
                # horizontal
                m_weakclasssifier = WeakClassifier("horizontal", True, j)
                m_error = 0
                for index, row in self.__data.iterrows():
                    m_prediction = m_weakclasssifier.predict(row)
                    m_error += self.__weight[index] if m_prediction != row[2] else 0
                if m_error < error_rate:
                    best_weakclassifier = m_weakclasssifier
                    error_rate = m_error
                elif 1 - m_error < error_rate:
                    best_weakclassifier = m_weakclasssifier.flip()
                    error_rate = 1 - m_error
            if error_rate >= 0.5:
                break
            flag = False
            for j in range(len(strongclassifier)):
                if best_weakclassifier.equals(strongclassifier[j][0]):
                    flag = True
                    break
            if flag:
                break
            alpha = 0.5 * log((1 - error_rate)/error_rate)
            strongclassifier.append((best_weakclassifier, alpha))
            for index in range(len(self.__weight)):
                prediction_result = -alpha * self.__data.iloc[index, 2] \
                                    * best_weakclassifier.predict(self.__data.iloc[index])
                self.__weight[index] = self.__weight[index] * e ** prediction_result
            m_sum = sum(self.__weight)
            self.__weight = np.array(self.__weight)/m_sum
        self.__classifier = strongclassifier

    def predict(self, point):
        """
        :type point:list
        """
        result = 0
        for i in range(len(self.__classifier)):
            result += self.__classifier[i][1] * self.__classifier[i][0].predict(point)
        if result > 0:
            return 1
        elif result < 0:
            return -1
        else:
            return 0

    def test(self, data):
        """
        :type data:pd.DataFrame
        """
        for i in range(len(data)):
            print(self.predict(data.iloc[i]) == data.iloc[i, 2])

    def display(self):
        data = self.__data
        f1 = plt.figure(2)
        positive = data[data["label"] == 1]
        negative = data[data["label"] == -1]
        positive_weight = self.__weight[data["label"] == 1]
        negative_weight = self.__weight[data["label"] == -1]
        plt.scatter(positive.iloc[:, 0], positive.iloc[:, 1], c="red", marker="+", s=positive_weight*1000)
        plt.scatter(negative.iloc[:, 0], negative.iloc[:, 1], c="green", s=negative_weight*1000)
        for i in range(len(data)):
            plt.text(x=data.iloc[i, 0], y=data.iloc[i, 1]+20, s=round(self.__weight[i], 4))
        for j in range(len(self.__classifier)):
            if self.__classifier[j][0].info()[0] == "vertical":
                x = self.__classifier[j][0].info()[2]
                line_x = [x, x]
                line_y = [0, 350]
                plt.plot(line_x, line_y)
            else:
                y = self.__classifier[j][0].info()[2]
                line_x = [50, 350]
                line_y = [y, y]
                plt.plot(line_x, line_y)
        plt.show()
