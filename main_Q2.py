"""
@author: raymondchen
@date: 2018/5/8
Description:
    This is the main of the AdaBoost algorithm.
    It contains a raw data of 10 point from 2 class.
"""

from adaboost import AdaBoost
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

data = pd.DataFrame(np.array([
    [88, 144, 1],
    [93, 232, 1],
    [136, 275, -1],
    [147, 131, -1],
    [159, 69, 1],
    [214, 31, 1],
    [214, 152, -1],
    [257, 83, 1],
    [307, 62, -1],
    [307, 231, -1]
]), columns=["x", "y", "label"])


def display():
    f1 = plt.figure(1)
    positive = data[data["label"] == 1]
    negative = data[data["label"] == -1]
    plt.scatter(positive.iloc[:, 0], positive.iloc[:, 1], c="red", marker="+")
    plt.scatter(negative.iloc[:, 0], negative.iloc[:, 1], c="green")
    plt.show()


if __name__ == '__main__':
    m_ada = AdaBoost(data, 5)
    display()
    m_ada.train()
    m_ada.display()
