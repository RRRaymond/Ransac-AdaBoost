"""
@author: raymondchen
@date: 2018/5/7
Description:
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ransac import Ransac


"""
    This is the origin data to fit a straight 2D line.
    We will use ransac, because there are some outlier points.
"""
data = pd.DataFrame(np.array([
    [-2, 0],
    [0, 0.9],
    [2, 2.0],
    [3, 6.5],
    [4, 2.9],
    [5, 8.8],
    [6, 3.95],
    [8, 5.03],
    [10, 5.97],
    [12, 7.1],
    [13, 1.2],
    [14, 8.2],
    [16, 8.5],
    [18, 10.1]
]), columns=list("xy"))


def display():
    line_x = [round(min(inlier["x"])) - 1, round(max(inlier["x"])) + 1]
    line_y = [x * weight[0, 0] + weight[1, 0] for x in line_x]
    f1 = plt.figure(1)
    plt.subplot(211)
    plt.title("origin data")
    plt.scatter(data.iloc[:, 0], data.iloc[:, 1])
    plt.subplot(212)
    plt.title("ransac fitting")
    plt.text(line_x[0]+2, line_y[0], "y = " + str(weight[0, 0])[:6] + " * x + " + str(weight[1, 0])[:6])
    plt.scatter(data.iloc[:, 0], data.iloc[:, 1])
    plt.scatter(inlier.iloc[:, 0], inlier.iloc[:, 1])
    plt.plot(line_x, line_y)
    plt.show()


if __name__ == "__main__":
    m_ransac = Ransac(data, threshold_t=1, threshold_T=10, N=20)
    m_ransac.train()
    weight = m_ransac.getWeight()
    inlier = m_ransac.getInlier()
    display()

