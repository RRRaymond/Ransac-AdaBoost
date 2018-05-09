"""
@author: raymondchen
@date: 2018/5/8
Description:
    This is the class of weak classifier for AdaBoost.
    They are only vertical or horizontal lines.
"""


class WeakClassifier(object):

    def __init__(self, direction="vertical", positive=True, position=0):
        """
        :type direction:str
        :type positive:bool
        :type position:float
        """
        self.__direction = direction
        self.__positive = positive
        self.__position = position

    def predict(self, point):
        """
        :type point:list
        """
        if self.__direction == "vertical":
            result = 1 if point[0] > self.__position else -1
            return result if self.__positive else -result
        else:
            result = 1 if point[1] > self.__position else -1
            return result if self.__positive else -result

    def flip(self):
        return WeakClassifier(self.__direction, not self.__positive, self.__position)

    def info(self):
        return self.__direction, self.__positive, self.__position

    def equals(self, anotherclassifier):
        """
        :type anotherclassifier:WeakClassifier
        """
        if anotherclassifier.info() == self.info():
            return True
        return False
