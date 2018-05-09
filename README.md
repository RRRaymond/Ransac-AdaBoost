# Ransac and AdaBoost algorithm 

This is the second homework of comupter vision course. I implement this two algorithm based on python instead of Matlab .

**Ransac:**

RANSAC is widely used in fitting models from sample points with outliers. In this project, i use this algorithm to fit a 2D line using the given data. And the output is like this:

![ransac result](https://i.loli.net/2018/05/09/5af2c34cb27d1.jpeg)

 There are too many difficulties.

1. Select 2 points every single iteration randomly
2. Compute a 2D line with these 2 point
3. Count how many points is within the tolerant deviation with this line
4. If the number of the points reach the threshold, compute the line again using all these points based on Least Square. Stop the iteration
5. Else  select 2 points again.
6. If reach the max iteration, using the best result during the iteration

**AdaBoost**

AdaBoost is an iterative training algorithm, the stopping criterion depends on concrete applications.

The output is like this:

![adaboost-1](https://i.loli.net/2018/05/09/5af2c61058695.jpg)

![adaboost-2](https://i.loli.net/2018/05/09/5af2c6a34bb47.jpg)

The main step of AdaBoost:

1. For each iteration, choose a best weak classifier
2. If the error rate of this classifier is above 0.5, stop iteration
3. Else use this classifier to classify objects.Compute its weight among all classifiers.
4. Increase the weight of the object is misclassified, decrease the weight of those classified correctly
5. After all iteration, output the strong classifier combined with all weak classifiers.

In the practice I found that if the iteration is set to large, the weak classifier might be repeated. So when the weak classifier become repeated. I stop the iteration immediately.

