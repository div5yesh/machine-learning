"""
Divyesh Chitroda

References:
1. Gradient Descent Skeleton Header, https://www.csee.umbc.edu/courses/graduate/678/spring19/materials/a1_q4_possible_header.hpp
"""

import pandas as pd
import numpy as np
import math

np.seterr(all='ignore')

class OptimizableFunction:

    def __init__(self):
        self.datasets = []
        self.data = []

    def getValue(self, point):
        pass

    def getGradient(self, point):
        pass

    def loadData(self, i):
        self.data = self.datasets[i]

    def getTheta_0(self, i):
        theta0 = np.random.rand(2,2)
        return theta0[i]


class FunctionF(OptimizableFunction):
    def __init__(self):
        data1 = pd.read_csv('./data1.csv').iloc[:,1]
        data2 = pd.read_csv('./data2.csv').iloc[:,1]
        self.datasets = [data1, data2]

    def getGradient(self, theta):
        h_w0 = self.data[self.data == 0].count()
        h_w1 = self.data[self.data == 1].count()
        grad_w0 = -h_w0 + np.exp(theta[0])/(np.exp(theta[0]) + np.exp(theta[1]))
        grad_w1 = -h_w1 + np.exp(theta[1])/(np.exp(theta[0]) + np.exp(theta[1]))
        return np.array([grad_w0, grad_w1], dtype='d')

    def getValue(self, theta):
        w_y = list(map(lambda x: theta[self.data[x]], self.data))
        return -np.sum(w_y) + np.log(np.exp(theta[0] + theta[1]))


class FunctionG(OptimizableFunction):
    def __init__(self):
        self.datasets = [[1, 100]]
        a = int(input("Enter value of a:"))
        b = int(input("Enter value of b:"))
        self.datasets.append([a, b])

    def getGradient(self, theta):
        k =  -4 * self.data[1] * theta[0] * (theta[1] - theta[0] ** 2)
        grad_zi0 = -2 * (self.data[0] - theta[0]) + k
        grad_zi1 = 2 * self.data[1] * (theta[1] - theta[0] ** 2)
        return np.array([grad_zi0, grad_zi1], dtype='d')

    def getValue(self, theta):
        return (self.data[0] - theta[0])**2 + self.data[1] * (theta[1] - theta[0]**2)**2


class OptimizationResult():

    def __init__(self, theta, value, iter):
        self.theta = theta
        self.value = value
        self.iter = iter

class GradientDescent:

    def __init__(self):
        self.threshold = 0.00001
        self.maxIter = 10000
    def optimize(self, function, theta):
        descent = np.full((2, 1), math.inf)
        i = 0
        while i < self.maxIter:# and (descent > self.threshold).all():
            value = function.getValue(theta)
            gradient = function.getGradient(theta)
            theta = self.step(theta, gradient, i)
            tempGradient = function.getGradient(theta)
            descent = abs(tempGradient - gradient)
            i += 1
    
        result = OptimizationResult(theta, value, i)
        return result

    def step(self, theta, gradeint, iteration):
        return []


class RobbinsMonro(GradientDescent):

    def step(self, theta, gradeint, iteration):
        scalingFactor = (iteration + 500) ** (-1)
        return theta - (scalingFactor * gradeint)

class AdaGrad(GradientDescent):

    def __init__(self):
        super().__init__()
        self.grad_sum = np.zeros((2))

    def step(self, theta, gradeint, iteration):
        n = 0.1
        eps = 10 ** (-8)
        self.grad_sum += (gradeint) ** 2
        scalingFactor = n/((eps + self.grad_sum)**0.5)
        return theta - (scalingFactor * gradeint)


def main():
    f_w = FunctionF()
    g_z = FunctionG()

    robbin = RobbinsMonro()
    ada = AdaGrad()

    for obj in {f_w, g_z}:
        print("Objective:",obj)
        for dataIdx in {0, 1}:
            print("Data:",dataIdx)
            for learner in {robbin, ada}:
                print("Learner:", learner)
                for startPoint in {0, 1}:
                    obj.loadData(dataIdx)
                    theta_0 = obj.getTheta_0(startPoint)
                    print("Start:",theta_0)
                    result = learner.optimize(obj, theta_0)
                    print("Theta*:", result.theta, ", Value at theta*:", result.value)

"""
Entry Point: main function
"""
main()