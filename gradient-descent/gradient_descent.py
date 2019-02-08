"""
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
        pass


class FunctionF(OptimizableFunction):
    def __init__(self):
        self.datasets = ["./data1.csv", ""]

    def getTheta_0(self, i):
        theta0 = [np.array([0,0], dtype='d'), np.array([-20,35], dtype='d')]
        return theta0[i]

    def loadData(self, i):
        self.data = pd.read_csv(self.datasets[i]).iloc[:,1]

    def getGradient(self, theta):
        h_w0 = self.data[self.data == 0].count()
        h_w1 = self.data[self.data == 1].count()
        grad_w0 = h_w0 + np.exp(theta[0])/(np.exp(theta[0]) + np.exp(theta[1]))
        grad_w1 = h_w1 + np.exp(theta[1])/(np.exp(theta[0]) + np.exp(theta[1]))
        return np.array([grad_w0, grad_w1], dtype='d')

    def getValue(self, theta):
        w_y = list(map(lambda x: theta[self.data[x]], self.data))
        return np.sum(w_y) + np.log(np.exp(theta[0] + theta[1]))


class FunctionG(OptimizableFunction):
    def __init__(self):
        self.datasets = [[1, 100]]

    def getTheta_0(self, i):
        theta0 = [np.array([0,0], dtype='d'), np.array([2,-3], dtype='d')]
        return theta0[i]

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
        self.maxIter = 5000

    def optimize(self, function, theta):
        descent = np.full((2, 1), math.inf)
        i = 0
        while i < self.maxIter and (descent > self.threshold).all():
            value = function.getValue(theta)
            gradient = function.getGradient(theta)
            theta = self.step(theta, gradient, 0)
            tempGradient = function.getGradient(theta)
            descent = abs(tempGradient - gradient)
            # print(value, theta, descent)
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
        eps = 10**-8
        self.grad_sum += (gradeint**2)
        scalingFactor = n/((eps + self.grad_sum)**0.5)
        return theta - (scalingFactor * gradeint)


def main():
    f_w = FunctionF()
    g_z = FunctionG()

    robbin = RobbinsMonro()
    ada = AdaGrad()

    for obj in {f_w, g_z}:
        for dataIdx in {0}:
            for learner in {robbin, ada}:
                for startPoint in {0,1}:
                    obj.loadData(dataIdx)
                    theta_0 = obj.getTheta_0(startPoint)
                    result = learner.optimize(obj, theta_0)
                    print(obj, learner)
                    print(result.theta, result.value, result.iter)

main()