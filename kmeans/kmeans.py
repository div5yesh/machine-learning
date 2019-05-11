#%%
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'kmeans'))
	print(os.getcwd())
except:
	pass

#%%
import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos, radians, pi, sqrt

#%%
plt.rcParams['axes.edgecolor']='black'
plt.rcParams['xtick.color']='black'
plt.rcParams['ytick.color']='black'
plt.rcParams['axes.facecolor']='white'
plt.rcParams['axes.labelcolor']='black'
plt.rcParams['text.color']='black'
plt.rcParams['figure.facecolor']='white'

#%%
points = []
fr = open("2d-kmeans-input.ssv", "r")
data = fr.read()
data = data.split("\n")
for i in range(len(data)):
    if data[i]:
        points.append(np.array(data[i].split(" ")).astype(float))

points = np.array(points)

#%%
def initUniform(data, k):
    (n, d) = data.shape
    centroids = np.random.uniform(low=np.min(data), high=np.max(data), size=(k,d))
    plt.plot(data[:,0], data[:,1], 'ro', marker='.', alpha=0.4)
    plt.plot(centroids[:,0], centroids[:,1], 'ro', marker='x', color='b')
    plt.axis([0,1,0,1])
    plt.show()
    return centroids

def initLlloyd(data, k):
    (n, d) = data.shape
    centroids = data[np.random.randint(low=0, high=n, size=k)]
    plt.plot(data[:,0], data[:,1], 'ro', marker='.', alpha=0.4)
    plt.plot(centroids[:,0], centroids[:,1], 'ro', marker='x', color='b')
    plt.axis([0,1,0,1])
    plt.show()
    return centroids

initUniform(points, 3)
initLlloyd(points, 3)

#%%
def kmeans(data, centroids, epoch):
    (n,d) = data.shape
    K = len(centroids)
    clusters = np.empty((n))
    for t in range(epoch):
        for i in range(n):
            distance = np.sqrt(np.sum((centroids - data[i])**2, axis=1))
            clusters[i] = np.argmin(distance)

        colors = "rgbymck"
        for k in range(K):
            cluster = data[clusters == k]
            plt.plot(cluster[:,0], cluster[:,1], 'ro', marker='.', color=colors[k], alpha=0.4)
            plt.plot(centroids[k,0], centroids[k,1], 'ro', marker='x', color=colors[k])
            plt.axis([0,1,0,1])
            centroids[k] = np.mean(cluster, axis=0)
        
        plt.show()

#%%
K = 2
centroids = initLlloyd(points, K)
kmeans(points, centroids, 10)

#%%
K = 4
centroids = initLlloyd(points, K)
kmeans(points, centroids, 10)

#%%
n = 150
K = 3
distr = np.random.normal(loc=[0.4, 0.6], scale=[0.1, 0.1], size=(n, 2))
centroids = initUniform(distr, K)
kmeans(distr, centroids, 10)

#%%
n = 500
K = 3
distr_uni = np.random.normal(loc=[0.4, 0.6], scale=[0.1, 0.1], size=(n, 2))
centroids = initUniform(distr_uni, K)
kmeans(distr_uni, centroids, 10)

#%%
n = 500
K = 3

def ring():
    angle = np.random.uniform(0, 2 * pi)
    distance = sqrt(np.random.uniform(0.001, 0.002)) * 10
    return np.exp([distance * cos(angle) - 0.5, distance * sin(angle) - 0.5])

distr_ring = np.zeros((n,2))
for i in range(n):
    distr_ring[i] = ring()

centroids = initUniform(distr_ring, K)
kmeans(distr_ring, centroids, 10)
#%%
