#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'neural-networks'))
	print(os.getcwd())
except:
	pass
    
#%% [markdown]
#  # CMSC 678 — Introduction to Machine Learning
#  ## Assignment 4
#  ## Divyesh Chitroda

#%%
import gzip, pickle
import numpy as np
import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix, classification_report
# get_ipython().run_line_magic('matplotlib', 'inline')

#%%
with gzip.open('mnist_rowmajor.pkl.gz', 'rb') as data_fh:
   data = pickle.load(data_fh, encoding='latin1')

#%%
plt.rcParams['axes.edgecolor']='black'
plt.rcParams['xtick.color']='black'
plt.rcParams['ytick.color']='black'
plt.rcParams['axes.facecolor']='white'
plt.rcParams['axes.labelcolor']='black'
plt.rcParams['text.color']='black'
plt.rcParams['figure.facecolor']='white'

#%%
trainX = data["images_train"]
trainY = data["labels_train"]

testX = data["images_test"]
testY = data["labels_test"]

#%%
trainY = np.eye(10)[trainY.flatten()]
testY = np.eye(10)[testY.flatten()]

#%%
img_index = 207
img = trainX[img_index].reshape(28,28)
plt.imshow(img, cmap='Greys')
print(trainY[img_index])

#%%
split_idx = int(trainX.shape[0] * 0.7)

train_splitX = trainX[:split_idx]
train_splitY = trainY[:split_idx]
print(train_splitX.shape)

dev_splitX = trainX[split_idx:]
dev_splitY = trainY[split_idx:]
print(dev_splitX.shape)

#%%
plt.plot( [1,0],[0,1],'ro', color='g')
plt.plot( [0,1],[0,1],'ro', color='r',marker='x')
plt.axhline(0, color='black')
plt.axvline(0,color='black')
plt.axis([-1,2,-1,2])
plt.xlabel("x_1")
plt.ylabel("x_2")

#%%
def relu(x):
	return np.maximum(x, 0)

def softmax(x):
	return np.exp(x)/sum(np.exp(x))

def sigmoid(x):
    return 1/(1 + np.exp(-x))

#%%
class Layer:
	def __init__(self, weights, bias):
		self.weights = weights
		self.bias = bias
		self.delta = np.zeros(weights.shape)

	def activation(self, x, func=sigmoid):
		z = np.dot(x, self.weights) #+ self.bias
		return func(z)

#%%
class NeuralNet:
	def __init__(self, batch_size, learning_rate, epochs, loss):
		self.batch_size = batch_size
		self.learning_rate = learning_rate
		self.epochs = epochs
		self.loss = loss
		self.layers = []
		self.act_func = None

	def addLayers(self, layers, func):
		self.act_func = func
		self.L = len(layers) - 1
		for l in range(0, self.L):
			weights = np.random.normal(size=(layers[l],layers[l+1]), scale=0.03)
			bias = np.random.normal(size=(layers[l+1]))
			layer = Layer(weights, bias)
			self.layers.append(layer)

	def gradientDescent(self, x, y):
		activations = []
		activation = x
		for l in range(self.L - 1):
			activation = self.layers[l].activation(activation, self.act_func)
			activations.append(activation)

		activations.append(self.layers[-1].activation(activation, softmax))

		# print(activations[-1][0], y[0])
		# print(activations[0].shape, activations[1].shape)
		# print(self.layers[0].weights.shape, self.layers[1].weights.shape)

		deltas = []
		delta = np.multiply((activations[-1] - y), activations[-1] * (1-activations[-1]))
		deltas.insert(0,delta)
		for l in range(self.L - 2, -1, -1):
			delta = np.multiply(np.dot(deltas[0], self.layers[l+1].weights.T), activations[l] * (1 - activations[l]))
			deltas.insert(0, delta)

		activations.insert(0, x)
		for l in range(self.L):
			self.layers[l].weights -= self.learning_rate * np.dot(activations[l].T, deltas[l])
		
		cost = self.loss(y, activations[-1])
		return cost

	def train(self, x, y):
		batches = int(len(x) / self.batch_size)
		for epoch in range(self.epochs):
			avg_cost = 0
			for i in range(batches):
				start = i * self.batch_size
				end = start + self.batch_size - 1
				batch_x = x[start:end]
				batch_y = y[start:end]
				cost = self.gradientDescent(batch_x, batch_y)
				avg_cost += cost / batches
			print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost))

	def predict(self, x, y):
		activations = []
		activation = x
		for l in range(self.L - 1):
			activation = self.layers[l].activation(activation)
			activations.append(activation)

		activations.append(self.layers[-1].activation(activation))
		accuracy = np.sum(np.argmax(activations[-1], axis=1) == np.argmax(y, axis=1))/len(y)
		print(accuracy)

#%%
learning_rate = 0.5
epochs = 50
batch_size = 100

# cross entropy loss
def loss(y, y_):
	y_ = np.clip(y_, 1e-10, 0.9999999)
	return -np.mean(np.sum(y * np.log(y_), axis=1))

#%%
# single layer - ReLu
model = NeuralNet(batch_size, learning_rate, epochs, loss)
model.addLayers([784,300,10], relu)
model.train(train_splitX, train_splitY)
model.predict(dev_splitX, dev_splitY)

#%%
# 3 layer - ReLu
model = NeuralNet(batch_size, learning_rate, epochs, loss)
model.addLayers([784,400,123,30,10], relu)
model.train(train_splitX, train_splitY)
model.predict(dev_splitX, dev_splitY)

#%%
# single layer - sigmoid
model = NeuralNet(batch_size, learning_rate, epochs, loss)
model.addLayers([784,300,10], sigmoid)
model.train(train_splitX, train_splitY)
model.predict(dev_splitX, dev_splitY)

#%%
# 3 layer - sigmoid
model = NeuralNet(batch_size, learning_rate, epochs, loss)
model.addLayers([784,400,123,30,10], sigmoid)
model.train(train_splitX, train_splitY)
model.predict(dev_splitX, dev_splitY)

#%%
# Test Evaluation
model = NeuralNet(batch_size, learning_rate, epochs, loss)
model.addLayers([784,300,10], relu)
model.train(trainX, trainY)
model.predict(testX, testY)