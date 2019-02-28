#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'perceptron'))
	print(os.getcwd())
except:
	pass


#%%
import gzip, pickle, numpy
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
get_ipython().run_line_magic('matplotlib', 'inline')


#%%
with gzip.open('mnist_rowmajor.pkl.gz', 'rb') as data_fh:
   data = pickle.load(data_fh, encoding='latin1')


#%%
trainX = data["images_train"]
trainY = data["labels_train"]

testX = data["images_test"]
testY = data["labels_test"]


#%%
img_index = 207
img = trainX[img_index].reshape(28,28)
plt.imshow(img, cmap='Greys');
print(trainY[img_index])


#%%
split_idx = int(trainX.shape[0] * 0.8)

train_splitX = trainX[:split_idx]
train_splitY = trainY[:split_idx]
print(train_splitX.shape)

dev_splitX = trainX[split_idx:]
dev_splitY = trainY[split_idx:]
print(dev_splitX.shape)


#%%
class model:
    def fit(self, x, y):
        pass
    def evaluate(self, x, y):
        pass
    def predict(self, x):
        pass

    def confusion_matrix(self, y_, y):
        cm = confusion_matrix(y_, y)
        return cm


#%%
class frequent(model):
    def __init__(self):
        self.prediction = -1

    def fit(self, x, y):
        labels, count = numpy.unique(y, return_counts = True)
#         print(labels, count)
        self.prediction = numpy.argmax(count)
#         print(self.prediction)
    
    def evaluate(self, x, y):
        total = y.shape[0]
        y_ = numpy.full(y.shape, self.prediction)
        tp = (y_ == y).sum()
        tn = 0
        fp = (y_ != y).sum()
        fn = 0
        self.accuracy = (tp + tn)/total
        self.precision = tp/(tp + fp)
        self.recall = tp/(tp + fn)
        print(self.confusion_matrix(y_, y))
        print("accuracy:",self.accuracy, ", precision:", self.precision, ", recall:", self.recall)

baseline = frequent()
baseline.fit(x=train_splitX, y=train_splitY)
baseline.evaluate(x=dev_splitX, y=dev_splitY)


#%%
class perceptron(model):
    def fit(self, x, y):
        (m, n) = x.shape
        weights = numpy.zeros((n,10))
        
        for t in range(1):
            for i in range(m):
                activation = numpy.dot(weights.T, x[i])
                label_ = numpy.argmax(activation)
                label = y[i][0]
                if label_ != label:
                    y_ = numpy.zeros(10)
                    y_[label_] = -1
                    y_[label] = 1
                    
                    k = x[i,numpy.newaxis]
                    y_ = y_[numpy.newaxis,:]

                    weights += numpy.dot(k.T, y_)
                            
        
        self.weights = weights
        # print("W:",weights[49:196,:])

    def evaluate(self, x, y):
        (m,n) = x.shape
        activation = numpy.dot(x, self.weights)
        y_ = numpy.argmax(activation, axis=1)
        y = y.flatten()
        tp = (y_ == y).sum()
        fp = (y_ != y).sum()
        self.accuracy = tp/m
        self.precision = 0
        self.recall = 0
        print(self.confusion_matrix(y_, y))
        print("accuracy:",self.accuracy, ", precision:", self.precision, ", recall:", self.recall)        

ptrn = perceptron()
ptrn.fit(x=train_splitX, y=train_splitY)
ptrn.evaluate(x=dev_splitX, y=dev_splitY)

#%%
class biasperceptron(perceptron):
    def fit(self, x, y):
        (m, n) = x.shape
        ones = numpy.ones((m, 1))
        bias = numpy.ones((10, 1))
        feat = numpy.concatenate((x, ones), axis=1)
        super().fit(x=feat, y=y)
        # print(feat.shape)

    def evaluate(self, x,y):
        (m, n) = x.shape
        ones = numpy.ones((m, 1))
        feat = numpy.concatenate((x, ones), axis=1)
        super().evaluate(x=feat, y=y)


bptrn = biasperceptron()
bptrn.fit(x=train_splitX, y=train_splitY)
bptrn.evaluate(x=dev_splitX, y=dev_splitY)

#%%
class avgperceptron(perceptron):
    def fit(self, x, y):
        pass


aptrn = avgperceptron()
aptrn.fit(x=train_splitX, y=train_splitY)


#%%
plt.rcParams['axes.edgecolor']='black'
plt.rcParams['xtick.color']='black'
plt.rcParams['ytick.color']='black'
plt.rcParams['axes.facecolor']='white'
plt.rcParams['figure.facecolor']='white'

#%%
# Q3.a: Perceptron Convergence Data Points
# plt.title("Perceptron Convergence Data Points",color="black")
plt.plot( [-2,-3,-1],[-1,1,-3],'ro', color='r',marker='x')
plt.plot( [2,-1,3],[2,3,-1],'ro', color='g')
plt.axhline(0, color='black')
plt.axvline(0,color='black')
plt.axis([-5,5,-5,5])
plt.plot([-4,4],[4,-4])

#%%
# Q3.b: Perceptron Non-Convergence Data Points
# plt.title("Perceptron Non-Convergence Data Points",color="black")
plt.plot( [2,-3,-1],[2,1,-3],'ro', color='r',marker='x')
plt.plot( [-2,-1,3],[-1,3,-1],'ro', color='g')
plt.axhline(0, color='black')
plt.axvline(0,color='black')
plt.axis([-5,5,-5,5])
plt.plot([-4,4],[4,-4])

#%%
labels, count = numpy.unique(train_splitY, return_counts = True)
print(count)
plt.bar(labels, count)

#%%
