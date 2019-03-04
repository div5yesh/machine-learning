#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'perceptron'))
	print(os.getcwd())
except:
	pass

#%% [markdown]
#  # CMSC 678 — Introduction to Machine Learning
#  ## Assignment 2
#  ## Divyesh Chitroda

#%%
import gzip, pickle, numpy
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
get_ipython().run_line_magic('matplotlib', 'inline')


#%%
plt.rcParams['axes.edgecolor']='black'
plt.rcParams['xtick.color']='black'
plt.rcParams['ytick.color']='black'
plt.rcParams['axes.facecolor']='white'
plt.rcParams['axes.labelcolor']='black'
plt.rcParams['text.color']='black'
plt.rcParams['figure.facecolor']='white'

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
plt.imshow(img, cmap='Greys')
print(trainY[img_index])

#%%
img_index = 218
img = trainX[img_index].reshape(28,28)
plt.imshow(img, cmap='Greys')
print(trainY[img_index])

#%%
img_index = 219
img = trainX[img_index].reshape(28,28)
plt.imshow(img, cmap='Greys')
print(trainY[img_index])

#%%
img_index = 209
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
labels, count = numpy.unique(train_splitY, return_counts = True)
print(count)
plt.bar(labels, count)
plt.xlabel("labels")
plt.ylabel("frequency")
plt.savefig("bctry.png")

#%%
labels, count = numpy.unique(dev_splitY, return_counts = True)
print(count)
plt.bar(labels, count)
plt.xlabel("labels")
plt.ylabel("frequency")
plt.savefig("bctsy.png")

#%%
class model:
    def fit(self, x, y, epoach = 1):
        pass

    def evaluate(self, x, y):
        pass           

    def get_metrics(self, y_, y):
        m = len(y)
        accuracy = (y_ == y).sum()/m
        cm = confusion_matrix(y_, y)
        print(cm)
        # plt.axis('off')
        # plt.table(cellText=[['']*11], colLabels=["sdfdf"], loc='center')
        # plt.table(cellText = cm, rowLabels=numpy.arange(10), colLabels=numpy.arange(10), loc='center')
        precision = numpy.diag(cm) / numpy.sum(cm, axis = 1)
        recall = numpy.diag(cm) / numpy.sum(cm, axis = 0)
        f1 = (2 * precision * recall)/(precision + recall)
        print("Accuracy:",numpy.round(accuracy, 5), 
        "\nPrecision:", numpy.round(precision, 5), 
            "\nRecall:", numpy.round(recall, 5), 
                "\nF1:", numpy.round(f1, 5)) 
        
        return accuracy

    def predict(self, x):
        pass


#%%
class frequent(model):
    def __init__(self):
        self.prediction = -1

    def fit(self, x, y, epoach = 1):
        labels, count = numpy.unique(y, return_counts = True)
        print("Label Counts:", count)
        self.prediction = numpy.argmax(count)
        print("Frequent Label:",self.prediction)
    
    def evaluate(self, x, y):
        y_ = numpy.full(y.shape, self.prediction)
        return self.get_metrics(y_, y)

baseline = frequent()
baseline.fit(x=train_splitX, y=train_splitY)
baseline.evaluate(x=dev_splitX, y=dev_splitY)


#%%
class perceptron(model):
    def fit(self, x, y, epoach = 1):
        (m, n) = x.shape
        l = len(labels)
        weights = numpy.zeros((n,l))
        # T = int(epoach * m)
        for t in range(epoach):
            for i in range(m):
                activation = numpy.dot(weights.T, x[i])
                label_ = numpy.argmax(activation)
                label = y[i][0]
                if label_ != label:
                    y_ = numpy.zeros(l)
                    y_[label_] = -1
                    y_[label] = 1
                    
                    k = x[i,numpy.newaxis]
                    y_ = y_[numpy.newaxis,:]

                    weights += numpy.dot(k.T, y_)
                                
        
        self.weights = weights
        # print("W:",weights[49:196,:])

    def evaluate(self, x, y):
        activation = numpy.dot(x, self.weights)
        y_ = numpy.argmax(activation, axis=1)
        y = y.flatten()
        return self.get_metrics(y_, y)


ptrn = perceptron()
ptrn.fit(x=train_splitX, y=train_splitY)
ptrn.evaluate(x=dev_splitX, y=dev_splitY)

#%%
class biasperceptron(perceptron):
    def fit(self, x, y, epoach = 1):
        (m, n) = x.shape
        ones = numpy.ones((m, 1))
        bias = numpy.ones((len(labels), 1))
        feat = numpy.concatenate((x, ones), axis=1)
        super().fit(x=feat, y=y, epoach=epoach)
        # print(feat.shape)

    def evaluate(self, x,y):
        (m, n) = x.shape
        ones = numpy.ones((m, 1))
        feat = numpy.concatenate((x, ones), axis=1)
        return super().evaluate(x=feat, y=y)


bptrn = biasperceptron()
bptrn.fit(x=train_splitX, y=train_splitY)
bptrn.evaluate(x=dev_splitX, y=dev_splitY)

#%%
class avgperceptron(perceptron):
    def fit(self, x, y, epoach = 1):
        (m, n) = x.shape
        l = len(labels)
        surv = numpy.zeros(l)
        weights = numpy.zeros((n,l))
        avgw = numpy.zeros(weights.shape)
        for t in range(epoach):
            for i in range(m):
                activation = numpy.dot(weights.T, x[i])
                label_ = numpy.argmax(activation)
                label = y[i][0]
                if label_ == label:
                    surv[label] = surv[label] + 1
                else:
                    y_ = numpy.zeros(l)
                    y_[label_] = -1
                    y_[label] = 1
                    
                    k = x[i,numpy.newaxis]
                    y_ = y_[numpy.newaxis,:]

                    avgw += surv * weights
                    weights += numpy.dot(k.T, y_)
                    surv[label] = 1
        
        print(surv)
        self.weights = avgw + (weights * surv)


aptrn = avgperceptron()
aptrn.fit(x=train_splitX, y=train_splitY)
aptrn.evaluate(x=dev_splitX, y=dev_splitY)

#%%
evals = {}

#%%
def evalplot(models):
    for mod in models:
        errors = {}
        for e in range(1,1):
            print("Iteration:", e)
            mod.fit(x=train_splitX, y=train_splitY, epoach=e)
            accuracy = mod.evaluate(x=dev_splitX, y=dev_splitY)
            errors[e] = 1 - accuracy
        evals[type(mod)] = errors

evalplot([ptrn, bptrn, aptrn])

#%%    
clr = "rgbc"
c=0
for errors in evals.values():
    plt.plot(errors.keys(), errors.values(), color=clr[c])
    c = c+1

plt.legend(('Standard', 'Biased', 'Averaged'))
plt.xlabel("Epoach")
plt.ylabel("Error")
plt.savefig("errorvsepoach.png")

#%%
# Q3.a: Perceptron Convergence Data Points
# plt.title("Perceptron Convergence Data Points",color="black")
plt.plot( [-2,-3,-1],[-1,1,-3],'ro', color='r',marker='x')
plt.plot( [2,-1,3],[2,3,-1],'ro', color='g')
plt.axhline(0, color='black')
plt.axvline(0,color='black')
plt.axis([-5,5,-5,5])
plt.xlabel("x_1")
plt.ylabel("x_2")
plt.plot([-4,4],[4,-4])

#%%
# Q3.b: Perceptron Non-Convergence Data Points
# plt.title("Perceptron Non-Convergence Data Points",color="black")
plt.plot( [2,-3,-1],[2,1,-3],'ro', color='r',marker='x')
plt.plot( [-2,-1,3],[-1,3,-1],'ro', color='g')
plt.axhline(0, color='black')
plt.axvline(0,color='black')
plt.axis([-5,5,-5,5])
plt.xlabel("x_1")
plt.ylabel("x_2")
plt.plot([-4,4],[4,-4])

#%% [markdown]
#  ## Test Set Evaluation

#%%
baseline = frequent()
baseline.fit(x=trainX, y=trainY)
baseline.evaluate(x=testX, y=testY)

bptrn = biasperceptron()
bptrn.fit(x=trainX, y=trainY,epoach=10)
bptrn.evaluate(x=testX, y=testY)