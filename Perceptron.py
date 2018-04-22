#!/bin/env python

import numpy as np
from matplotlib.colors import ListedColormap
import  matplotlib.pyplot as plt

class Perceptron(object):
  """ Perceptron Classifier

  Parameters
  ----------
  eta: float
    Learning rate (between 0 and 1)
  n_iter: int
    Number of iternations
  X: 
    Passes over the training dataset
  random_state: int
    Random number generator seed for random weight initialization

  Attributes
  ----------
  w_: 1d-array
    Weights after fitting
  errors: list
    Number of misclassifications(updates) in each epoch
  """
  def __init__(self,eta=0.01, n_iter=50, random_state=1):
    self.eta=eta
    self.n_iter=n_iter
    self.random_state = random_state


  def fit2(self, X, y):
    """
      Fit training data, online preceptron algorithm
    Parameters:
    ----------
    X: {array-like}, shape = [n_samples, n_features]
     Training vectors
    y: array-like, shape = [n_samples]
     Target values

    Returns
    -------
    self: object
    """
    rgen = np.random.RandomState(self.random_state)
    self.w_ = np.zeros(X.shape[1]+1)
    self.errors_=[]
   
    for _ in range(self.n_iter):
      errors=0
      for xi, target in zip(X,y): 
        if target*self.predict(xi) <= 0:
          updates=self.eta*target*xi
          self.w_[1:]+=updates
          self.w_[0]+=self.eta*target
          errors+= 1
      self.errors_.append(errors)
    return self


  def fit(self, X, y):
    """
      Fit training data
    Parameters:
    ----------
    X: {array-like}, shape = [n_samples, n_features]
     Training vectors
    y: array-like, shape = [n_samples]
     Target values

    Returns
    -------
    self: object
    """
    rgen = np.random.RandomState(self.random_state)
    self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1+X.shape[1])
    self.errors_=[]

    for _ in range(self.n_iter):
      errors=0
      for xi, target in zip(X,y):
        updates=self.eta*(target-self.predict(xi))
        self.w_[1:]+=updates*xi
        self.w_[0]+=updates
        errors+= int(updates !=0.0)
      self.errors_.append(errors)
    return self


  def net_input(self,X):
    """ Calculate net input"""
    return np.dot(X,self.w_[1:]) + self.w_[0]

  def predict(self,X):
    """ Return class label after unit setp """
    return np.where(self.net_input(X) >=0.0 ,1,-1)    

class Adaline(Perceptron):
  """ inheritance from perceptron class"""
  def __init__(self,eta=0.1,n_iter=50,random_state=1):
    Perceptron.__init__(self,eta,n_iter,random_state)

  def activation(self,X):
    return X
  
  def fit(self,X,y):
    rgen = np.random.RandomState(self.random_state)
    self.w_ = rgen.normal(loc=0.0,scale=0.01,size=X.shape[1]+1)
    self.cost_ = []
 
    for _ in range(self.n_iter):
      net_input = self.net_input(X)
      output = self.activation(net_input)
      errors = (y - output)
      self.w_[1:] += self.eta*X.T.dot(errors)  
      self.w_[0] += self.eta*errors.sum()
      cost = (errors**2).sum() /2.0
      self.cost_.append(cost)
    return self

  def predict(self,X):
    return (self.activation(self.net_input(X)))

class AdalineSGD(Adaline):
  """ Stochastic gradient descent (online) """
  def __init__(self,eta=0.01,n_iter=10,random_state=1,shuffle=True):
    Adaline.__init__(self,eta,n_iter,random_state)
    self.shuffle = shuffle
    self.w_initialized = False
  def fit(self,X,y):
    self._initialize_weights(X.shape[1])
    self.cost_ = []

    for _ in range(self.n_iter):
      if self.shuffle:
        X,y = self._shuffle(X,y)  
      cost =[]
      for xi, target in zip(X,y):
        cost.append(self._update_weights(xi,target)) 
      avg_cost = sum(cost)/len(y)
      self.cost_.append(avg_cost)

    return self

  def partial_fit(self,X,y):
    """ Fit training data without reinitilizing the weights"""
    if not self.w_initialized:
      self._initialize_weights(X.shape[1])
    if y.ravel().shape[0] > 1:
      for xi,target in zip (X,y):
        self._update_weights(xi,target)
    else:
      self._update_weights(X,y)
    return self

  def _shuffle(self,X,y):
    """ Shuffle training data"""
    r = self.rgen.permutation(len(y))
    return X[r], y[r]

  def _initialize_weights(self,m):
    """ Initialize weights to small random numbers"""
    self.rgen = np.random.RandomState(self.random_state)
    self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size = 1+m)
    self.w_initialized = True

  def _update_weights(self,xi,target):
    """ Apply Adline learning rule to update the weights"""
    output = self.activation(self.net_input(xi))
    error = (target - output)
    self.w_[1:] += self.eta*xi.dot(error)
    self.w_[0] += self.eta*error
    cost = 0.5*error**2
    return cost

def plot_decision_regions(X,y,classifier,resolution=0.02):
  # setup marker generator and color map
  markers =('s','x','o','^','v')
  colors = ('red','blue','lightgreen','gray','cyan')
  cmap = ListedColormap(colors[:len(np.unique(y))])

  # plot the decision surface
  x1_min, x1_max = X[:,0].min()-1, X[:,0].max() +1
  x2_min, x2_max = X[:,1].min()-1, X[:,1].max() +1
  
  xx1,xx2 = np.meshgrid(np.arange(x1_min,x1_max,resolution),np.arange(x2_min,x2_max,resolution))
  Z = classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T) 
  Z = Z.reshape(xx1.shape)
  plt.contourf(xx1,xx2,Z,alpha=0.3,cmap=cmap)
  plt.xlim(xx1.min(),xx1.max())
  plt.ylim(xx2.min(),xx2.max())

  # plot calss samples
  for idx, cl in enumerate(np.unique(y)):
    plt.scatter(x=X[y==cl,0],y=X[y==cl,1],alpha=0.8, c=colors[idx],
    marker=markers[idx],
    label=cl,
    edgecolor='black')

 


def main():
  """ main function to test the perceptron algorithm"""
  import pandas as pd
  import matplotlib.pyplot as plt
  import numpy as np
  
  df=pd.read_csv('iris.data')
  df.tail()
  
  # extract setosa and versicolor
  y = df.iloc[0:100,4].values
  y = np.where(y =='Iris-setosa', -1,1)
  #y[60]=-y[60] # flip a sample such that its not linear separable
  # extract sepal length and petal length
  X = df.iloc[0:100,[0,2]].values

  # plt data
  C1=X[y==-1,:]
  C2=X[y==1,:]
  plt.scatter(C1[:,0],C1[:,1],marker='o',color='red',label='setosa')
  plt.scatter(C2[:,0],C2[:,1],marker='x',color='blue',label='versicolor')
  plt.xlabel('sepal length [cm]')
  plt.ylabel('petal length [cm]')
  plt.legend(loc='upper left')
  plt.show()

  ppn = Perceptron(eta=0.1,n_iter=10)
  ppn.fit2(X,y)

  # plot # of updates/errors in each epoch
  plt.plot(ppn.errors_,marker='o')
  plt.xlabel('Epochs')
  plt.ylabel('# of updates')
  plt.show()
  
  # plot decision region
  #plot_decision_regions(X,y,classifier=ppn)
  #plt.xlabel('sepal length [cm]')
 # plt.ylabel('petal length [cm]')
 # plt.legend(loc='upper left')
 # plt.show()

  X_std = np.copy(X)
  X_std[:,0] = (X[:,0]-X[:,0].mean()) /X[:,0].std()
  X_std[:,1] = (X[:,1]-X[:,1].mean()) /X[:,1].std()
  
  
  ada1 = Adaline(n_iter=15,eta=0.01).fit(X_std,y)
  ada2 = Adaline(n_iter=100,eta = 0.0001).fit(X,y)

  plot_decision_regions(X,y,classifier=ada1)
  
  # compare different learning rates of adaline algorithm
  fig,ax = plt.subplots(nrows=1,ncols=2,figsize=(10,4))
  ax[0].plot(range(1,len(ada1.cost_)+1),np.log10(ada1.cost_),marker='o')
  ax[0].set_xlabel('Epochs')
  ax[0].set_ylabel('log(Sum-squared-error')
  ax[0].set_title('Adaline-Learning rate 0.01')

  ax[1].plot(range(1,len(ada2.cost_)+1),np.log10(ada2.cost_),marker='x')
  ax[1].set_xlabel('Epochs')
  ax[1].set_ylabel('log(Sum-squared-error')
  ax[1].set_title('Adaline-Learning rate 0.0001')
  plt.show()


  plot_decision_regions(X,y,classifier=ada2)
  
  # plt.close()
  # use Adaline SGD classifier
  ada = AdalineSGD(n_iter=15,eta=0.01,random_state=1)
  ada.fit(X_std,y)

  plot_decision_regions(X_std,y,classifier=ada)
  plt.title('Adaline - Stochastic Gradient Descent')
  plt.xlabel('sepal length [standardized]')
  plt.ylabel('petal length [standardized]')
  plt.show()

  plt.plot(range(1,len(ada.cost_)+1),ada.cost_,marker='o')
  plt.xlabel('Epochs')
  plt.ylabel('Average Cost')
  plt.show()




if __name__ == '__main__':
  main()
