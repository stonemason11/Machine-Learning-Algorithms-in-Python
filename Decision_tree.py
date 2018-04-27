#!/bin/env python

import numpy as np

class Decision_Tree(object):
  """ Decision Tree, by Wensheng Sun"""
  def __init__(self,max_depth=3,random_state=1,impurity_fun='entropy',debug=False):
    self.random_state=random_state
    self.impurity_fun = impurity_fun
    self.max_depth = max_depth
    self.debug = debug

  def get_imp(self,p):
    """ Calculate the impurity index
    p: is the probability vector, the sum of which equal to 1 """
    if self.debug:
      print(p)
    return {
         'entropy': (-p*np.log2(p)).sum(),
         'error' : 1-np.max(p),
         'gini': (p*(1-p)).sum()
        }.get(self.impurity_fun)
  def to_freq(self,y):
    """convert to frequencies for each class"""
    p =np.array([len(y[y==cl])/len(y) for cl in np.unique(y)])
    if self.debug:
        print('from function to_freq',p)
    return p

  def split_data(self,D,split):
    """split the data set"""
    X,y = D
    idx, threshold = split
    idx=int(idx)
    Xl=X[X[:,idx] <= threshold,:]
    yl=y[X[:,idx] <= threshold]
    Xr=X[X[:,idx] > threshold,:]
    yr=y[X[:,idx] > threshold]

    Dl = (Xl,yl)
    Dr = (Xr,yr)
    return (Dl,Dr)
  
  def info_gain(self,D,split):
    """ the information gain of a given split
    split: (idx_feature, threshold) is a tuple """
    Dl,Dr = self.split_data(D,split)
    Nl = len(Dl[1])
    Nr = len(Dr[1])
    Np = len(D[1])
    Il = self.get_imp(self.to_freq(Dl[1]))
    Ir = self.get_imp(self.to_freq(Dr[1]))
    Ip = self.get_imp(self.to_freq(D[1]))
    return Ip - Nl/Np * Il - Nr/Np*Ir

  def get_thresholds_from_vec(self,x):
    """ thresholds are the middle points of consecutive unique points """
    return 0.5*np.diff(np.unique(x)) + np.unique(x)[:-1]

  def get_threshold_from_mat(self,X):
    """ call get_thresholds from vec, and zip all the results """
    ns,nf = X.shape
    tmp=[self.get_thresholds_from_vec(X[:,i]) for i in range(nf)]
    ids = np.concatenate([(i*np.ones(len(item))) for i, item in zip(np.arange(nf),tmp)])
    return [(idx,t) for idx, t in zip(ids,np.concatenate(tmp))]   

  def label_it(self,D):
    """ return the most likely class """
    X,y = D
    v,c = np.unique(y,return_counts=True)
    ind = np.argmax(c)
    return v[ind]

  def fit(self,X,y):
    """ train the decision tree """
    self.cls = np.unique(y) # number of classes
    self.ns,self.nf = X.shape # number of samples
    sps = self.get_threshold_from_mat(X)
    IGs = np.array([self.info_gain((X,y),sp) for sp in sps])
    # print(sps)
    if self.debug:
      print('the (dim, threshold) pairs:',sps)
      print('Obtained info gains: ',IGs)
    # return the first maximum information gain
    cond = IGs==np.max(IGs)
    self.IG_max = IGs[cond]
    tmp = np.array(sps)[cond]
    self.split = (int(tmp[0,0]),tmp[0,1])
    #print(self.split[0])
    # calculate the class label 
    Dl,Dr = self.split_data((X,y),self.split)
    self.children = {'left D':Dl, 'right D':Dr, 'left c': self.label_it(Dl),'right c': self.label_it(Dr)}
    #print(self.children)
    return self

  def predict(self,X):
    """ predict the results"""
    return np.where(X[:,self.split[0]] <= self.split[1], self.children['left c']
    ,self.children['right c'])

def main():
  import pandas as pd
  import matplotlib.pyplot as plt
  import numpy as np
  from Perceptron import plot_decision_regions

  df=pd.read_csv('iris.data')
  df.tail()

  # extract setosa and versicolor
  y = df.iloc[0:100,4].values
  y = np.where(y =='Iris-setosa', 0,1)
  # extract sepal length and petal length
  X = df.iloc[0:100,[0,2]].values

  # standardize the features
  X_std = np.copy(X)
  X_std[:,0] = (X[:,0]-X[:,0].mean()) /X[:,0].std()
  X_std[:,1] = (X[:,1]-X[:,1].mean()) /X[:,1].std()
  plt.figure()
  plt.scatter(X[y==0,0],X[y==0,1],marker='o',color='red',label='0')
  plt.scatter(X[y==1,0],X[y==1,1],marker='x',color='green',label='1')
  plt.show()
  
  # instantiate the logistic regression classifier
  dc = Decision_Tree()
  dc.fit(X,y)
  # plot the decision regions
  plt.figure()
  plot_decision_regions(X,y,classifier=dc)
  plt.title('Decision Tree')
  plt.xlabel('sepal length [standardized]')
  plt.ylabel('petal length [standardized]')
  plt.show()

if __name__=='__main__':
  main()

