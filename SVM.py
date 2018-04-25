#!/bin/env python

from scipy import optimize
import numpy as np

class SVM(object):
  """ Support Vector Machine Class, by Wensheng Sun"""
  def __init__(self,C=100,n_iter=50,random_state=1,opt='linear'):
    self.random_state=random_state
    self.opt = opt
    self.C=C # inverse of the regularization weight

  def kernal_matrix(self,X,Y):
    """ Need to impelment more kernal options"""
    if self.opt=='linear':
      return np.dot(X,Y.T)
    else:
      return np.dot(X,Y.T)

  def loss(self,x):
    """ return the negative loss function (dual form)
     inorder to use the minimizer"""
    K = self.kernal_matrix(self.X,self.X)
    zz = x*self.y
    return -np.dot(np.ones(self.y.shape),x.T)+0.5*np.dot(np.dot(zz.T,K),zz)

  def jac(self,x):
    """ the Jacobian matrix of the loss function 
    with respect to lambda_n s """
    K = self.kernal_matrix(self.X,self.X)
    zz = x*self.y
    return np.ones(self.y.shape) - np.dot(zz.T,K) 

  def fit(self,X,y):
    self.X = X
    self.y = y
    cons = {'type':'eq','fun':lambda x: np.dot(x,self.y),'jac':lambda x: self.y }
    opt = {'disp':False}
    rgen = np.random.RandomState(self.random_state)
    x0 = rgen.normal(loc=0.0,scale=0.5,size=len(self.y))
    bnds = tuple((0,C) for C in self.C*np.ones(self.y.shape))
    # call minimizer to solve QP problem
    QP_res=optimize.minimize(self.loss,x0,jac=self.jac,constraints=cons,method='SLSQP',options=opt,bounds=bnds)

    # collect results
    lambdas_all=QP_res.x
    # pick out the support vectors
    idx = lambdas_all !=0
    self.lambdas_s = lambdas_all[idx]
    self.ys = self.y[idx]
    self.Xs = self.X[idx,:] 
    self.theta_hat = np.dot(self.lambdas_s*self.ys,self.Xs)
    self.theta0 = (self.ys-np.dot(self.Xs,self.theta_hat.T)).sum()/len(idx)
    return self

  def predict(self,Xtest):
    """ predict the results"""
    k = self.kernal_matrix(Xtest,self.Xs)
    return np.dot(k,self.lambdas_s*self.ys) + self.theta0

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

  # instantiate the logistic regression classifier
  svm = SVM(C=1)
  svm.fit(X_std,y)
  # plot the decision regions
  plt.figure()
  plot_decision_regions(X_std,y,classifier=svm)
  plt.title('Batched Logistic Regression')
  plt.xlabel('sepal length [standardized]')
  plt.ylabel('petal length [standardized]')
  plt.show()

if __name__=='__main__':
  main()

