
import numpy as np

class LDA(object):
  """ The Fisher's Linear Discriminant Analysis classifier
    Currently only for two-class problems"""
  def __init__(self):
    pass
  
  def fit(self,X,y):
    N = len(y)
    cls = np.unique(y)
    X1,X2 = X[y==cls[0],:], X[y==cls[1],:]
    p1,p2 = X1.shape[0]/N,  X2.shape[0]/N
    mu1 , mu2= np.mean(X1,axis=0), np.mean(X2,axis=0)
    Q1 = 1/(X1.shape[0]-1)*np.dot((X1-mu1).T,X1-mu1)
    Q2 = 1/(X2.shape[0]-1)*np.dot((X2-mu2).T,X2-mu2)
    Q = p1*Q1 + p2*Q2
    self.theta = np.dot(np.linalg.inv(Q),mu1-mu2)
    b1 = np.mean(np.dot(self.theta,X1.T))
    b2 = np.mean(np.dot(self.theta,X2.T))
    self.b = 0.5*(b2+b1)
    self.cls=cls
    return self

  def predict(self,X):
    return np.array([np.where(np.dot(self.theta,x) +self.b > 0, self.cls[0],self.cls[1]) for x in X])

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
  lda = LDA()
  lda.fit(X_std,y)
  # plot the decision regions
  plt.figure()
  plot_decision_regions(X_std,y,classifier=lda)
  plt.title(lda.__class__.__name__)
  plt.xlabel('sepal length [standardized]')
  plt.ylabel('petal length [standardized]')
  plt.show()

if __name__=='__main__':
  main()

