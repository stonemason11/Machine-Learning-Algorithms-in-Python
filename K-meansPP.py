
# coding: utf-8

# # K-Means

# define the K-means class
import numpy as np

def dist(X,x):
    """Calculate the Euclidean distance between 
    a sample and the k centroids"""
    return [np.linalg.norm(x-x1) for x1 in X]

class KMeansPP(object):
    """The K-means ++ algorithm,
    which choose the centroids smartly.
    Reference: Python_Machine_learning_with_Scikit_and_TensorFlow
    Implementation: Wensheng Sun"""
    def __init__(self,k=1,random_state=5,tol=1,N_max=100):
        self.k = k
        self.random_state=5
        self.tol=tol
        self.N_max=N_max
        
    def fit(self,X):
        rgen = np.random.RandomState(self.random_state)
        N,Nfeatures = X.shape
        
        
        M = []
        pmf = np.ones(N)/N
        # smart start of KMeans++
        for _ in range(0,self.k):
            ind = rgen.choice(np.arange(0,len(pmf)),p=pmf,size=1)
            M.append(ind)
            #print(np.array(M))
            ind_rest = [x for x in range(0,N) if x not in M]
            D=np.array([ np.min(dist(X[M,:],x)) for x in X[ind_rest,:]])
            #print(D)
            pmf = D/D.sum()
        
        centroids = X[M,:]
        # label the samples
        D=[ dist(centroids,x) for x in X]
        y_old=np.argmin(D,axis=1)
        
        
        ii = 0
        err = np.inf
        while ii< self.N_max and err > self.tol:
            
            #print(centroids)
            # update the centroids
            centroids = np.array([ np.mean(X[y_old==lb,:],axis=0) for lb in np.unique(y_old)])
            D=np.array([ dist(centroids,x) for x in X])
            y_new=np.argmin(D,axis=1)

            err = (y_old!=y_new).sum()
            y_old=y_new
            
            ii+=1
            
        self.y = y_old
        self.cluster_centers_=centroids
        return self

def main():
    # import data from sklearn 
    from sklearn.datasets import make_blobs
    X,y = make_blobs(n_samples=150,n_features=2,centers=3,cluster_std=0.5,shuffle=True,random_state=1)

    # visualize the data 
    import matplotlib.pyplot as plt

    plt.scatter(X[:,0],X[:,1],c='black',marker='o',s=50)
    plt.grid()
    plt.show()

    km = KMeansPP(k=3,tol=0)
    km.fit(X)
    y_km = km.y
    #print(km.cluster_centers_)

    # visualize the results
    plt.scatter(X[y_km == 0, 0],
                X[y_km == 0, 1],
                s=50, c='lightgreen',
                marker='s', edgecolor='black',
                label='cluster 1')
    plt.scatter(X[y_km == 1, 0],
                X[y_km == 1, 1],
                s=50, c='orange',
                marker='o', edgecolor='black',
                label='cluster 2')
    plt.scatter(X[y_km == 2, 0],
                X[y_km == 2, 1],
                s=50, c='lightblue',
                marker='v', edgecolor='black',
                label='cluster 3')
    plt.scatter(km.cluster_centers_[:,0],
                km.cluster_centers_[:,1],
                s=250, marker='*',
                c='red', edgecolor='black',
                label='centroids')
    plt.legend(scatterpoints=1)
    plt.grid()
    plt.show()

if __name__=='__main__':
    main()
