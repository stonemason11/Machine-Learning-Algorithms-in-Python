
# coding: utf-8

# # Orthogonal Matching Pursuit (OMP)
# reference: Sergios' Machine Learning book, Chapt.10
import numpy as np


def corr(x,y):
    return abs(x.dot(y))/np.sqrt((x**2).sum())

class OMP(object):
    """ Orthogonal Matching Pursuit (OMP) algorithm class"""
    def __init__(self, err_tol=0.001, random_state=0):
        self.err_tol = err_tol
        self.random_state = random_state

    def estimate(self,X,y):
        L = X.shape[1]
        theta = np.zeros(L)
        error = y
        S = [] # the support
        ii = 0

        while np.linalg.norm(error) > self.err_tol:
            # find the column has maximum correlation with the error
            corrs = [ corr(x,error) for x in X.T]
            ind = np.argmax(corrs)
            S.append(ind)
            #print(S)
            X_active = X[:,S]
            # LS estimate using active support
            theta_tilde = np.linalg.inv(X_active.T.dot(X_active)).dot(X_active.T).dot(y)
    
            # insert estimated theta into the correct location
            theta[S] = theta_tilde
    
            # update the error vector
            error = y-X.dot(theta)
            ii+=1
        self.theta = theta
        self.n_iters = ii
        self.errors = np.linalg.norm(error)
        return self

def main():
    
    L = 20 # dimension of the unknown vector w 
    k0 = 3 # assume w is k0-sparse
    w = np.zeros(L)
    rgn = np.random.RandomState(0)
    N_max = 20 # max number of sensing samples
   
    # randomly choose k0 entries, and randomly assign values
    w[rgn.randint(0,L,k0)] = rgn.normal(loc=0.0,scale=1.0,size=k0)

    omp = OMP(err_tol=0.001)
    errors = [] # trace the errors 
    for N in range(1,N_max):
        X = rgn.normal(loc=0.0,scale=1.0,size=(N,L))
        y = X.dot(w)
        omp.estimate(X,y)
        w_hat = omp.theta
        errors.append(np.linalg.norm(w-w_hat))
        

    # visualize the errors
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(np.arange(1,N_max), errors,marker='o')
    plt.ylabel('l2-norm error')
    plt.xlabel('# of samples')
    plt.title('Performance of the OMP algorithm')
    plt.show()

if __name__=='__main__':
    main()

