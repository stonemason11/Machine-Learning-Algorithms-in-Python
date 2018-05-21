
# coding: utf-8


import numpy as np
import matplotlib.pyplot as plt

def lin_regplot(X,y,model):
    plt.scatter(X,y,c='steelblue',edgecolor='white',s=70)
    plt.plot(X,model.predict(X),color='black',lw=2)
    return None


# define OLR class
class LinearRegressionGD(object):
    """ Ordinary Linear Regression , Gradient Descent"""
    def __init__(self,eta=0.001,n_iter = 20):
        self.eta = eta
        self.n_iter = 20
        
    def fit(self,X,y):
        self.w_ = np.zeros(1+X.shape[1])
        self.cost_ = []
        
        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            self.w_[1:]+= self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() /2.0
            self.cost_.append(cost)
            
        return self
    
    def net_input(self,X):
        return np.dot(X,self.w_[1:]) + self.w_[0]
    
    def predict(self,X):
        return self.net_input(X)


def main():

	# import house data
	import pandas as pd

	df = pd.read_csv('housing.data',header=None,sep='\s+')
	df.columns = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV'];


	# Visualize using exploratory data analysis (EDA) toolbox
	# Create scatterplot matrix using pairplot from seaborn library

	import seaborn as sns
	cols = ['LSTAT','INDUS','NOX','RM','MEDV']
	sns.pairplot(df,size=2.5)
	plt.tight_layout()
	plt.show()


	# calculate Pearson's correlation coefficient using numpy corrcoef
	# and visualize it using seaborn heatmap
	import numpy as np
	cm = np.corrcoef(df[cols].values.T)
	sns.set(font_scale=1.5)
	hm = sns.heatmap(cm,cbar=True,
		        annot=True,
		        square=True,
		        fmt='.2f',
		        annot_kws={'size':15},
		        yticklabels= cols,
		        xticklabels= cols)
	plt.show()


	X = df[['RM']].values
	y = df['MEDV'].values



	from sklearn.preprocessing import StandardScaler
	sc_x = StandardScaler()
	sc_y = StandardScaler()
	X_std = sc_x.fit_transform(X)
	y_std = sc_y.fit_transform(y[:,np.newaxis]).flatten()
	lr = LinearRegressionGD()
	lr.fit(X_std,y_std)


	sns.reset_orig() # resets matplotlib style



	plt.plot(range(1,lr.n_iter+1),lr.cost_)
	plt.ylabel('SSE')
	plt.xlabel('Epoch')
	plt.show()

	lin_regplot(X_std,y_std,lr)
	plt.xlabel('Average number of rooms [RM] (standardized)')
	plt.ylabel('Price in $1000 [MEDV] (standardized)')
	plt.show()

if __name__ =='__main__':
	main()



