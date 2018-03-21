import numpy as np
from utils import minibatch_indexes

class Logistic_regression:
    
    def __init__(self, max_iter=10, eps=0.1):
        self.max_iter = max_iter
        self.eps = eps
    
    def log_likelihood(self, X, Y, W): # w0
        return -sum([np.log(1 + np.exp(-(2 * Y[i] - 1) * np.dot(W.T, X[i]))) for i in range(len(X))])
        
    def derivative_log_likelihood(self, X, Y, W): # w0 ?
        d_W = W.copy()
        for i in range(len(W)):
            d_W[i] = -sum([-(2 * Y[j] - 1) * X[j, i] * (1 / (1 + np.exp(-(2 * Y[j] - 1) * np.dot(W.T, X[j])))) for j in range(len(X))])
        return d_W
    
    def minibatch_optimize(self,datax, datay,xinit):
        ### optimize
        x_histo = []
        f_histo = []
        
        batchsize = 32
        batches = minibatch_indexes(datax.shape[0],batchsize, shuffle=True)
        batch_count = batches.shape[0]
       
        grad_histo = []
        dernier = xinit
        for i in range(self.max_iter):
            batch = batches[i % batch_count]
            X = datax[batch]
            Y = datay[batch]
            fonc = lambda x: self.log_likelihood(X, Y, x)
            dfonc = lambda x: self.derivative_log_likelihood(X, Y, x)
            
            x_histo.append(dernier - (self.eps * dfonc(dernier)))
            f_histo.append(fonc(dernier))
            
            grad_histo.append(dfonc(dernier))
            dernier = x_histo[i]
        
        return np.array(x_histo), np.array(f_histo)
        
    def fit(self, datax, datay):
        # pour que on ne doit pas souvenir les indexes
        #columns = {'W':0,'W0':1}
        features = datax.shape[1]
        initial_W = np.zeros(features)
        #initial_W0 = 0
        initial = initial_W
        
        x_histo,f_histo = self.minibatch_optimize(datax, datay, initial)
        optimal_idx = f_histo.argmin()
        self.W = x_histo[optimal_idx]
        #self.W0 = x_histo[optimal_idx][columns['W0']]
        return self
        
    def predict(self,datax):
        assert datax.shape[1] == len(self.W)
        return np.array([np.sign(-np.dot(self.W.T, x)) for x in datax])
        
    def score(self,datax,datay):
        labels = self.predict(datax)
        return np.count_nonzero(self.predict(datax) == datay) / datax.shape[0], labels