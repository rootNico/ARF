import numpy as np

class Logistic_regression:
    
    def __init__(self, max_iter=10, eps=0.1):
        self.max_iter = max_iter
        self.eps = eps
        
    def estimate(self,x, w):
        assert len(x) == len(w)
        # w_x = w0 + sum([x[i]*w[i] for i in range(len(x))])
        #proba = 1 / (1 + np.exp(-w_x)
        proba = np.dot(x.T,w) #w0
        return np.log(proba / (1 - proba)) > 0
    
    def log_likelihood(self, X, Y, W): # w0
        return -sum([np.log(1 + np.exp(-(2 * Y[i] - 1) * np.dot(W.T, X[i]))) for i in range(len(X))])
        
    def derivative_log_likelihood(self, X, Y, W): # w0 ?
        d_W = W.copy()
        for i in range(len(W)):
            d_W[i] = -sum([-(2 * Y[j] - 1) * X[j, i] * (1 / (1 + np.exp(-(2 * Y[j] - 1) * np.dot(W.T, X[j])))) for j in range(len(X))])
        return d_W
    
    #def mse(self, x, y, w, w0):
    #    N = len(x)
    #    error = sum([(y[i] - self.estimate(x[i, :], w, w0))**2 for i in range(N)])
    #    return error / N

    #def mse_derive(self, x, y, w, w0):
    #    N = len(x)
    #    derive_w = np.zeros(len(w))
    #    for w_ix in range(len(w)):
    #        derive_w[w_ix] = sum([-2 * x[i, w_ix] *(y[i] - self.estimate([x[i,w_ix]], [w[w_ix]], w0)) for i in range(N)]) / N
    #    derive_w0 = sum([-2 * (y[i] - self.estimate(x[i], w, w0)) for i in range(N)]) / N
    #    return np.array([derive_w, derive_w0])
    
    def optimize(self,fonc,dfonc,xinit,eps,max_iter):
        x_histo = []
        f_histo = []
        grad_histo = []
        dernier = xinit
        for i in range(max_iter):
            x_histo.append(dernier - (eps * dfonc(dernier)))
            f_histo.append(fonc(dernier))
            grad_histo.append(dfonc(dernier))
            dernier = x_histo[i]
        return (np.array(x_histo), np.array(f_histo), np.array(grad_histo))
        
    def fit(self, datax, datay):
        # pour que on ne doit pas souvenir les indexes
        #columns = {'W':0,'W0':1}
        features = datax.shape[1]
        initial_W = np.zeros(features)
        #initial_W0 = 0
        initial = initial_W
        log = lambda x : self.log_likelihood(datax, datay, x)
        log_derive = lambda x : self.derivative_log_likelihood(datax, datay, x)
        
        x_histo,f_histo,_ = self.optimize(log, log_derive, initial, self.eps, self.max_iter)
        optimal_idx = f_histo.argmin()
        self.W = x_histo[optimal_idx]
        #self.W0 = x_histo[optimal_idx][columns['W0']]
        return self
        
    def predict(self,datax):
        assert datax.shape[1] == len(self.W)
        return np.array([self.estimate(x, self.W) for x in datax])
        
    def score(self,datax,datay):
        labels = self.predict(datax)
        return np.count_nonzero(self.predict(datax) == datay) / datax.shape[0], labels