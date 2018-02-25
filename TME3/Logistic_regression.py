import numpy as np

class Logistic_regression:
    
    def __init__(self, max_iter=10, eps=0.1):
        self.max_iter = max_iter
        self.eps = eps
        
    def estimate(self,x, w, w0):
        assert len(x) == len(w)
        proba = w0 + sum([x[i]*w[i] for i in range(len(x))])
        return np.log(proba / (1 - proba)) > 0
    
    def mse(self, x, y, w, w0):
        N = len(x)
        error = sum([(y[i] - self.estimate(x[i, :], w, w0))**2 for i in range(N)])
        return error / N

    def mse_derive(self, x, y, w, w0):
        N = len(x)
        derive_w = np.zeros(len(w))
        for w_ix in range(len(w)):
            derive_w[w_ix] = sum([-2 * x[i, w_ix] *(y[i] - self.estimate([x[i,w_ix]], [w[w_ix]], w0)) for i in range(N)]) / N
        derive_w0 = sum([-2 * (y[i] - self.estimate(x[i], w, w0)) for i in range(N)]) / N
        return np.array([derive_w, derive_w0])
    
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
        columns = {'W':0,'W0':1}
        features = datax.shape[1]
        initial_W = np.zeros(features)
        initial_W0 = 0
        initial = np.array([initial_W,initial_W0])
        mse = lambda x : self.mse(datax, datay, x[columns['W']],x[columns['W0']])
        mse_derive = lambda x : self.mse_derive(datax, datay, x[columns['W']],x[columns['W0']])
        
        x_histo,f_histo,_ = self.optimize(mse, mse_derive, initial, self.eps, self.max_iter)
        optimal_idx = f_histo.argmin()
        self.W = x_histo[optimal_idx][columns['W']]
        self.W0 = x_histo[optimal_idx][columns['W0']]
        return self
        
    def predict(self,datax):
        assert datax.shape[1] == len(self.W)
        return np.array([self.estimate(x, self.W, self.W0) for x in datax])
        
    def score(self,datax,datay):
        return np.count_nonzero(self.predict(datax) == datay) / datax.shape[0]