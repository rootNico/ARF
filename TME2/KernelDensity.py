import numpy as np

class KernelDensity:
    
    def parzen(x1, x2, h):
        norm = np.linalg.norm(x1 - x2)
        return 1 if norm <= h/2 else 0
    
    def gauss(x1, x2, h):
        x = np.linalg.norm((x1 - x2) / h)
        return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x**2)
    
    kernels = {
        'parzen' : parzen,
        'gauss' : gauss
    }
    
    def __init__(self, data, kernel='parzen'):
        self.data = data;
        self.kernel = self.kernels[kernel];

    def calcul_kernel_density_proba(self, x, X, h):
        n = X.shape[0]
        probas = np.array([self.kernel(x, xi, h) / n for xi in X])
        return probas.sum()
    
    def predict(self,grid, h):
        return np.array([self.calcul_kernel_density_proba(g, self.data, h) for g in grid])