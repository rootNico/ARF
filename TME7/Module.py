# -*- coding: utf-8 -*-
import numpy as np

class Loss(object):
    def forward(self, y, yhat):
        pass
    
    def backward(self, y, yhat):
        pass 

class Module(object):
    def __init__(self):
        self._parameters = None
        self._gradient = None
        
    def zero_grad(self):
        ### Annule gradien
        pass

    def forward(self, X):
        ### Calcule la passe forward
        pass

    def update_parameters(self, gradient_step):
        ## Calcule la mise a jour des parametres selon le gradient calcule et
        ## le pas de gradient_step
        pass
    def backward_update_gradient(self, input, delta):
        ## Met a jour la valeur du gradient
        pass
    def backward_delta(self, input, delta):
        ## Calcul la dérivée de l'erreur
        pass


class ModuleLineaire(Module):
    def __init__(self, nbEntree, nbSortie):
        self._nbEntree = nbEntree
        self._nbSortie = nbSortie
        self._parameters = np.random.rand(nbEntree,nbSortie)
        self._gradient = np.zeros((nbEntree,nbSortie))
        self._loss = MSELoss()
        
    def zero_grad(self):
        ### Annule gradient
        self._gradient = np.zeros((nbEntree,nbSortie))
    
    def forward(self, X):
        ### Calcule la passe forward        
        #self._biais = np.random.rand()
        return np.dot(X.T, self._parameters) # + self._biais
    
    def update_parameters(self, gradient_step):
        ## Calcule la mise a jour des parametres selon le gradient calcule et
        ## le pas de gradient_step
        self._parameters -= gradient_step * self._gradient
        
    def backward_update_gradient(self, input, delta):
        ## Met a jour la valeur du gradient
        input = input.reshape(-1,1)
        print(input.shape)
        print(delta.shape)
        print(self._gradient.shape)
        
        #delta = delta.reshape(-1,1)
        gradient_change = np.dot(delta, input)
        print(gradient_change.shape)
        print('------')
        self._gradient += gradient_change
    
    def backward_delta(self, input, delta):
        ## Calcul la dérivée de l'erreur
        input = input.reshape(-1,1)
        sortie = np.dot(self._parameters, input)
        return self._loss.backward(sortie, delta).T


class FASigmoide(Module):
    def __init__(self):
        pass
        
    def f_activation(self, input):
        return np.power((1 + np.exp(input)), -1)
        
    def f_activation_g(self, input):
        return np.dot(input, 1 - input)
    
    def forward(self, X):
        ### Calcule la passe forward
        return self.f_activation(X)
    
    def backward_delta(self, input, delta):
        ## Calcul la dérivée de l'erreur
        return np.dot(self.f_activation_g(input), delta) 

class MSELoss(Loss):
    def forward(self, y, yhat):
        return ((y - yhat)**2).mean()

    def backward(self, y, yhat):
        return (yhat - y)#.mean()