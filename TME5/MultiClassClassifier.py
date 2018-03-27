
import numpy as np
from utils import filter_values, one_againt_others

class MultiClassClassifier:
    
    def __init__(self, classifier,strategy="1vs1"):
        assert callable(classifier), "Please give a fonction, not a instance of classifier"
        self.clf = classifier
        self.strategy = strategy

        
    def fit_one_vs_one(self,X,Y):
        k = np.unique(Y)
        classifiers = {}
        for y in k:
            classifiers[y] = {}
        
        for yi in k:
            for yj in k:
                if yi == yj:
                    continue
                if classifiers[yi].get(yj, False) == False and \
                    classifiers[yj].get(yi, False) == False:
                        clfInstance = self.clf()
                        x,y = filter_values(X,Y, yi,yj)
                        clfInstance.fit(x,y)
                        classifiers[yi][yj] = clfInstance
                        
        return classifiers
    
    def fit_one_vs_all(self,X,Y):
        k = np.unique(Y)
        classifiers = {}
        for y in k:
            clfInstance = self.clf()
            x,y = one_againt_others(X,Y, y)
            clfInstance.fit(x,y)
            classifiers[y] = clfInstance
        
    def fit(self, X,Y):
        if self.strategy == "1vs1":
            self.classifiers = self.fit_one_vs_one(X,Y)
        if self.strategy == "1vsALL":
            self.classifiers = self.fit_one_vs_all(X,Y)
        
    def predict_one_vs_one(self, X):
        N = X.shape[0]
        D = len(self.classifiers.keys())
        results = np.zeros((N,D,D))
        for yi in range(D):
            for yj in range(D):
                if yi < yj:
                    clfI = self.classifiers[yi][yj]
                    prediction = clfI.predict(X)
                    results[:,yi,yj] = prediction
        prediction = np.zeros(N)
        for i in range(N):
            prediction[i] = results[i].sum(axis=0).argmax()
        return results,prediction
        
    def predict_one_vs_all(self,X):
        return np.zeros(X.shape[0])
        
    def predict(self,X):
        if self.strategy == "1vs1":
            return self.predict_one_vs_one(X)
        if self.strategy == "1vsALL":
            return self.predict_one_vs_all(X)
        return np.zeros(X.shape[0])
    
    def score(self,X,Y):
        _,pred = self.predict(X)
        return np.count_nonzero(pred == Y) / Y.shape[0]