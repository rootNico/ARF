import numpy as np

class NaiveBayes_classifier:
    
    def fit(self, datax, datay):
        model = {}
        for cl in np.unique(datay):
            X = datax[datay == cl]
            model[cl] = learnML_class_parameters(X)
        self.model = model
        return self
        
    def predict(self,datax):
        predictions = []
        for x in datax:
            res = np.array([[log_likelihood(x,self.model[m]),m] for m in self.model.keys()])
            probas = res[:,0]
            classes = res[:,1]
            predictions.append(classes[probas.argmax()])
        return np.array(predictions)
    
    def score(self,X,Y):
        labels = self.predict(X)
        return np.count_nonzero(labels == Y) / X.shape[0], labels
    

def learnML_class_parameters(table):
    return (np.mean(table, axis=0),np.var(table, axis=0))

def learnML_all_parameters(data):
    return np.array([learnML_class_parameters(dim) for dim in data])

def log_likelihood(image,table):
    a = 0
    for i in range(256):
        if table[1][i] != 0:
            a += -0.5*np.log(2*np.pi*table[1][i])-0.5*(image[i]-table[0][i])**2 / table[1][i]
    return a
    

def log_likelihoods(image,tables):
    return np.array([ log_likelihood ( image, tables[i] ) for i in range (10) ])