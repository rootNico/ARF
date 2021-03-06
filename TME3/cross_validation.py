
import numpy as np
from scipy.sparse import vstack

def divide_intervalles(X, N):
    indexes = np.arange(X.shape[0])
    np.random.shuffle(indexes)
    return np.array_split(indexes, N)

def validation_croisee(X,Y,method, N):
    """
    
    """
    intervalles = divide_intervalles(X,N)
    results = []
    for i in range(N):
        train = np.array([], dtype=int)
        test = np.array([], dtype=int)
        for j in range(N):
            if i != j:
                train = np.concatenate((train,intervalles[j]))
            else:
                test = np.concatenate((test,intervalles[j]))
        model = method.fit(X[train,:], Y[train])
        score,_ = model.score(X[test,:], Y[test])
        results.append(score)
    return np.array(results).mean()