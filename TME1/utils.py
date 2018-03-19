import numpy as np

def train_test_data(x,y, partition):
    train_n = int(len(x) * partition)
    test_n = int(len(x) - train_n)
    indices = np.random.permutation(len(x))
    training_idx, test_idx = indices[:train_n], indices[test_n:]
    return(x[training_idx,:], x[test_idx,:] ,y[training_idx], y[test_idx])

def divide_intervalles(X, Y, N):
    assert X.shape[0] == Y.shape[0]
    indexes = np.arange(X.shape[0])
    np.random.shuffle(indexes)
    index_intevals = np.array_split(indexes, N)
    res = np.array([(X[ind, :],Y[ind]) for ind in index_intevals])
    return (res[:, 0], res[:, 1])

def validation_croisee(X,Y, dt, N):
    intervalles_x, intervalles_y = divide_intervalles(X,Y,N)
    results = []
    for i in range(N):
        train_x = []
        train_y = []
        test_x = []
        test_y = []
        for j in range(N):
            if i != j:
                train_x.extend(intervalles_x[j])
                train_y.extend(intervalles_y[j])
            else:
                test_x.extend(intervalles_x[j])
                test_y.extend(intervalles_y[j])
        dt.fit(np.array(train_x), np.array(train_y))
        results.append(dt.score(np.array(test_x), np.array(test_y)))
    return np.array(results).mean()