import numpy as np

def filter_values(x, y, a,b):
    indexes = np.where(np.logical_or(y == a, y == b))[0]
    Y = np.array(y[indexes])
    Y[Y == a] = -1
    Y[Y == b] = 1
    return np.array(x[indexes]),Y

def one_againt_others(x, y, num):
    indexes = np.where(y == num)[0]
    new_y = -np.ones(y.shape[0])
    new_y[indexes] = 1
    return x,new_y

def minibatch_indexes(N, batchsize, shuffle=False):
    if shuffle:
        indices = np.arange(N)
        np.random.shuffle(indices)
    batches = []
    for start_idx in range(0, N - batchsize + 1, batchsize):
        if shuffle:
            batches.append(indices[start_idx:start_idx + batchsize])
        else:
            batches.append(slice(start_idx, start_idx + batchsize))
    return np.array(batches)

def add_product_column(X):
    new_col = np.product(X, axis=1).reshape(-1,1)
    return np.append(X, new_col, axis=1)

def add_one_column(X):
    N = X.shape[0]
    return np.append(np.ones((N,1)),X, axis=1)

def polynomial(X, degree):
    assert degree >= 0
    if degree == 0:
        return X
    if degree == 1:
        return add_one_column(X)
    N = X.shape[0]
    D = X.shape[1] * (degree - 1)
    new_cols = np.zeros((N,D))
    original_columns = X.shape[1]
    for d in range(2, degree+1):
        for i in range(original_columns):
            c = (d-2) * (original_columns) + i
            new_cols[:,c] = X[:,i] ** d
    X_p = add_product_column(X)
    out = np.append(X_p, new_cols, axis=1)
    return add_one_column(out)