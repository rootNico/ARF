import numpy as np
import matplotlib.pyplot as plt
from random import sample as rand_sample

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

def load_usps(filename, sample=-1):
    with open(filename,"r") as f:
        f.readline()
        data =[ [float(x) for x in l.split()] for l in f if len(l.split())>2]
    tmp = np.array(data)
    if sample > 0:
        idx = rand_sample(range(len(tmp)),sample)
        tmp = tmp[idx,:]
    return tmp[:,1:],tmp[:,0].astype(int)

def display_image ( X ):
    """
    Etant donné un tableau X de 256 flotants représentant une image de 16x16
    pixels, la fonction affiche cette image dans une fenêtre.
    """
    # on teste que le tableau contient bien 256 valeurs
    if X.size != 256:
        raise ValueError ( "Les images doivent être de 16x16 pixels" )

    # on crée une image pour imshow: chaque pixel est un tableau à 3 valeurs
    # (1 pour chaque canal R,G,B). Ces valeurs sont entre 0 et 1
    Y = X / X.max ()
    img = np.zeros ( ( Y.size, 3 ) )
    for i in range ( 3 ):
        img[:,i] = X

    # on indique que toutes les images sont de 16x16 pixels
    img.shape = (16,16,3)

    # affichage de l'image
    plt.imshow( img )
    plt.show ()