
import numpy as np
import matplotlib.pyplot as plt

def load_usps(filename):
    with open(filename,"r") as f:
        f.readline()
        data =[ [float(x) for x in l.split()] for l in f if len(l.split())>2]
    tmp = np.array(data)
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
    
def filter_values(x, y, a,b):
    indexes = np.where(np.logical_or(y == a, y == b))[0]
    Y = np.array(y[indexes])
    Y[Y == a] = 0
    Y[Y == b] = 1
    Y_vector = np.zeros((Y.shape[0],2))
    Y_vector[:,0] = np.where(Y == 0, 1, 0)
    Y_vector[:,1] = np.where(Y == 1, 1, 0)
    return np.array(x[indexes]),Y_vector