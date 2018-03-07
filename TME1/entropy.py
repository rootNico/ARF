import numpy as np
from collections import Counter

def entropie(vect):
    c = Counter()
    c.update(vect)
    probas = np.array(list(c.values())) / len(vect)
    return -np.array([y * np.log(y) for y in probas]).sum()

def cond_entropie(list_vect):
    res = 0
    total = np.array([len(v) for v in list_vect]).sum()
    for vect in list_vect:
        res += (len(vect)/total) * entropie(vect)
    return res