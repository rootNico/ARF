import numpy as np

def optimize(fonc,dfonc,xinit,eps,max_iter):
        x_histo = []
        f_histo = []
        grad_histo = []
        dernier = xinit
        for i in range(max_iter):
            x_histo.append(dernier - (eps * dfonc(dernier)))
            f_histo.append(fonc(dernier))
            grad_histo.append(dfonc(dernier))
            dernier = x_histo[i]
        return (np.array(x_histo), np.array(f_histo), np.array(grad_histo))