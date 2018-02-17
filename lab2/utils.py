import numpy as np
from random import randrange
from math import sqrt, ceil

def eval_numerical_gradient(f, x, verbose=True, h=0.00001):
    """ 
    Implementare naiva pentru calcularea gradientului numeric al functiei f
    in punctul x 
    * f trebuie sa fie o functie care primeste un singur argument
    * x este un punct intr-un spatiu n dimensional al parametrilor functiei f (numpy array) in care vrem sa evaluam gradientul
    """ 

    fx = f(x) # valoare functiei in punctul initial
    grad = np.zeros_like(x)
    # iteram prin toate dimensiunile lui x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        # evaluam functia in x+h
        ix = it.multi_index
        oldval = x[ix]
        x[ix] = oldval + h # incrementam cu h
        fxph = f(x) # evaluam f(x + h)
        x[ix] = oldval - h
        fxmh = f(x) # evaluam f(x - h)
        x[ix] = oldval # restore

        # calculam derivata partial folosing formula centrata
        grad[ix] = (fxph - fxmh) / (2 * h) # panta
        if verbose:
            print(ix, grad[ix])
        it.iternext() # trecem la urmatoarea dimensiune

    return grad


def visualize_grid(Xs, ubound=255.0, padding=1):
    """
    Redimensionarea unui tensor 4D reprezentand niste parametrii (filtre) pentru a fi vizualizat mai usor ca un grid.

    Input:
    - Xs: Parametrii de tip (N, H, W, C)
    - ubound: Gridul de iesire va avea valori scalate intre [0, ubound]
    - padding: Numarul de pixeli blank intre celulele 
    """
    (N, H, W, C) = Xs.shape
    grid_size = int(ceil(sqrt(N)))
    grid_height = H * grid_size + padding * (grid_size - 1)
    grid_width = W * grid_size + padding * (grid_size - 1)
    grid = np.zeros((grid_height, grid_width, C))
    next_idx = 0
    y0, y1 = 0, H
    for y in range(grid_size):
        x0, x1 = 0, W
        for x in range(grid_size):
            if next_idx < N:
                img = Xs[next_idx]
                low, high = np.min(img), np.max(img)
                grid[y0:y1, x0:x1] = ubound * (img - low) / (high - low)
                # grid[y0:y1, x0:x1] = Xs[next_idx]
                next_idx += 1
            x0 += W + padding
            x1 += W + padding
        y0 += H + padding
        y1 += H + padding
    # grid_max = np.max(grid)
    # grid_min = np.min(grid)
    # grid = ubound * (grid - grid_min) / (grid_max - grid_min)
    return grid