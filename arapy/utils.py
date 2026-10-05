"""
Utilities: simulation progress and the Gaisser-Hillas function.
"""

import numpy as np
from tqdm import tqdm
import io
import os

c = 3e8 # Speed of light [m/s]

def d2rad(ang):
    """Degrees -> radians."""
    return ang * np.pi / 180

def rad2d(ang):
    """Radians -> degrees."""
    return 180*ang/np.pi


def GaisserHillas(X, nmax, xmax, x0, lam=70.0):
    """
    Gaisser-Hillas longitudinal profile with fixed lambda (default
    70 g/cm^2; set through DataCard.lambdaGH in the reconstruction).

    Accepts scalar X (e.g. scipy quad) or array (e.g. curve_fit); returns
    0 where X <= x0 (no complex values/NaN during the fit).

    Parameters
    ----------
    X : float or ndarray
        Depth(s) [g/cm^2].
    nmax : float
        Value at the maximum.
    xmax : float
        Depth of the maximum [g/cm^2].
    x0 : float
        Effective starting point [g/cm^2].
    lam : float
        Fixed shape parameter lambda [g/cm^2].

    Returns
    -------
    float or ndarray
        Profile evaluated at X (scalar if X was scalar).
    """
    # Make sure X is handled as a numpy array internally
    X_arr = np.asarray(X)

    # Avoid division by zero or a negative base during optimizer guesses
    if xmax <= x0:
        return np.zeros_like(X_arr)

    base = (X_arr - x0) / (xmax - x0)
    expoente = (xmax - x0) / lam

    # Evaluate the function only where it is physically meaningful (X > X0 and base > 0)
    resultado = np.zeros_like(X_arr, dtype=float)
    mascara = (X_arr > x0) & (base > 0)

    resultado[mascara] = nmax * (base[mascara] ** expoente) * np.exp((xmax - X_arr[mascara]) / lam)

    # If the original X was a scalar (as in quad), return a scalar
    if np.isscalar(X):
        return resultado.item()

    return resultado
    
def ShowerProgress(nShowers, verbose, desc=""):
    """
    Progress bar for the shower loop, controlled by datacard.verbose.
    
    0: disabled, 1: live bar, 2: nothing during the loop, the completed
    bar is printed afterwards (safe for redirected logs).
    """
    if verbose == 2:
        # output goes to a throwaway buffer; nothing reaches the terminal
        return tqdm(range(nShowers), desc=desc, unit=" shower", ascii=False,
                  file=io.StringIO(), mininterval=float("inf"))
    
    return tqdm(range(nShowers), desc=desc, unit=" shower",
              disable=(verbose == 0))

