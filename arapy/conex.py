"""
Reading CONEX (.root) files with uproot: all required branches are read
ONCE per file into numpy arrays, and showers are iterated in memory
(without reopening the file per shower).
"""

import numpy as np
import uproot


class ConexFile:
    """
    A CONEX file opened for batch reading.

    Usage::

        cx = ConexFile('showers.root')
        for idx in range(cx.nShowers):
            chuveiro = Shower()
            chuveiro.FillFromConex(cx, rodada, idx)

    Attributes
    ----------
    nShowers : int
        Number of showers in the Shower tree.
    particle : int
        Primary particle code (Header tree, CONEX convention).
    data : dict
        Mapping branch name -> numpy array (branches in BRANCHES).
    """

    # Branches used by the simulation (Shower tree)
    BRANCHES = [
        "lgE", "zenith", "azimuth", "Xfirst",
        "X0", "Xmax", "Nmax", "p1", "p2", "p3", "chi2", "dEdXmx", "nX",
        "X", "N", "dEdX",
    ]

    def __init__(self, filename):
        with uproot.open(filename) as f:
            self.data = f["Shower"].arrays(self.BRANCHES, library="np")
            self.particle = int(f["Header"]["Particle"].array(library="np")[0])

        self.nShowers = len(self.data["lgE"])
