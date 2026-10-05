"""
arapy: simulation of the response of fluorescence detectors (HiRes/
Auger/Telescope Array style) to CONEX longitudinal shower profiles.

Modernization in Python of the C++/ROOT program of de Souza,
Medina-Tanco & Ortiz, Phys. Rev. D 72 (2005) 103009, extended in
Carvalho Jr., Albuquerque & de Souza, Astropart. Phys. 28 (2007) 89.

Public API:
    DataCard   -- simulation configuration
    Simulation -- main loop (Simulation().Run(datacard, ApplyCuts))
"""

__version__ = "0.1.0"

from .datacard import DataCard
from .simulation import Simulation

from .atmosphere import Atmosphere
from .fluorescence import ReadFluoSpectrum