"""
Detector (fluorescence telescope), set of PMTs (elevation pixels) and
simulation-run metadata.
"""

import numpy as np


class Detector:
    """
    Fluorescence telescope: position, field of view, optics and local
    aerosol atmosphere. Filled from the DataCard by setDetector();
    HIRESI() loads the HiRes-I configuration.
    """

    def __init__(self):
        self.Xtel = None
        self.Ytel = None
        self.Ztel = None
        self.MaxDist = None # Maximum distance the signal reaches [m]
        self.ElevMin = None # FoV elevation min [deg]
        self.ElevMax = None # FoV elevation max [deg]
        self.AziMin = None # FoV azimuth min [deg]
        self.AziMax = None # FoV azimuth max [deg]
        self.rTel = None
        self.nPMT = None
        self.nPMTtr = None
        self.HALAtSea = None
        self.AScaH = None
        self.HAL = None
        self.HALZ = None
        self.IsHIRESI = False

    def setDetector(self, datacard):
        """
        Copy the telescope configuration from the DataCard.
        """
        self.IsHIRESI = datacard.IsHIRESI
        self.Xtel = datacard.Xtel
        self.Ytel = datacard.Ytel
        self.Ztel = datacard.Ztel
        self.MaxDist = datacard.MaxDist
        self.ElevMin = datacard.ElevMin
        self.ElevMax = datacard.ElevMax
        self.AziMin = datacard.AziMin
        self.AziMax = datacard.AziMax
        self.rTel = datacard.rTel
        self.HAL = datacard.HAL
        self.HALZ = datacard.HALZ
        self.AScaH = datacard.AScaH
        self.HALAtSea = datacard.HALAtSea

        self.mirrorArea = datacard.mirrorArea
        self.telEff = datacard.telEff
        self.intTime = datacard.intTime

    def HIRESI(self):
        """
        Fixed HiRes-I detector configuration (for comparative studies).
        """
        self.IsHIRESI = True
        self.Xtel = 0
        self.Ytel = 0
        self.Ztel = 1.597e3
        self.MaxDist = 40e3
        self.ElevMin = 3
        self.ElevMax = 17
        self.AziMin = 0
        self.AziMax = 360
        self.rTel = np.array([self.Xtel, self.Ytel, self.Ztel])
        self.HAL = 25e3
        self.HALZ = 1.5e3 # reference height of HAL [m]
        self.AScaH = 1100
        self.HALAtSea = self.HAL*np.exp(-self.HALZ/self.AScaH)

        self.mirrorArea = 5.1
        self.telEff = 0.2
        self.intTime = 0.1 # microseconds


class Run:
    """
    Run metadata: file being read, current shower, primary particle,
    energy and geometry-generation parameters.
    """

    def __init__(self):
        self.slant = True
        self.NShwinFile = None
        self.NumShower = 0
        self.PrimType = None
        self.EnSim = None
        self.LongNum = None
        self.RMaxGen = None
        self.RMinGen = None
        self.ZenMaxGen = None


class PMT:
    """
    Set of pixels ("PMTs") observing the shower track.

    Each list has one value per pixel, filled by GetShowerInField
    (geometry) and Reconstruct/FluorescenceNpeKakimoto (signal).
    """

    def __init__(self):
        self.length = [] # track length seen by the PMT in meters
        self.lengthdeg = []
        self.dist = [] # distance of each PMT to the telescope
        self.r = [] # vector of each PMT to the telescope
        self.depth = [] # g/cm^2
        self.slantdepth = [] # g/cm^2
        self.elev = [] # elevation (alpha) of the PMT
        self.lmid = [] # value of l for each PMT
        self.NPart = [] # number of particles seen in each PMT
        self.factor = [] # conversion factor for the photons seen by the PMT

        self.NPhotonsIdeal = []
        self.TriggerEv = None
        self.NPhotonsMedido = []
        self.Triggers = []
        self.TimeSeen = [] # microsecond

        self.pixsize = 1 # degree
