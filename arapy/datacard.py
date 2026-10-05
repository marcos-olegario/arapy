"""
DataCard: configuration of a simulation run (input files, detector
geometry, aerosol atmosphere, trigger and output).
"""

import numpy as np


class DataCard:
    """
    Simulation configuration parameters.

    Parameters
    ----------
    files : list of str
        List of CONEX (.root) input files.
    save_hdf5 : bool
        If True, every surviving event is written to `output_file`
        (HDF5). Independently of this flag, the accumulated arrays are
        always available in Simulation.out after the run.
    output_file : str
        Name of the HDF5 output file (overwritten at the start of each
        run when save_hdf5 is True).
    verbose : int
        Progress reporting (0 none, 1 live progress bar,
        2 single progress bar printed after each file).
    nRep : int
        Number of geometries drawn per shower.
    nMax : int or None
        Maximum number of simulated showers (None = no limit).
    slant : bool
        True if the CONEX profile is in slant depth.
    IsHIRESI : bool
        Use the HiRes-I fit prescription (Xmax bounds and fixed-Xmax grid
        refit, only with free X0).

    Other Parameters
    ----------------
    Xtel, Ytel, Ztel : float
        Telescope position [m] (detector geometry / generation).
    MaxDist : float
        Maximum signal range [m].
    RMinGen, RMaxGen : float
        Core sampling ring [m].
    ZenMaxGen : float
        Maximum generated zenith angle [degrees].
    ElevMin, ElevMax : float
        Field of view in elevation [degrees].
    AziMin, AziMax : float
        Field of view in azimuth [degrees].
    HAL : float
        Horizontal attenuation length at height HALZ [m]
        (exponential Mie aerosol model).
    HALZ : float
        Reference height for HAL (default: Ztel) [m].
    AScaH : float
        Aerosol scale height h_M [m].
    HALAtSea : float or None
        Attenuation length at sea level l_M [m]
        (None: computed as HAL*exp(-HALZ/AScaH)).
    mirrorArea : float
        Mirror area [m^2] (telescope and trigger).
    telEff : float
        Total optical efficiency.
    intTime : float
        Integration window [us].
    numPMTact : int
        Minimum number of activated pixels (trigger).
    numPMTadj : int
        Minimum number of adjacent activated pixels (trigger).
    numPMTtrig : int
        Minimum number of points for the GH fit.
    pixsize : float
        Pixel angular size [degrees].
    x0 : float or None
        Fixed X0 in the GH fit [g/cm^2] (None: free X0).
    lambdaGH : float
        Fixed lambda of the GH function [g/cm^2].
    """

    def __init__(self,
                files,
                save_hdf5 = False,
                output_file = 'arapy_output.h5',
                verbose = 1,
                nRep = 1,                       # number of repetitions for the same shower
                nMax = None,
                slant = True,
                IsHIRESI = False,
                Xtel = 0,
                Ytel = 0,
                Ztel = 1.597e3,
                MaxDist = 40e3,
                RMinGen = 0,
                RMaxGen = 35e3,
                ZenMaxGen = 60,
                ElevMin = 3,
                ElevMax = 31,
                AziMin = 0,
                AziMax = 360,
                HAL = 25e3,
                HALZ = None,                    # default: telescope height
                AScaH = 1200,                   # h_M
                HALAtSea = 14000,               # l_M  if None: HAL*np.exp(-HALZ/AScaH)
                mirrorArea = 5.1,
                telEff = 0.2,
                intTime = 0.1,                  # microseconds
                numPMTact = 5,                  # Trigger condition
                numPMTadj = 3,                  # Trigger condition
                numPMTtrig = 3,                 # Rec. condition
                x0 = -100,
                lambdaGH = 70,
                pixsize = 1
            ):

        self.files = files
        self.save_hdf5 = save_hdf5
        self.output_file = output_file
        self.verbose = verbose
        self.nRep = nRep
        self.nMax = nMax
        self.slant = slant
        self.IsHIRESI = IsHIRESI
        self.Xtel = Xtel
        self.Ytel = Ytel
        self.Ztel = Ztel
        self.MaxDist = MaxDist
        self.RMinGen = RMinGen
        self.RMaxGen = RMaxGen
        self.ZenMaxGen = ZenMaxGen
        self.ElevMin = ElevMin
        self.ElevMax = ElevMax
        self.AziMin = AziMin
        self.AziMax = AziMax
        self.HAL = HAL
        self.HALZ = HALZ if HALZ is not None else Ztel
        self.AScaH = AScaH
        self.mirrorArea = mirrorArea
        self.telEff = telEff
        self.intTime = intTime
        self.numPMTact = numPMTact
        self.numPMTadj = numPMTadj
        self.numPMTtrig = numPMTtrig
        self.x0 = x0
        self.lambdaGH = lambdaGH
        self.pixsize = pixsize

        if HALAtSea is None:
            self.HALAtSea = HAL*np.exp(-self.HALZ/AScaH)
        else:
            self.HALAtSea = HALAtSea

        self.rTel = np.array([Xtel, Ytel, Ztel])
