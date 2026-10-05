"""
Shower: state of an extensive air shower -- longitudinal profile read
from CONEX, sampled geometry and trigger/reconstruction results.
"""

import numpy as np


class Shower:
    """
    A simulated shower.

    Groups three blocks of state:

    - longitudinal profile and CONEX parameters (depth, LongProf, Xmax...)
      -- filled by FillFromConex and preserved across repetitions;
    - sampled geometry (rCore, nHat, Rp...) -- SetShowerGeometry /
      GetShowerInField;
    - trigger and reconstruction results (Trigger, NmaxGH, EcalEeV...)
      -- Reconstruct.

    Clear() erases the last two blocks so the same CONEX profile can be
    reused with a new geometry.
    """

    def __init__(self):

        # Shower geometry:

        self.Xc = None
        self.Yc = None
        self.Zc = None
        self.ZenAng = None
        self.AziAng = None
        self.Xin = None
        self.Yin = None
        self.Zin = None
        self.Xout = None
        self.Yout = None
        self.Zout = None
        self.IsInFoV = None
        self.nHat = None
        self.rCore = None
        self.inFoV = None
        self.RinField = None
        self.RoutField = None
        self.linField = None
        self.loutField = None
        self.Rp = None
        self.coreDist = None
        self.nSDP = None      # unit normal of the shower-detector plane
        self.SDPangle = None  # inclination of the SDP w.r.t. the vertical [deg]

        # Acceptance and reconstruction:

        self.cut = None
        self.cutReason = None

        self.pathlength = None
        self.chi0 = None
        self.Psi = None
        self.speed = None
        self.speedrad = None
        self.time = None
        self.firstPMTelev = None
        self.lastPMTelev = None
        self.MinViewAng = None        # minimum viewing angle [deg]

        # Longitudinal profile:

        self.ZenSim = None
        self.AziSim = None
        self.depth = None
        self.depthMid = None
        self.LongProf = None
        self.EnDeposit = None
        self.dEdXMaxSim = None
        self.NmaxSim = None
        self.XmaxSim = None
        self.X0Sim = None
        self.p1Sim = None
        self.p2Sim = None
        self.p3Sim = None
        self.xfirst = None
        self.EnSim = None
        self.chiconex = None

        # Reconstruction:

        self.Trigger = None
        self.FitGH = None
        self.NmaxGH = None
        self.XmaxGH = None
        self.X0GH = None
        self.ChisqGH = None
        self.NdofGH = None
        self.ChisqNdofGH = None
        self.EcalEeV = None
        self.E0EeV = None
        self.UsdPointsGH = None
        self.DepHiEle = None
        self.DepLoEle = None
        self.ShowerSize = None
        self.ProfMed = None
        self.DepthMed = None
        self.XmaxVis = None

        # Signal, in the three scales that coexist in the chain (see
        # Reconstruct): per integration window [pe], the intermediate
        # product signal x time seen [pe.us], and the total integrated
        # photoelectrons [pe] -- the last one is what experiments publish.
        self.SumSignal = None
        self.SumSignalPeUs = None
        self.SumSignalPe = None
        self.AveSignal = None
        self.AveSignalPeUs = None
        self.AveSignalPe = None

        # Relative deviations reconstructed vs. simulated:
        self.XmaxErr = None
        self.NmaxErr = None
        self.E0err_GH = None
        self.E0rat_GH = None

    def Clear(self):
        """
        Erase geometry and reconstruction results, preserving the CONEX
        profile -- used when repeating the same shower with a new geometry.
        """
        self.Xc = None
        self.Yc = None
        self.Zc = None
        self.ZenAng = None
        self.AziAng = None
        self.Xin = None
        self.Yin = None
        self.Zin = None
        self.Xout = None
        self.Yout = None
        self.Zout = None
        self.Rp = None
        self.coreDist = None
        self.IsInFoV = None
        self.nHat = None
        self.rCore = None
        self.inFoV = None
        self.RinField = None
        self.RoutField = None
        self.linField = None
        self.loutField = None
        self.nSDP = None
        self.SDPangle = None

        self.cut = None
        self.cutReason = None

        self.pathlength = None
        self.chi0 = None
        self.Psi = None
        self.speed = None
        self.speedrad = None
        self.time = None
        self.firstPMTelev = None
        self.lastPMTelev = None
        self.MinViewAng = None

        self.Trigger = None
        self.FitGH = None
        self.NmaxGH = None
        self.XmaxGH = None
        self.X0GH = None
        self.ChisqGH = None
        self.NdofGH = None
        self.ChisqNdofGH = None
        self.EcalEeV = None
        self.E0EeV = None
        self.UsdPointsGH = None
        self.DepHiEle = None
        self.DepLoEle = None
        self.ShowerSize = None
        self.ProfMed = None
        self.DepthMed = None
        self.XmaxVis = None
        self.SumSignal = None
        self.SumSignalPeUs = None
        self.SumSignalPe = None
        self.AveSignal = None
        self.AveSignalPeUs = None
        self.AveSignalPe = None
        self.XmaxErr = None
        self.NmaxErr = None
        self.E0err_GH = None
        self.E0rat_GH = None

    def ConvertPartID(self, ID):
        """
        Convert the CONEX primary-particle code to the CORSIKA one.
        """
        if ID == 0: ID = 1
        elif ID == 100: ID = 14
        elif ID == 400: ID = 402
        elif ID == 1200: ID = 1206
        elif ID == 2800: ID = 2814
        elif ID == 5600: ID = 5626

        return ID

    def FillFromConex(self, cxfile, rodada, idx):
        """
        Fill the longitudinal profile and simulated parameters from
        shower `idx` of an already loaded ConexFile (arapy.conex).

        If rodada.slant is False, the CONEX depths are taken as vertical
        and converted once to slant depth with the CONEX zenith angle.
        """
        d = cxfile.data

        depth = np.asarray(d["X"][idx], dtype=float)

        self.ZenSim = float(d["zenith"][idx])
        self.AziSim = float(d["azimuth"][idx])

        if not rodada.slant:
            depth = depth / np.cos(np.radians(self.ZenSim))

        self.depth = depth.tolist()
        self.LongProf = np.asarray(d["N"][idx], dtype=float).tolist()
        self.depthMid = (0.5 * (depth[:-1] + depth[1:])).tolist()
        self.EnDeposit = np.asarray(d["dEdX"][idx], dtype=float)[:-1].tolist()
        self.dEdXMaxSim = float(d["dEdXmx"][idx])
        self.NmaxSim = float(d["Nmax"][idx])
        self.XmaxSim = float(d["Xmax"][idx])
        self.X0Sim = float(d["X0"][idx])
        self.p1Sim = float(d["p1"][idx])
        self.p2Sim = float(d["p2"][idx])
        self.p3Sim = float(d["p3"][idx])
        self.chiconex = float(d["chi2"][idx])

        rodada.PrimType = self.ConvertPartID(cxfile.particle)
        rodada.EnSim = 10**(float(d["lgE"][idx])-18) # EeV
        rodada.LongNum = int(d["nX"][idx]) # number of points in slant depth

        self.xfirst = float(d["Xfirst"][idx])
        self.EnSim = 10**(float(d["lgE"][idx])-18) # EeV

        rodada.NShwinFile = idx + 1

    def getE0eV(self, E):
        """
        Convert energy in EeV to log10(E/eV).
        """
        return 18 + np.log10(E)
