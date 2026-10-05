"""
Main simulation loop: reads the CONEX showers, draws geometries,
simulates the detector response, applies cuts and accumulates/writes
the output.
"""

import os
import numpy as np
import time
import copy
import h5py

from .atmosphere import Atmosphere
from .conex import ConexFile
from .shower import Shower
from .detector import Detector, Run, PMT
from .geometry import SetShowerGeometry, GetShowerInField
from .fluorescence import ReadFluoSpectrum
from .reconstruction import Reconstruct
from .utils import ShowerProgress


def DefaultCuts(chuveiro, telescopio, pmt):
    """
    Default selection: keep every triggered event with a converged GH fit.

    cutReason: 0 = no trigger, 1 = GH fit failed.
    """
    if not chuveiro.Trigger:
        chuveiro.cut, chuveiro.cutReason = True, 0
    elif not chuveiro.FitGH:
        chuveiro.cut, chuveiro.cutReason = True, 1
    else:
        chuveiro.cut = False


class Simulation:
    """
    Orchestrates the full simulation.

    Usage::

        sim = Simulation()
        sim.Run(datacard)              # default cuts (trigger + GH fit)
        sim.Run(datacard, ApplyCuts)   # user-defined cuts

    where ApplyCuts(chuveiro, telescopio, pmt) sets chuveiro.cut (bool)
    and chuveiro.cutReason (int in [0, 20)). A custom ApplyCuts must
    reject events without trigger or without a GH fit (see DefaultCuts).

    After the run, sim.out holds the accumulated numpy arrays of the
    surviving events (and the HDF5 file is written if
    datacard.save_hdf5). If sim.debug is set to True before the run,
    sim.shower/detector/pmt/run keep a copy of the first event that
    reached the reconstruction.
    """

    def __init__(self):
        self.fluspc = ReadFluoSpectrum()
        self.shower = None
        self.detector = None
        self.pmt = None
        self.run = None
        self.out = None

        #### Debug ####
        self.debug = False

    class Output:
        """
        Output accumulator: per-event quantities of the surviving events,
        converted to numpy arrays at the end of the run (SaveArrays).
        Rp is filled for every generated geometry and CutReason for every
        cut event.
        """

        def __init__(self):
            self.time = None
            self.eff = None
            self.nSim = None
            self.survs = None
            self.nTrigger = None
            self.notInFOV = None
            self.cutReason = None

            self.Xfirst = []

            #### Plots: ####
            self.path_ErrEgh = []
            self.Nmax_ErrEgh = []
            self.Xmax_ErrEgh = []
            self.core = []

            #### Histograms: ####
            # Energy and reconstruction:
            self.E0Sim = []
            self.E0Rec_GH = []
            self.E0Err_GH = []
            self.E0Rat_GH = []
            self.ChisqNDOFGH = []

            # Longitudinal profile:
            self.XmaxSim = []
            self.XmaxRec = []
            self.NmaxSim = []
            self.NmaxRec = []
            self.XmaxError = []
            self.NmaxError = []
            self.ShowerSize = []

            # Geometry and trigger:
            self.ZenAng = []
            self.AziAng = []
            self.Rp = []
            self.RpCut = []
            self.Path = []
            self.AveSig = []      # mean signal per triggered tube [pe/window]
            self.AveSigPe = []    # mean INTEGRATED photoelectrons per tube [pe]
            self.Npmt = []
            self.NphtPath = []     # signal per degree of track [pe/window/deg]
            self.NpeUsPerDeg = []  # idem, intermediate scale [pe.us/deg]
            self.NpePerDeg = []    # INTEGRATED photoelectrons per degree [pe/deg]
            self.Sinal = []
            self.CutReason = []
            self.Chi0 = []
            self.corDist = []

            #### Observed longitudinal profiles (one array per event): ####
            self.depth_obs = []
            self.prof_obs = []

        def saveDataset(self, out_name, event_id, chuveiro, telescopio):
            """
            Write a surviving event as group `run_NNNNNNN` in the HDF5
            file out_name: scalars as attributes, profiles (simulated and
            observed) as compressed datasets.
            """
            with h5py.File(out_name, 'a') as f:
                # Create a group for this event
                group = f.create_group(f'run_{event_id:07}')

                # ===== 1D VARIABLES =====
                # Fixed arrays (one value per event each)
                group.attrs['firstPMTelev'] = chuveiro.firstPMTelev
                group.attrs['pathlength'] = chuveiro.pathlength
                group.attrs['DepHiEle'] = chuveiro.DepHiEle
                group.attrs['DepLoEle'] = chuveiro.DepLoEle
                group.attrs['FitGH'] = chuveiro.FitGH
                group.attrs['XmaxVis'] = chuveiro.XmaxVis
                group.attrs['Rp'] = chuveiro.Rp
                group.attrs['speed'] = chuveiro.speed
                group.attrs['time'] = chuveiro.time
                group.attrs['ChisqNdofGH'] = chuveiro.ChisqNdofGH
                group.attrs['chi0'] = chuveiro.chi0
                group.attrs['nPMTtr'] = telescopio.nPMTtr
                group.attrs['SumSignal_per_pathlength'] = chuveiro.SumSignal / chuveiro.pathlength
                group.attrs['SumSignal'] = chuveiro.SumSignal
                group.attrs['AveSignal'] = chuveiro.AveSignal
                group.attrs['nPMTtr_per_pathlength'] = telescopio.nPMTtr / chuveiro.pathlength
                group.attrs['nPMTtr_per_nPMT'] = telescopio.nPMTtr / telescopio.nPMT
                group.attrs['zenAng'] = chuveiro.ZenAng
                group.attrs['aziAng'] = chuveiro.AziAng
                group.attrs['zenSim'] = chuveiro.ZenSim
                group.attrs['aziSim'] = chuveiro.AziSim
                group.attrs['SDPangle'] = chuveiro.SDPangle
                group.attrs['AveSignalPe'] = chuveiro.AveSignalPe
                group.attrs['SumSignalPe_per_pathlength'] = chuveiro.SumSignalPe / chuveiro.pathlength
                group.attrs['ShowerSize'] = chuveiro.ShowerSize
                group.attrs['coreDist'] = chuveiro.coreDist
                group.attrs['xfirst'] = chuveiro.xfirst
                group.attrs['deltaSim'] = chuveiro.XmaxSim - chuveiro.xfirst
                group.attrs['EnSim'] = chuveiro.EnSim
                group.attrs['E0EeV'] = chuveiro.E0EeV
                group.attrs['EcalEeV'] = chuveiro.EcalEeV
                group.attrs['chiconex'] = chuveiro.chiconex

                # ===== VARIABLE-SIZE ARRAYS =====
                # depth_med and long_med (same size, vary per event)
                group.create_dataset('depth_med', data=chuveiro.DepthMed, compression='gzip')
                group.create_dataset('long_med', data=chuveiro.ProfMed, compression='gzip')

                # ===== FIXED-SIZE ARRAYS =====
                core = [chuveiro.Xc, chuveiro.Yc, chuveiro.Zc]
                group.create_dataset('core', data=core, compression='gzip')

                params_rec = [chuveiro.NmaxGH, chuveiro.XmaxGH, chuveiro.X0GH]
                group.create_dataset('params_rec', data=params_rec, compression='gzip')

                params_sim = [chuveiro.dEdXMaxSim, chuveiro.XmaxSim, chuveiro.X0Sim]
                group.create_dataset('params_sim', data=params_sim, compression='gzip')

                group.create_dataset('depth_sim', data=chuveiro.depthMid, compression='gzip')
                group.create_dataset('long_sim', data=chuveiro.EnDeposit, compression='gzip')

        def SaveArrays(self):
            """
            Convert the accumulated lists into numpy arrays. The observed
            profiles (depth_obs, prof_obs) stay as lists, since their
            length varies from event to event.
            """
            self.Xfirst = np.array(self.Xfirst)

            self.path_ErrEgh = np.array(self.path_ErrEgh)
            self.Nmax_ErrEgh = np.array(self.Nmax_ErrEgh)
            self.Xmax_ErrEgh = np.array(self.Xmax_ErrEgh)
            self.core = np.array(self.core)
            self.E0Sim = np.array(self.E0Sim)
            self.E0Rec_GH = np.array(self.E0Rec_GH)
            self.E0Err_GH = np.array(self.E0Err_GH)
            self.E0Rat_GH = np.array(self.E0Rat_GH)
            self.ChisqNDOFGH = np.array(self.ChisqNDOFGH)
            self.XmaxSim = np.array(self.XmaxSim)
            self.XmaxRec = np.array(self.XmaxRec)
            self.NmaxSim = np.array(self.NmaxSim)
            self.NmaxRec = np.array(self.NmaxRec)
            self.XmaxError = np.array(self.XmaxError)
            self.NmaxError = np.array(self.NmaxError)
            self.ShowerSize = np.array(self.ShowerSize)
            self.ZenAng = np.array(self.ZenAng)
            self.AziAng = np.array(self.AziAng)
            self.Rp = np.array(self.Rp)
            self.RpCut = np.array(self.RpCut)
            self.Path = np.array(self.Path)
            self.AveSig = np.array(self.AveSig)
            self.AveSigPe = np.array(self.AveSigPe)
            self.Npmt = np.array(self.Npmt)
            self.NphtPath = np.array(self.NphtPath)
            self.NpeUsPerDeg = np.array(self.NpeUsPerDeg)
            self.NpePerDeg = np.array(self.NpePerDeg)
            self.Chi0 = np.array(self.Chi0)
            self.corDist = np.array(self.corDist)

            self.Sinal = np.array(self.Sinal)
            self.CutReason = np.array(self.CutReason)

    def Run(self, datacard, ApplyCuts=None):
        """
        Execute the full simulation described by the datacard.

        For each CONEX file, read the showers in sequence; for each
        shower, draw datacard.nRep geometries; for each geometry, project
        the track onto the detector, simulate the signal (fluorescence +
        noise), apply the trigger, reconstruct the profile
        and call ApplyCuts (DefaultCuts if None). Surviving events are
        accumulated in self.out and, if datacard.save_hdf5, written to
        datacard.output_file.
        """
        if ApplyCuts is None:
            ApplyCuts = DefaultCuts

        verbose = datacard.verbose

        self.out = self.Output()

        if datacard.save_hdf5:
            # Start from an empty file (overwrites a previous run)
            with h5py.File(datacard.output_file, 'w'):
                pass

        #### Bookkeeping: ####
        nSim = 0       # number of simulated showers
        survs = 0      # number of surviving showers
        nTrigger = 0   # number of triggered showers
        notInFOV = 0   # number of showers outside the FoV
        cutReason = np.zeros(20, dtype=int)

        # -------------
        # FIRST LOOP:
        # -------------
        # Initial simulation parameters:
        rng = np.random.default_rng()
        atm = Atmosphere()

        inicio1 = time.time()
        if verbose:
            print("Started", time.strftime("%Y-%m-%d %H:%M:%S"))
        for file in datacard.files:
            '''
            Read all input files
            '''
            rodada = Run()
            rodada.NShwinFile = 0
            rodada.slant = datacard.slant
            rodada.RMinGen = datacard.RMinGen
            rodada.RMaxGen = datacard.RMaxGen
            rodada.ZenMaxGen = datacard.ZenMaxGen

            # All branches of the file are read at once (uproot)
            cxfile = ConexFile(file)

            # -------------
            # SECOND LOOP:
            # -------------
            bar = ShowerProgress(cxfile.nShowers, verbose,
                                 desc=os.path.basename(file))
            for iShw in bar:
                '''
                Read all showers of the file
                '''
                chuveiro = Shower()
                chuveiro.FillFromConex(cxfile, rodada, iShw)

                # -------------
                # THIRD LOOP:
                # -------------
                for _ in range(datacard.nRep):
                    '''
                    Simulate the same shower several times
                    '''
                    if datacard.nMax and nSim >= datacard.nMax:
                        continue

                    rodada.NumShower = nSim
                    chuveiro.Clear() # Erase leftovers from other showers, keep only the .root content

                    telescopio = Detector()
                    telescopio.setDetector(datacard)

                    SetShowerGeometry(chuveiro, telescopio, rodada, rng)

                    pmt = PMT()
                    pmt.pixsize = datacard.pixsize
                    GetShowerInField(chuveiro, telescopio, pmt, atm)
                    self.out.Rp.append(chuveiro.Rp)

                    nSim += 1
                    if not chuveiro.IsInFoV:
                        notInFOV += 1
                        continue

                    Reconstruct(chuveiro, telescopio, rodada, pmt, rng, datacard, atm, self.fluspc)

                    if chuveiro.Trigger:
                        nTrigger += 1

                    if self.debug:
                        self.shower = copy.deepcopy(chuveiro)
                        self.detector = copy.deepcopy(telescopio)
                        self.pmt = copy.deepcopy(pmt)
                        self.run = copy.deepcopy(rodada)
                        self.debug = False

                    ApplyCuts(chuveiro, telescopio, pmt)

                    if chuveiro.cut:
                        cutReason[chuveiro.cutReason] += 1
                        self.out.CutReason.append(chuveiro.cutReason)
                        continue

                    '''
                    Survived everything
                    '''
                    survs += 1

                    if datacard.save_hdf5:
                        self.out.saveDataset(datacard.output_file, survs, chuveiro, telescopio)

                    # Observed profiles:
                    self.out.depth_obs.append(chuveiro.DepthMed)
                    self.out.prof_obs.append(chuveiro.ProfMed)

                    self.out.Xfirst.append(chuveiro.xfirst)

                    # Save plots:
                    self.out.path_ErrEgh.append([chuveiro.pathlength,chuveiro.E0err_GH])
                    self.out.Nmax_ErrEgh.append([chuveiro.NmaxGH,chuveiro.E0err_GH])
                    self.out.Xmax_ErrEgh.append([chuveiro.XmaxGH,chuveiro.E0err_GH])
                    self.out.core.append([chuveiro.Xc,chuveiro.Yc,chuveiro.Zc])

                    self.out.corDist.append(chuveiro.coreDist)

                    # Save histograms:
                    self.out.E0Sim.append(chuveiro.getE0eV(chuveiro.EnSim))

                    self.out.E0Rec_GH.append(chuveiro.getE0eV(chuveiro.E0EeV))
                    self.out.E0Err_GH.append(chuveiro.E0err_GH)

                    self.out.E0Rat_GH.append(chuveiro.E0rat_GH)

                    self.out.ChisqNDOFGH.append(chuveiro.ChisqNdofGH)
                    self.out.Chi0.append(chuveiro.chi0)

                    self.out.XmaxSim.append(chuveiro.XmaxSim)
                    self.out.XmaxRec.append(chuveiro.XmaxGH)
                    self.out.NmaxSim.append(chuveiro.NmaxSim)
                    self.out.NmaxRec.append(chuveiro.NmaxGH)

                    self.out.XmaxError.append(chuveiro.XmaxErr)
                    self.out.NmaxError.append(chuveiro.NmaxErr)
                    self.out.ShowerSize.append(chuveiro.ShowerSize)

                    self.out.ZenAng.append(chuveiro.ZenAng)
                    self.out.AziAng.append(chuveiro.AziAng)
                    self.out.Path.append(chuveiro.pathlength)
                    self.out.RpCut.append(chuveiro.Rp)
                    self.out.AveSig.append(chuveiro.AveSignal)
                    self.out.AveSigPe.append(chuveiro.AveSignalPe)
                    self.out.Npmt.append(telescopio.nPMTtr)
                    self.out.NphtPath.append(chuveiro.SumSignal/chuveiro.pathlength)
                    self.out.NpeUsPerDeg.append(chuveiro.SumSignalPeUs/chuveiro.pathlength)
                    self.out.NpePerDeg.append(chuveiro.SumSignalPe/chuveiro.pathlength)
                    self.out.Sinal.append(chuveiro.SumSignal)

            fim = time.time()

            self.out.time = fim - inicio1
            self.out.eff = survs/nSim if nSim != 0 else np.nan
            self.out.nSim = nSim
            self.out.survs = survs
            self.out.nTrigger = nTrigger
            self.out.notInFOV = notInFOV
            self.out.cutReason = cutReason

            bar.close()
            if verbose == 2:
                print(bar)
            if verbose:
                print(f'{file}\n\t{((time.time() - inicio1)/60):.2f} min')

        if verbose:
            print("Finished", time.strftime("%Y-%m-%d %H:%M:%S"), "\n")

        self.out.SaveArrays()
