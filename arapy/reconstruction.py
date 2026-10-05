"""
Detector response and longitudinal-profile reconstruction: physical
signal (fluorescence), electronics noise, trigger and Gaisser-Hillas fit.
"""

import warnings

import numpy as np
from scipy.optimize import curve_fit
from scipy.integrate import quad

from .utils import d2rad, rad2d, GaisserHillas, c
from .fluorescence import FluorescenceNpeKakimoto


def Reconstruct(chuveiro, telescopio, rodada, pmt, rng, datacard, atm, fluspc):
    """
    Simulate the detector response and reconstruct the longitudinal profile.

    Steps:

    1. interpolate the CONEX profile (dE/dX and N) at the depths seen
       by each pixel;
    2. compute the fluorescence photoelectrons (FluorescenceNpeKakimoto);
    3. add night-sky and PMT/ADC-chain noise and apply the pixel
       trigger (signal above 4 sigma of the background) and the
       topological event trigger;
    4. if triggered: convert the signal into a profile (dE/dX) and fit
       Gaisser-Hillas, with the calorimetric energy via analytic
       integration of the fit.

    Fills the Shower object with the result (Trigger, NmaxGH, XmaxGH,
    EcalEeV, E0EeV, ...) and the PMT object with the per-pixel signals.
    """
    # First of all check whether the shower is in the FoV:
    if not chuveiro.IsInFoV:
        return

    # Build an interpolation of the profile values (CONEX depths are
    # already in slant depth, see Shower.FillFromConex):
    slantdepth = np.asarray(pmt.slantdepth)
    pmtdEdX = np.interp(slantdepth, chuveiro.depthMid, chuveiro.EnDeposit)
    pmtNpart = np.interp(slantdepth, chuveiro.depth, chuveiro.LongProf)

    # If a PMT looks at a depth before the cascade starts or after it ends,
    # make sure no particles are wrongly extrapolated:
    pmtdEdX[slantdepth < chuveiro.depthMid[0]] = 0.0
    pmtdEdX[slantdepth > chuveiro.depthMid[-1]] = 0.0
    pmtNpart[slantdepth < chuveiro.depth[0]] = 0.0
    pmtNpart[slantdepth > chuveiro.depth[-1]] = 0.0

    pmt.dEdX = pmtdEdX.tolist()
    pmt.Npart = pmtNpart.tolist()

    # --- Photoelectrons in the detector: ---
    NpeFluo = FluorescenceNpeKakimoto(telescopio, pmt, rng, atm, fluspc)
    pmt.NphFluo = NpeFluo

    # The telescope sees only the fluorescence signal
    NPhotoelectronsVis = NpeFluo

    if telescopio.nPMT == 0:
        chuveiro.Trigger = False

    # --- Electronics noise: ---

    # Expected number of sky photoelectrons (BG photoelectrons mean per bin)
    # flux_NSB [ photons / (m^2 * deg^2 * us) ] * [m^2] * pixel_area [deg^2] * [microsecond]
    npeBG = 200.0 * telescopio.mirrorArea * pmt.pixsize**2 * telescopio.intTime * telescopio.telEff

    noiseThr = np.sqrt(npeBG) # baseline threshold
    sigG = np.sqrt(npeBG * 0.4) # Gaussian width
    gainADC = 0.8
    sigADC = np.sqrt(3.3)

    # Generate the noise arrays for all pixels at ONCE
    noisePoisson = rng.poisson(lam=npeBG, size=telescopio.nPMT)
    noiseGaussPMT = rng.normal(loc=0.0, scale=sigG, size=telescopio.nPMT)
    noiseGaussADC = rng.normal(loc=0.0, scale=sigADC, size=telescopio.nPMT)

    # Signal processing (digitization)
    peNet = noisePoisson + noiseGaussPMT - npeBG # averaged to zero
    adc = peNet / gainADC # convert to ADC (analog-to-digital converter)

    # Effect of the loss of digital resolution (truncation to integers)
    digitalSignal = np.trunc(adc + noiseGaussADC).astype(int)
    noiseFinal = digitalSignal * gainADC # convert to photoelectrons

    # The noisy signal observed by the telescope
    NPhotonsMed = NPhotoelectronsVis + noiseFinal

    # Pixel trigger (the 4-sigma cut)
    # Returns a boolean array: True where activated, False where it is noise
    pixelsAct = (NPhotonsMed > 0) & (NPhotonsMed > 4.0 * noiseThr)

    ###############################
    # Event trigger (topology)
    ###############################

    ## Condition 1: at least numPMTact activated pixels in total
    n_ativados = np.sum(pixelsAct)

    telescopio.n_ativados = n_ativados

    ## Condition 2: at least numPMTadj adjacent ones
    tem_adjacentes = False
    for i in range(len(pixelsAct) - datacard.numPMTadj + 1):
        if np.all(pixelsAct[i:i+datacard.numPMTadj]):
            tem_adjacentes = True
            break

    telescopio.tem_adjacentes = int(tem_adjacentes)

    TriggerEvent = (n_ativados >= datacard.numPMTact) and tem_adjacentes
    if TriggerEvent:
        chuveiro.Trigger = True
        telescopio.nPMTtr = int(n_ativados)
    else:
        chuveiro.Trigger = False

    # Store results
    pmt.NPhotonsMedido = NPhotonsMed.tolist()
    pmt.Triggers = pixelsAct.tolist()

    # --- Reconstructing the signal ---
    if chuveiro.Trigger:
        triggerMask = np.array(pmt.Triggers)

        ## Shower kinematics after the trigger:
        r_trig = np.array(pmt.r)[triggerMask]   # triggered PMTs only

        v_in  = r_trig[0]   # vector to the shallowest point (highest elevation)
        v_out = r_trig[-1]    # vector to the deepest point (lowest elevation)

        # Angle between the two extreme vectors
        cos_path = np.dot(v_in, v_out) / (np.linalg.norm(v_in) * np.linalg.norm(v_out))
        cos_path = np.clip(cos_path, -1.0, 1.0)
        path_rad = np.arccos(cos_path)
        path_deg = rad2d(path_rad)

        camivisto  = np.linalg.norm(v_in - v_out)      # meters
        tempovisto = (camivisto / c) * 1e6             # us
        speed      = path_deg / tempovisto             # degrees/us
        speed_rad  = path_rad / tempovisto             # rad/us

        # Overwrite the values computed in GetShowerInField (which used all PMTs)
        chuveiro.pathlength   = path_deg
        chuveiro.speed        = speed
        chuveiro.speedrad     = speed_rad
        chuveiro.time         = tempovisto
        chuveiro.firstPMTelev = np.array(pmt.elev)[triggerMask][0]
        chuveiro.lastPMTelev  = np.array(pmt.elev)[triggerMask][-1]

        # Signal reconstruction
        depthVis = np.array(pmt.depth)[triggerMask] / np.cos(d2rad(chuveiro.ZenAng)) # slant
        signalMed = np.array(pmt.NPhotonsMedido)[triggerMask]
        factors = np.array(pmt.factor)[triggerMask]
        timeMed = np.array(pmt.TimeSeen)[triggerMask]

        # Conversion of the measured signal into a profile proportional
        # to dE/dX, ready for the GH fit.
        fitProf = signalMed / factors

        # Three scales coexist here, and the name states which one:
        #   *Signal     : photoelectrons inside ONE integration window [pe].
        #                 This is the scale the trigger works on, since the
        #                 background is computed for the same window.
        #   *SignalPeUs : signal x time seen by the pixel [pe.us]
        #                 (intermediate, no direct physical meaning).
        #   *SignalPe   : TOTAL integrated photoelectrons [pe] -- the
        #                 quantity the experiments publish (e.g. the TA
        #                 "Npe/degree" of Abu-Zayyad et al. 2013, Sec. 3).
        # The ratio between the last and the first is TimeSeen/intTime,
        # typically 3-35 depending on the geometry.
        chuveiro.SumSignal     = np.sum(signalMed)
        chuveiro.SumSignalPeUs = np.sum(signalMed * timeMed)
        chuveiro.SumSignalPe   = np.sum(signalMed * timeMed / telescopio.intTime)

        chuveiro.AveSignal     = np.mean(signalMed)
        chuveiro.AveSignalPeUs = np.mean(signalMed * timeMed)
        chuveiro.AveSignalPe   = np.mean(signalMed * timeMed / telescopio.intTime)

        chuveiro.DepHiEle = depthVis[0] # Depth of Highest Elevation Hit
        chuveiro.DepLoEle = depthVis[-1] # Depth of Lowest Elevation Hit
        chuveiro.ShowerSize = abs(depthVis[-1] - depthVis[0])

        chuveiro.ProfMed = fitProf
        chuveiro.DepthMed = depthVis

        if len(fitProf) >= datacard.numPMTtrig:

            # Gaisser-Hillas with the lambda of the datacard
            lam = datacard.lambdaGH

            def GH(X, nmax, xmax, x0):
                return GaisserHillas(X, nmax, xmax, x0, lam)

            freeX0 = datacard.x0 is None

            # Estimated error: 20% of the measured signal, which is always
            # > 0 in triggered pixels.
            estimErrors = 0.2 * fitProf

            # Initial guesses
            idx_max = np.argmax(fitProf)
            NmaxTry = fitProf[idx_max]
            XmaxTry = 750

            if freeX0:
                X0Try = 40.0 if telescopio.IsHIRESI else -60.0 # Logic kept from the original
                limits = (
                    [0, 680 if telescopio.IsHIRESI else 0, -200],                           # Min [nmax, xmax, x0]
                    [np.inf, 900 if telescopio.IsHIRESI else 2200, min(XmaxTry-10, 200)]    # Max [nmax, xmax, x0]
                )
                # Clamping: force the initial guesses inside the SciPy bounds
                # The 0.1 factor keeps the value away from the exact edge
                X0Try = np.clip(X0Try, limits[0][2] + 0.1, limits[1][2] - 0.1)

            else:
                limits = (
                    [0, 0],             # Min [nmax, xmax]
                    [np.inf, 2200]      # Max [nmax, xmax]
                )

            # Clamping: force the initial guesses inside the SciPy bounds
            # The 0.1 factor keeps the value away from the exact edge
            XmaxTry = np.clip(XmaxTry, limits[0][1] + 0.1, limits[1][1] - 0.1)

            # Chi-square minimization (equivalent to MINUIT)
            try:
                if freeX0:
                    paramsOpt, cov = curve_fit(
                        GH,
                        depthVis,
                        fitProf,
                        p0=[NmaxTry, XmaxTry, X0Try],
                        sigma=estimErrors,
                        absolute_sigma=True,
                        bounds=limits,
                        maxfev=3000
                    )
                    NmaxFit, XmaxFit, X0Fit = paramsOpt
                else:
                    X0Fit = datacard.x0

                    def GaisserHillas2Params(depth, nmax, xmax):
                        return GH(depth, nmax, xmax, X0Fit)

                    paramsOpt, cov = curve_fit(
                        GaisserHillas2Params,
                        depthVis,
                        fitProf,
                        p0=[NmaxTry, XmaxTry],
                        sigma=estimErrors,
                        absolute_sigma=True,
                        bounds=limits,
                        maxfev=3000
                    )
                    NmaxFit, XmaxFit = paramsOpt

                # Manual computation of the chi-square
                TheoProf = GH(depthVis, NmaxFit, XmaxFit, X0Fit)
                chisq = np.sum(((fitProf - TheoProf) / estimErrors) ** 2)
                ndof = len(fitProf) - len(paramsOpt)

                ########################################################################
                # HIRESI prescription: double fit with Xmax fixed on a grid
                # (only with free X0)
                if telescopio.IsHIRESI and freeX0:
                    chisq_tries = []
                    params_tries = []

                    # Base index for the 35 g/cm^2 grid starting at 680
                    base_k = int((XmaxFit - 680) / 35) if XmaxFit >= 680 else 0

                    for k in range(2):
                        fixed_xmax = 680 + 35 * (base_k + k)

                        # Closure (wrapper) to fix Xmax and expose only nmax and x0 to curve_fit
                        def GH_fixed_xmax(X, nmax, x0):
                            return GH(X, nmax, fixed_xmax, x0)

                        bounds_fixed = (
                            [0, -200],      # Lower bounds for [nmax, x0]
                            [np.inf, 200]   # Upper bounds for [nmax, x0]
                        )

                        try:
                            paramsOpt_fixed, _ = curve_fit(
                                GH_fixed_xmax,
                                depthVis,
                                fitProf,
                                p0=[NmaxFit, X0Fit], # Start from the best values of the free fit
                                sigma=estimErrors,
                                absolute_sigma=True,
                                bounds=bounds_fixed,
                                maxfev=3000
                            )

                            nmax_f, x0_f = paramsOpt_fixed

                            # Recompute the chi-square for the constrained fit
                            TheoProf_fixed = GH(depthVis, nmax_f, fixed_xmax, x0_f)
                            chisq_fixed = np.sum(((fitProf - TheoProf_fixed) / estimErrors) ** 2)

                            chisq_tries.append(chisq_fixed)
                            params_tries.append((nmax_f, fixed_xmax, x0_f))

                        except Exception:
                            # If the fit fails for this grid bin, penalize heavily
                            chisq_tries.append(np.inf)
                            params_tries.append((None, None, None))

                    # Select the best fit following the original heuristic
                    best_fit_idx = 0
                    if chisq_tries[1] < chisq_tries[0] and params_tries[1][1] < 900:
                        best_fit_idx = 1

                    # If at least one of the fits converged, update the final parameters
                    if chisq_tries[best_fit_idx] != np.inf:
                        NmaxFit, XmaxFit, X0Fit = params_tries[best_fit_idx]
                        chisq = chisq_tries[best_fit_idx]

                        # With Xmax fixed we lose one degree of freedom in the fit
                        ndof = len(fitProf) - 2

                ########################################################################

                chisq_ndof = chisq / ndof if ndof > 0 else 9999.0

                XmaxErr = (XmaxFit - chuveiro.XmaxSim) / chuveiro.XmaxSim
                NmaxErr = (NmaxFit - chuveiro.dEdXMaxSim) / chuveiro.dEdXMaxSim

                chuveiro.FitGH = True
                chuveiro.NmaxGH = NmaxFit
                chuveiro.XmaxGH = XmaxFit
                chuveiro.X0GH = X0Fit
                chuveiro.ChisqGH = chisq
                chuveiro.NdofGH = ndof
                chuveiro.ChisqNdofGH = chisq_ndof
                chuveiro.XmaxErr = XmaxErr
                chuveiro.NmaxErr = NmaxErr

                if depthVis[0] < XmaxFit < depthVis[-1]:
                    chuveiro.XmaxVis = True
                else:
                    chuveiro.XmaxVis = False

                # Analytic integration (calorimetric energy)
                LimInferior = max(0.0, chuveiro.X0GH)
                try:
                    areaIntegral, _ = quad(
                        GH,
                        LimInferior,
                        10000.0,
                        args=(NmaxFit, XmaxFit, chuveiro.X0GH),
                        limit=100
                    )
                except Exception as e:
                    warnings.warn(f"Integration of the GH fit failed: {e}")
                    areaIntegral = 0.0

                if areaIntegral > 0:

                    EcalEeV = areaIntegral/1e9

                    # Missing-energy correction (Abbasi et al. 2023)
                    EcalE0 = -0.5717 + 0.1416 * np.log10(EcalEeV*1e18) - 0.003328 * (np.log10(EcalEeV*1e18))**2
                    E0EeV = EcalEeV/EcalE0
                else:
                    # Protection against negative/complex energies in bad fits
                    EcalEeV = 0.0
                    E0EeV = 0.0

                # Relative energy errors:
                E0err_GH = (E0EeV - chuveiro.EnSim) / chuveiro.EnSim
                E0rat_GH = E0EeV / chuveiro.EnSim

                chuveiro.EcalEeV = EcalEeV
                chuveiro.E0EeV = E0EeV
                chuveiro.UsdPointsGH = len(fitProf)
                chuveiro.E0err_GH = E0err_GH
                chuveiro.E0rat_GH = E0rat_GH

            except Exception as e:
                # If the fit does not converge (typical of very noisy events)
                warnings.warn(
                    f"GH fit failed ({e}); initial guesses: "
                    f"Nmax={NmaxTry} (CONEX {chuveiro.dEdXMaxSim}), "
                    f"Xmax={XmaxTry} (CONEX {chuveiro.XmaxSim})"
                )
                chuveiro.FitGH = False

        else:
            chuveiro.FitGH = False
            chuveiro.ChisqNdofGH = 9999.0
