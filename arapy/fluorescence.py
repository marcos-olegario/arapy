"""
Fluorescence light emission and detection: Kakimoto et al. (1996) two-term
yield, Bunner (1967) emission spectrum (packaged in fluspc.dat) and
conversion of the dE/dX profile into photoelectrons per pixel.

The atmospheric transmittance is averaged over the whole emission
spectrum (deterministic), instead of evaluated at a single sampled
wavelength. Dry air, no temperature dependence of the collisional cross
section, grey telescope efficiency.
"""

import numpy as np
from importlib import resources
from .utils import c

# Physical constants
dEdX14 = 1.659e-3 # GeV/(g/cm^2) - energy loss of a 1.4 MeV electron (Nagano et al. 2003)

# Kakimoto et al. (1996) two-term constants, 300-400 nm
# (de Souza, Medina-Tanco & Ortiz 2004, Tab. 1)
A1 = 89.0  # m^2/kg
A2 = 55.0  # m^2/kg
B1 = 1.85  # m^3 kg^-1 K^-1/2
B2 = 6.50  # m^3 kg^-1 K^-1/2


def YieldKakimoto(rho, T):
    """
    Total fluorescence yield [photons/m] for a 1.4 MeV electron, at
    density rho [kg/m^3] and temperature T [K], integrated over 300-400 nm.

    Term 1: 2P bands (weak quenching); term 2: 1N 391 nm band (strong quenching).
    """
    sqrtT = np.sqrt(T)
    term1 = rho * A1 / (1.0 + rho * B1 * sqrtT)
    term2 = rho * A2 / (1.0 + rho * B2 * sqrtT)

    return term1 + term2


def ReadFluoSpectrum(file_name='fluspc.dat'):
    """
    Load the packaged fluorescence spectrum and return its normalized
    CDF: column 0 = wavelengths [nm], column 1 = cumulative probability.
    """
    file_path = resources.files("arapy").joinpath(file_name)

    with file_path.open('r') as f:
        fluspc_raw = np.loadtxt(f)

    fluspc = np.zeros_like(fluspc_raw)
    fluspc[:, 0] = fluspc_raw[:, 0] # Keep the wavelengths untouched
    fluspc[:, 1] = np.cumsum(fluspc_raw[:, 1]) / np.sum(fluspc_raw[:, 1]) # normalized CDF

    return fluspc


def SpectrumWeights(fluspc):
    """
    Recover the normalized emission spectrum from the CDF returned by
    ReadFluoSpectrum().

    Returns
    -------
    (lam, w) : wavelengths [nm] and weights, with sum(w) = 1.
    """
    lam = fluspc[:, 0]
    cdf = fluspc[:, 1]
    w = np.concatenate(([cdf[0]], np.diff(cdf)))

    return lam, w / w.sum()


def FluorescenceNpeKakimoto(telescopio, pmt, rng, atm, fluspc, dEdX_ref=dEdX14):
    """
    Fluorescence photoelectrons per pixel.

    For each pixel: local yield (Kakimoto) scaled by dE/dX, atmospheric
    attenuation averaged over the emission spectrum, mirror solid angle,
    optical efficiency and the ratio between the integration window and
    the time during which the pixel sees the track.

    rng is not used (the computation is deterministic).
    dEdX_ref is the reference electron energy loss [GeV/(g/cm^2)].

    Fills pmt.TimeSeen, pmt.factor (dE/dX -> pe conversion factor, used
    later to reconstruct the profile) and pmt.NpeFluo.

    Returns
    -------
    ndarray
        Array [nPMT] -- fluorescence photoelectrons.
    """
    pmt_z = np.array(pmt.r)[:,2] + telescopio.Ztel # Relative to the origin
    ang = np.deg2rad(np.array(pmt.elev))
    pmt_dist = np.array(pmt.dist)
    pmt_length = np.array(pmt.length)

    # Fluorescence yield of the reference electron
    h_km = pmt_z/1000
    rho = atm.GetRho(h_km) * 1000 # g/cm^3 -> kg/m^3
    T = atm.GetTemperature(h_km)
    Y = YieldKakimoto(rho, T)

    # Attenuation averaged over the spectrum: [nPMT, nlambda] @ [nlambda]
    lam, w = SpectrumWeights(fluspc)
    atenua = atm.GetAtenuation(telescopio, pmt_z[:,None], ang[:,None], lam) @ w

    # Solid angle - fraction of the sphere covered by the mirror area
    solAng = telescopio.mirrorArea / (4.0 * np.pi * pmt_dist**2) # area/(4*pi*r^2)

    # Time seen (in microseconds)
    pmtTempoVisto = 1e6 * pmt_length / c
    pmt.TimeSeen = pmtTempoVisto.tolist()

    # (length) * (attenuation) * (Solid Angle) * (Tel. efficiencies) * (Integration Time) / (time seen)
    fator = (pmt_length * atenua * solAng * telescopio.telEff * telescopio.intTime) / pmtTempoVisto
    convfactor = fator * Y / dEdX_ref

    # Store the results back into the class
    pmt.factor = convfactor.tolist()

    # total observed light
    NpeFluo = np.array(pmt.dEdX) * convfactor # Number of photoelectrons generated in the PMT by the shower
    pmt.NpeFluo = NpeFluo.tolist()

    return NpeFluo
