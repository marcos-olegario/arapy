"""
Shower geometry relative to the telescope: core/direction sampling,
intersection of the axis with the field of view and projection of the
track onto the pixels.
"""

import numpy as np

from .utils import d2rad, rad2d, c

# Preshower: gamma-ray primaries (CORSIKA code) above this energy [EeV]
# keep the CONEX arrival direction (see SetShowerGeometry).
GAMMA_ID = 1
PRESHOWER_EEV = 10.0


def RandZenAng(ThetaMin, ThetaMax, rng):
    """
    Draw a zenith angle [degrees] with distribution ~ sin(2*theta)
    (isotropic flux on a flat area; Allkofer, "Introduction to Cosmic
    Radiation", p. 27).
    """
    thmin = d2rad(ThetaMin)
    thmax = d2rad(ThetaMax)

    CT1 = np.sin(thmin)**2
    CT2 = np.sin(thmax)**2
    CTT = np.sqrt(1.0 - rng.random()*(CT2 - CT1) - CT1)

    THETAP = np.arccos(CTT)

    return rad2d(THETAP)


def SetShowerGeometry(chuveiro, telescopio, rodada, rng):
    """
    Draw the core position (ring RMinGen-RMaxGen in the telescope plane)
    and the shower direction (zenith up to ZenMaxGen, uniform azimuth),
    filling rCore, nHat, ZenAng and AziAng in the Shower object.

    For gamma-ray primaries with E >= PRESHOWER_EEV, only the core is
    drawn: the direction is kept as simulated by CONEX, since the
    preshower effect makes the profile depend on the direction relative
    to the geomagnetic field.
    """
    raio_max = rodada.RMaxGen
    raio_min = rodada.RMinGen
    zen_max = rodada.ZenMaxGen

    phi = 2 * np.pi * rng.random()

    r2_min = raio_min**2
    r2_max = raio_max**2
    raio = np.sqrt(r2_min + rng.random() * (r2_max - r2_min))

    xc, yc, zc = raio*np.cos(phi), raio*np.sin(phi), telescopio.Ztel

    chuveiro.Xc = xc
    chuveiro.Yc = yc
    chuveiro.Zc = zc

    if rodada.PrimType == GAMMA_ID and chuveiro.EnSim >= PRESHOWER_EEV:
        # Keep the CONEX direction.
        zen, azi = chuveiro.ZenSim, chuveiro.AziSim
    else:
        # Draw a new direction for the shower:
        zen, azi = RandZenAng(0, zen_max, rng), 360*rng.random()

    chuveiro.ZenAng = zen
    chuveiro.AziAng = azi

    zenr = d2rad(zen)
    azir = d2rad(azi)

    chuveiro.rCore = np.array([xc, yc, zc])

    nHat = np.array([np.sin(zenr)*np.cos(azir), np.sin(zenr)*np.sin(azir), np.cos(zenr)])
    chuveiro.nHat = nHat


def isIN(chuveiro, telescopio, l):
    """
    True if the axis point r = rCore + l*nHat is inside the telescope
    field of view (above the plane, within MaxDist and within the
    elevation and azimuth limits).
    """
    rt = telescopio.rTel
    rc = chuveiro.rCore
    n = chuveiro.nHat
    maxDist = telescopio.MaxDist
    amin = d2rad(telescopio.ElevMin)
    amax = d2rad(telescopio.ElevMax)
    bmin = d2rad(telescopio.AziMin)
    bmax = d2rad(telescopio.AziMax)

    Robs = (rc-rt)+n*l # relative to the telescope, not the origin

    # Constraint 1: the point must be above the telescope plane (z > 0)
    if Robs[2] <= 0:
        return False

    # Constraint 2: maximum-range sphere (Rmax)
    r_dist = np.linalg.norm(Robs) # radial distance (r)
    if r_dist > maxDist:
        return False

    # Constraint 3: elevation cones
    alpha = np.arcsin(Robs[2] / r_dist)
    if not (amin <= alpha <= amax):
        return False

    # Constraint 4: azimuthal limits (planes)
    # arctan2 handles the quadrants correctly, returning between -pi and pi.
    beta = np.arctan2(Robs[1], Robs[0])

    # Convert to the [0, 2pi) convention to simplify the comparison
    if beta < 0:
        beta += 2 * np.pi

    # Does the FoV cross zero (North)? E.g. telescope looking from 350 to 10 degrees.
    if bmin < bmax:
        if not (bmin <= beta <= bmax):
            return False
    else: # FoV crosses North
        if not (beta >= bmin or beta <= bmax):
            return False

    return True


def lPMT(r_in, lin, chi, theta):
    """
    Position l along the shower axis seen under angle chi from the FoV
    entry point (analytic solution of the telescope-entry-point triangle;
    theta is the internal angle at Rin).
    """
    return lin - r_in * (np.sin(chi) / np.sin(chi + theta))


def solveQuadratic(A, B, C):
    """
    Real roots of A*x^2 + B*x + C (empty list if delta < 0; handles A ~ 0).
    """
    delta = B**2 - 4 * A * C

    if delta < 0:
        return []

    if np.isclose(A, 0):
        return [-C/B] if not np.isclose(B, 0) else []

    sqrt_delta = np.sqrt(delta)
    return [(-B - sqrt_delta)/(2*A), (-B + sqrt_delta)/(2*A)]


def GetShowerInField(chuveiro, telescopio, pmt, atm):
    """
    Intersection of the shower axis with the field of view and projection
    of the track onto the pixels.

    Computes the impact parameter Rp and the entry/exit points of the
    axis in the FoV (solving for the elevation, azimuth and maximum
    distance limits); if the shower is visible (IsInFoV), splits the
    track into pixels of size pmt.pixsize and fills, per pixel: physical
    length seen, distance, elevation, depth (vertical and slant) and
    position l along the axis. Also computes pathlength, chi0 (angle
    between the axis and the core-telescope line) and the apparent
    angular speed of the track.
    """
    rt = telescopio.rTel
    rc = chuveiro.rCore
    n = chuveiro.nHat

    zen = d2rad(chuveiro.ZenAng) # Here, always in radians
    azi = d2rad(chuveiro.AziAng) # Here, always in radians

    maxDist = telescopio.MaxDist
    amin = d2rad(telescopio.ElevMin)
    amax = d2rad(telescopio.ElevMax)
    bmin = d2rad(telescopio.AziMin)
    bmax = d2rad(telescopio.AziMax)

    # helpers:
    dr = rc - rt
    dr_sq = dr.dot(dr)
    dr_dot_n = dr.dot(n)

    chuveiro.coreDist = np.linalg.norm(dr)

    # Shower-detector plane (SDP): the cross product below IS the normal
    # of the plane containing the shower axis and the telescope, and its
    # norm is the impact parameter (|n| = 1).
    sdp_n = np.cross(dr, n)
    Rp = np.linalg.norm(sdp_n)
    chuveiro.Rp = Rp

    if Rp > 0:
        sdp_n = sdp_n / Rp
        chuveiro.nSDP = sdp_n
        # Inclination of the plane with respect to the vertical:
        #   vertical plane -> 0 deg ; horizontal plane -> 90 deg

        chuveiro.SDPangle = np.degrees(np.arcsin(abs(sdp_n[2])))
    else:
        chuveiro.nSDP = None
        chuveiro.SDPangle = np.nan

    if Rp > maxDist:
        chuveiro.IsInFoV = False
        chuveiro.RinField = 0
        chuveiro.RoutField = 0
        chuveiro.linField = 0
        chuveiro.loutField = 0
        return None

    # Solve for beta:
    lbmin = (dr[1] * np.cos(bmin) - dr[0] * np.sin(bmin) )
    lbmin = lbmin / (np.sin(zen)*np.cos(azi)*np.sin(bmin) - np.sin(zen)*np.sin(azi)*np.cos(bmin))
    lbmax = (dr[1] * np.cos(bmax) - dr[0] * np.sin(bmax) )
    lbmax = lbmax / (np.sin(zen)*np.cos(azi)*np.sin(bmax) - np.sin(zen)*np.sin(azi)*np.cos(bmax))

    # Solve for alpha:
    A_min = n[2]**2 - np.sin(amin)**2
    B_min = -2 * dr_dot_n * np.sin(amin)**2
    C_min = -dr_sq * np.sin(amin)**2
    lamin = solveQuadratic(A_min, B_min, C_min)

    A_max = n[2]**2 - np.sin(amax)**2
    B_max = -2 * dr_dot_n * np.sin(amax)**2
    C_max = -dr_sq * np.sin(amax)**2
    lamax = solveQuadratic(A_max, B_max, C_max)

    # Solve for Rmax
    lrmax = solveQuadratic(1.0, 2*dr_dot_n, dr_sq - maxDist**2)

    # Boundary candidates of the visible stretch along the axis:
    ls = np.sort([lbmin] + [lbmax] + lamin + lamax + lrmax)
    ls = np.unique(ls[ls >= 0.0])

    ls_ = []

    for i in range(len(ls)-1):
        lmid = (ls[i] + ls[i+1])/2
        if isIN(chuveiro, telescopio, lmid):
            ls_.append(ls[i])
            ls_.append(ls[i+1])

    if len(ls_) == 0 or abs(int(np.max(ls_) - np.min(ls_))) < 10:
        chuveiro.IsInFoV = False
        chuveiro.RinField = 0
        chuveiro.RoutField = 0
        chuveiro.linField = 0
        chuveiro.loutField = 0
        return None

    lin, lout = np.max(ls_), np.min(ls_)
    Rout = rc+n*lout
    Rin = rc+n*lin

    chuveiro.IsInFoV = True
    chuveiro.RinField = Rin
    chuveiro.RoutField = Rout
    chuveiro.linField = lin
    chuveiro.loutField = lout

    # Fill the PMTs:
    Robs_in = Rin - rt
    Robs_out = Rout - rt
    r_in = np.linalg.norm(Robs_in)
    r_out = np.linalg.norm(Robs_out)

    # Total path length
    cos_chi = Robs_in.dot(Robs_out) / (r_in * r_out)
    cos_chi = np.clip(cos_chi, -1.0, 1.0)  # avoid arccos(>1)
    chi = np.arccos(cos_chi)

    # Total internal angle theta
    cos_theta = Robs_in.dot(n) / r_in
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))

    # Pixel size (radians)
    pixfov = np.radians(pmt.pixsize)

    # number of whole pixels
    npix = int(np.floor(chi / pixfov))

    for k in range(npix + 1):
        # Angular limits of pixel k
        chi_i = k * pixfov
        chi_f = min((k + 1) * pixfov, chi) # The last pixel may be < 1 degree

        if chi_i >= chi:
            break

        chi_mid = (chi_i + chi_f) / 2.0

        # Physical length inside the pixel (pmt_length in meters)
        l_i = lPMT(r_in, lin, chi_i, theta)
        l_f = lPMT(r_in, lin, chi_f, theta)
        pmt_length = l_i - l_f # in meters

        Li = rc+l_i*n - rt
        Lf = rc+l_f*n - rt
        cos_angpath = np.dot(Li, Lf) / (np.linalg.norm(Li) * np.linalg.norm(Lf))
        angulo_path = np.arccos(np.clip(cos_angpath, -1.0, 1.0))

        pmt_length_rad = angulo_path # in rad

        # Properties at the CENTER of the pixel (for attenuation and Gaisser-Hillas)
        l_mid = lPMT(r_in, lin, chi_mid, theta)
        Rmid = rc+l_mid*n - rt # Relative to the telescope
        r_mid = np.linalg.norm(Rmid)

        # Elevation (Ang)
        z_mid = Rmid[2]
        elev_mid = np.arcsin(z_mid / r_mid)

        # Depth
        dep_mid = atm.h2dep((z_mid+rt[2])/1000)
        slant_dep_mid = dep_mid / np.cos(d2rad(chuveiro.ZenAng))

        # Fill the class
        pmt.length.append(pmt_length)
        pmt.lengthdeg.append(rad2d(pmt_length_rad))
        pmt.dist.append(r_mid)
        pmt.r.append(Rmid)
        pmt.elev.append(rad2d(elev_mid)) # PMT->SetAng(), in deg
        pmt.lmid.append(l_mid)
        pmt.depth.append(dep_mid)
        pmt.slantdepth.append(slant_dep_mid)

    telescopio.nPMT = npix + 1

    # Shower kinematics:
    Rshower_axis = Robs_in - Robs_out
    r_shower_axis = np.linalg.norm(Rshower_axis)

    cos_chi0 = Rshower_axis.dot(dr) / (r_shower_axis * np.linalg.norm(dr))
    chi0 = np.arccos(np.clip(cos_chi0, -1.0, 1.0) ) # Psi

    tempo_visto_us = (r_shower_axis / c) * 1e6 # microseconds
    speed = rad2d(chi) / tempo_visto_us # degrees/microsecond
    speed_rad = chi / tempo_visto_us # rad/microsecond

    chuveiro.pathlength = rad2d(chi)
    chuveiro.chi0 = rad2d(chi0)
    
    # Psi in the convention of Abbasi et al. (2016/2023): angle, in the
    # SDP, between the shower axis and the direction from the shower
    # impact (core) to the telescope. chi0 above is measured between the
    # UPWARD axis and the telescope->core line, hence Psi = 180 - chi0.
    # Receding showers (moving away from the telescope) have Psi < 90.
    chuveiro.Psi = 180 - chuveiro.chi0
    chuveiro.speed = speed
    chuveiro.speedrad = speed_rad

    # Minimum viewing angle (Abbasi et al. 2016, Table 1 cut #5): smallest
    # angle between the shower axis and any line of sight to the visible
    # track. The viewing angle at track point Robs(l) = dr + n*l equals
    # angle(n, Robs(l)), whose cosine grows monotonically with l
    # (d cos/dl = Rp^2/|Robs|^3 > 0), so the minimum over the visible
    # stretch is at the topmost point l = lin -- i.e. `theta` above.
    chuveiro.MinViewAng = rad2d(theta)
