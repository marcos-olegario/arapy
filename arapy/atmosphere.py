"""
Atmospheric model: Linsley parametrization (US Standard Atmosphere, as
in CORSIKA) and Rayleigh/Mie transmittances, inherited from the original
C++ program (Fly's Eye prescription, Baltrusaitis et al. 1985); the same
expressions appear as Eqs. 1-3 of Prado et al. (2005).
"""

import numpy as np


class Atmosphere:
    """
    Linsley layered atmosphere (CORSIKA/US Standard Atmosphere).

    Provides height <-> vertical depth conversions, density, temperature
    and atmospheric transmittances (Rayleigh and Mie) used in the
    propagation of fluorescence light.
    """

    def __init__(self):
        class AtmLayer:
            def __init__(self, h_min, h_max, a, b, c):
                self.h_min = h_min  # km
                self.h_max = h_max  # km
                self.a = a          # g/cm^2
                self.b = b          # g/cm^2
                self.c = c          # cm

        # Linsley layers: X(h) = a + b*exp(-h/c) (last layer: linear)
        self.layers = [
            AtmLayer(0.0,   4.0,  -186.5562, 1222.6562, 994186.38),
            AtmLayer(4.0,  10.0,   -94.919,  1144.9069, 878153.55),
            AtmLayer(10.0, 40.0,     0.61289, 1305.5948, 636143.04),
            AtmLayer(40.0,100.0,     0.0,     540.1778,  772170.16),
            AtmLayer(100.0,112.8,     0.01128, 1.0,       1.0e9)      # linear
        ]

    def h2dep(self, h_km):
        """
        Vertical atmospheric depth [g/cm^2] at altitude h_km [km].
        Accepts scalar or array.
        """
        h_km = np.asarray(h_km)
        escalar = h_km.ndim == 0

        h_cm = h_km * 1.0e5

        result = np.zeros_like(h_km, dtype=float)

        for i in range(4):
            mask = (h_km >= self.layers[i].h_min) & (h_km < self.layers[i].h_max)
            if np.any(mask):
                result[mask] = self.layers[i].a + self.layers[i].b * np.exp(-h_cm[mask] / self.layers[i].c)

        # Layer 5: linear
        mask_linear = h_km >= self.layers[4].h_min
        if np.any(mask_linear):
            result[mask_linear] = self.layers[4].a - self.layers[4].b * h_cm[mask_linear] / self.layers[4].c

        if escalar:
            return result.item()
        return result

    def dep2h(self, dep):
        """
        Altitude [km] corresponding to the vertical depth dep [g/cm^2]
        (inverse of h2dep in the first 4 Linsley layers). Scalar only.
        """
        if 631.13 < dep <= 20004.65:
            return -9.9419 * np.log((dep + 1.86556E+02) / 1.2227E+03)

        elif 271.70 < dep <= 631.13:
            return -8.7815 * np.log((dep + 9.49190E+01) / 1.1449E+03)

        elif 3.0395 < dep <= 271.70:
            return -6.3614 * np.log((dep - 6.12890E-01) / 1.3056E+03)

        elif 1.2829E-3 < dep <= 3.0395:
            return -7.7217 * np.log(dep / 5.4018E+02)

        else:
            raise ValueError(f"dep2h: depth outside the valid range ({dep} g/cm^2)")

    def GetRho(self, h_km):
        """
        Air density [g/cm^3] at altitude h_km [km]. Accepts scalar or array.
        """
        h_km = np.asarray(h_km)
        escalar = h_km.ndim == 0

        h_cm = h_km * 1.0e5

        result = np.zeros_like(h_km, dtype=float)

        for i in range(4):
            mask = (h_km >= self.layers[i].h_min) & (h_km < self.layers[i].h_max)
            if np.any(mask):
                Xv = self.layers[i].a + self.layers[i].b * np.exp(-h_cm[mask] / self.layers[i].c)
                result[mask] = (Xv - self.layers[i].a) / self.layers[i].c

        # Layer 5: constant
        mask_linear = h_km >= self.layers[4].h_min
        if np.any(mask_linear):
            result[mask_linear] = self.layers[4].b / self.layers[4].c

        if escalar:
            return result.item()
        return result

    def GetTemperature(self, h_km):
        """
        Temperature [K] at altitude h_km [km] (US Standard Atmosphere 1976).
        Accepts scalar or array.
        """
        h_km = np.asarray(h_km)
        escalar = h_km.ndim == 0

        result = np.zeros_like(h_km, dtype=float)

        mask1 = h_km < 11.0
        if np.any(mask1):
            result[mask1] = 288.15 - 6.5 * h_km[mask1]

        mask2 = (h_km >= 11.0) & (h_km < 20.0)
        if np.any(mask2):
            result[mask2] = 216.65

        mask3 = (h_km >= 20.0) & (h_km < 32.0)
        if np.any(mask3):
            result[mask3] = 216.65 + 1.0 * (h_km[mask3] - 20.0)

        mask4 = (h_km >= 32.0) & (h_km < 47.0)
        if np.any(mask4):
            result[mask4] = 228.65 + 2.8 * (h_km[mask4] - 32.0)

        mask5 = h_km >= 47.0
        if np.any(mask5):
            result[mask5] = 270.65

        if escalar:
            return result.item()
        return result

    def GetRayTrans(self, telescopio, z, ang, wl):
        """
        Rayleigh transmittance (molecular scattering) between an emission
        point and the telescope (Baltrusaitis et al. 1985; Prado et al. 2005, Eq. 1).

        Parameters
        ----------
        telescopio : Detector
            Detector object (uses Ztel).
        z : float or ndarray
            Altitude of the emission point [m].
        ang : float or ndarray
            Elevation of the line of sight [rad].
        wl : float or ndarray
            Wavelength [nm].

        Notes
        -----
        Numpy broadcasting is supported (e.g. z[:,None] with wl[None,:]).
        """
        X1 = self.h2dep(telescopio.Ztel / 1000)
        X2 = self.h2dep(z / 1000)

        sin_ang = np.sin(ang)

        return np.exp(-np.abs(X1 - X2) * (400/wl)**4 / (2974 * sin_ang))

    def GetMieTrans(self, telescopio, z, ang):
        """
        Mie transmittance (aerosols) between an emission point and the
        telescope (Baltrusaitis et al. 1985; Prado et al. 2005, Eq. 2),
        exponential model with scale height AScaH and attenuation
        length HALAtSea.

        Parameters
        ----------
        telescopio : Detector
            Detector object (uses Ztel, HALAtSea, AScaH).
        z : float or ndarray
            Altitude of the emission point [m].
        ang : float or ndarray
            Elevation of the line of sight [rad].
        """
        lm = telescopio.HALAtSea   # attenuation length at sea level [m]
        hm = telescopio.AScaH      # aerosol scale height [m]

        sin_ang = np.sin(ang)

        return np.exp((np.exp(-z/hm) - np.exp(-telescopio.Ztel/hm)) * hm / (lm * sin_ang))

    def GetAtenuation(self, telescopio, z, ang, wl):
        """
        Total Rayleigh x Mie transmittance (Baltrusaitis et al. 1985;
        Prado et al. 2005, Eq. 3).
        """
        TRay = self.GetRayTrans(telescopio, z, ang, wl)
        TMie = self.GetMieTrans(telescopio, z, ang)

        return TRay * TMie
