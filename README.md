# ARAPy

**ARAPy** simulates the response of fluorescence telescopes
to longitudinal profiles of extensive air
showers simulated with CONEX.

The package emulates the full detection chain: shower geometry sampling,
fluorescence light emission (Kakimoto et al. yield, Bunner spectrum),
atmospheric propagation (Rayleigh + Mie), telescope optics, night-sky
background and electronics noise, pixel and topological triggers, and
Gaisser-Hillas reconstruction of the longitudinal profile with
calorimetric energy estimation. Cherenkov light is not included.

## Installation

Requires Python ≥ 3.12.

```bash
git clone https://github.com/marcos-olegario/arapy.git
cd arapy
pip install -e .
```

Dependencies (installed automatically): `numpy`, `scipy`, `h5py`,
`uproot`, `tqdm`.

## Quick start

```python
from arapy import DataCard, Simulation

datacard = DataCard(
    files=["showers.root"],   # CONEX .root files
    nRep=10,                  # geometries drawn per shower
    save_hdf5=True,           # optional: also write the events to disk
    output_file="events.h5",
)

sim = Simulation()
sim.Run(datacard)                   # default cuts: trigger + converged GH fit

out = sim.out                 # arrays of the surviving events
print(out.nSim, out.nTrigger, out.nSurv)  # simulated events, triggered events, and reconstructed events passing all quality cuts
print(out.XmaxRec, out.E0Rec_GH)   # reconstructed Xmax, log10(E0/eV)
```

After the run, `sim.out` always holds numpy arrays with one entry per
surviving event (e.g. `E0Sim`, `E0Rec_GH`, `XmaxSim`, `XmaxRec`,
`NmaxRec`, `ZenAng`, `RpCut`, `Npmt`, `ChisqNDOFGH`) and the observed
profiles `depth_obs` / `prof_obs`. With `save_hdf5=False` nothing is
written to disk.

### Quality cuts

`Run` accepts an optional function `ApplyCuts(shower, telescope, pmt)`
that sets `shower.cut` (bool) and `shower.cutReason` (int in `[0, 20)`,
counted in `sim.out.cutReason`). It must reject events without trigger
or without a Gaisser-Hillas fit:

```python
def ApplyCuts(shower, telescope, pmt):
    if not (shower.Trigger and shower.FitGH):
        shower.cut, shower.cutReason = True, 0
    elif shower.ChisqNdofGH > 10 or not shower.XmaxVis:
        shower.cut, shower.cutReason = True, 1
    else:
        shower.cut = False

sim.Run(datacard, ApplyCuts)
```

## Main parameters

All parameters are set in `DataCard` (see its docstring for the full
list). Lengths in m, angles in degrees.

| Parameter                | Default     | Description                                              |
| ------------------------ | ----------- | -------------------------------------------------------- |
| `Xtel, Ytel, Ztel`       | 0, 0, 1597  | telescope position                                       |
| `ElevMin, ElevMax`       | 3, 31       | field of view in elevation                               |
| `AziMin, AziMax`         | 0, 360      | field of view in azimuth                                 |
| `MaxDist`                | 40e3        | maximum observation distance                             |
| `RMinGen, RMaxGen`       | 0, 35e3     | core generation ring                                     |
| `ZenMaxGen`              | 60          | maximum generated zenith angle                           |
| `mirrorArea`, `telEff`   | 5.1 m², 0.2 | mirror area, optical efficiency                          |
| `pixsize`, `intTime`     | 1, 0.1 µs   | pixel size, integration window                           |
| `AScaH`, `HALAtSea`      | 1200, 14000 | aerosol scale height and attenuation length at sea level |
| `numPMTact`, `numPMTadj` | 5, 3        | trigger: active / adjacent pixels                        |
| `x0`, `lambdaGH`         | -100, 70    | GH fit: fixed X0 (`None` = free) and λ [g/cm²]           |

## HDF5 output

Each surviving event is stored as a group `run_NNNNNNN` containing
scalar attributes (geometry, signal and reconstruction quantities, e.g.
`Rp`, `zenAng`, `chi0`, `nPMTtr`, `ChisqNdofGH`, `EnSim`, `E0EeV`,
`EcalEeV`) and the datasets:

- `depth_med`, `long_med`: observed (reconstructed) profile;
- `depth_sim`, `long_sim`: CONEX energy-deposit profile;
- `params_rec`, `params_sim`: GH parameters (dE/dX max, Xmax, X0);
- `core`: core position.

## Lineage

ARAPy is a Python modernization of the C++/ROOT program of V. de Souza
et al. ([Phys. Rev. D **72** (2005) 103009](https://doi.org/10.1103/PhysRevD.72.103009)),
extended by W. Carvalho Jr. et al.
([Astropart. Phys. **28** (2007) 89](https://doi.org/10.1016/j.astropartphys.2007.04.010)).
The fluorescence spectrum in `arapy/fluspc.dat` is the one measured by
A. N. Bunner (PhD thesis, Cornell University, 1967), as used in the
original program.

## Citation

If you use ARAPy, please cite the accompanying paper:

> M. Olegario and V. de Souza, *ARAPy: an open-source tool for simulating
> the fluorescence detector response to air-shower longitudinal
> profiles* (in preparation).

## Acknowledgments

This work was supported by the São Paulo Research Foundation (FAPESP),
grants No. 2021/01089-1 and 2024/08946-5.

## License

MIT — see [LICENSE](LICENSE).
