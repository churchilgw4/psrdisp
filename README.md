# `PSRDISP`: A novel dispersive noise modeling framework
A single-pulsar noise analysis package to model dispersive processes using high-precision epoch-wise Dispersion Measure time series.

Currently supported models:
1. DM Taylor series (up to second order)
2. SW Taylor series
3. DM noise -- GP implementation with powerlaw kernel
4. SW noise -- GP implementation with powerlaw kernel
5. White noise -- up to epoch-wise quadrature variation to uncertainties

## Usage
```
python3 psrdisp.py --psrname PSR --parfile ./sample/sample.par --dmfile ./sample/sample.dms.txt
                   --paramfile ./sample/sample.pars.txt --timfile ./sample/sample.tim
                   --truthsfile ./sample/sample.truths.txt --outdir ./sample/ --sampler emcee
                   --Nsamples 1000 --DMNharm 100 --plotchains --plot_realisations --threshold 5.0
                   --add_dms --n_realisations 20
```

## Citation
```
@ARTICLE{2027JHEAp..5500722D,
       author = {{Dwivedi}, Churchil and {Susobhanan}, Abhimanyu},
        title = "{PSRDISP: A novel approach to modeling dispersive processes in single-pulsar noise analysis using epoch-wise dispersion measures}",
      journal = {Journal of High Energy Astrophysics},
     keywords = {Pulsars, Interstellar medium, Pulsar timing arrays, Gravitational waves, Instrumentation and Methods for Astrophysics, High Energy Astrophysical Phenomena},
         year = 2027,
        month = jan,
       volume = {55},
          eid = {100722},
        pages = {100722},
          doi = {10.1016/j.jheap.2026.100722},
archivePrefix = {arXiv},
       eprint = {2607.12609},
 primaryClass = {astro-ph.IM},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2027JHEAp..5500722D},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

## Requirements
```
NUMPY
ASTROPY
PINT-PULSAR
MATPLOTLIB
SCIPY
EMCEE
CORNER
POCOMC

```
