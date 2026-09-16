import numpy as np
import astropy.units as u
import argparse
import time
import sys

from disputils import *
from pint.models.model_builder import get_model
from pint.simulation import make_fake_toas_fromtim, make_fake_toas_fromMJDs
from pint.logging import setup

_ = setup(level='ERROR')

if __name__ == '__main__':

    process_start = time.time()

    # Parser Arguments
    parser = argparse.ArgumentParser(description = f'Options to run SPNA on epoch-wise DMs')
    
    parser.add_argument(f'--psrname', required=True, 
                        help=f'Name of the pulsar')
    
    parser.add_argument(f'--parfile', required=True, 
                        help=f'Path to the pulsar timing ephermis file')
    
    parser.add_argument(f'--dmfile', required=True, 
                        help=f'Path to the epoch-wise DM time series file')
    
    parser.add_argument(f'--paramfile', required=True, 
                        help=f'Path to the parameter file')

    parser.add_argument(f'--narrowband', action=f'store_true', default=False, 
                        help=f'Whether DM time series is derived from sub-banded ToAs')

    parser.add_argument(f'--timfile', type=str, default=None, 
                        help=f'Path to the pulsar tim file (only for Wideband DMs)')
    
    parser.add_argument(f'--Nsamples', type=int, default=1000, 
                        help=f'Number of samples per walker to be used in emcee')
    
    parser.add_argument(f'--outdir', type=str, default=None, 
                        help=f'Path to the output directory')
    
    parser.add_argument(f'--sampler', type=str, default=f'emcee', 
                        help=f'Sampler to be used for estimating the posteriors (supported: emcee or pocomc)')

    parser.add_argument(f'--DMNharm', type=int, default=None, 
                        help=f'Number of harmonics for DM noise process')

    parser.add_argument(f'--SWNharm', type=int, default=None, 
                        help=f'Number of harmonics for SW noise process')

    parser.add_argument(f'--truthsfile', type=str, default=None, 
                        help=f'Path to the parameter injection file')

    parser.add_argument(f'--fref', type=float, default=1400, 
                        help=f'Reference chromatic frequency for transformations')

    parser.add_argument(f'--plotchains', action=f'store_true', default=False, 
                        help=f'Whether to plot the parameter chains')

    parser.add_argument(f'--psd', type=str, default=f'powerlaw', 
                        help=f'Power Spectral Density of the GP')

    parser.add_argument(f'--plot_realisations', action=f'store_true', default=False, 
                        help=f'Whether to plot the reconstructed noise realisations')

    parser.add_argument(f'--threshold', type=float, default=3.0, 
                        help=f'Threshold in units of sigma to construct distribution for realisations')

    parser.add_argument(f'--add_dms', action=f'store_true', default=False, 
                        help=f'Whether to plot the DM time series with realisations')

    parser.add_argument(f'--n_realisations', type=int, default=20, 
                        help=f'Number of realisations to be generated for the estimated parameters')

    parser.add_argument(f'--vectorise', action=f'store_true', default=False, 
                        help=f'Whether to vectorise the likelihood computations')

    args = parser.parse_args()

    if args.outdir:
        outdir = args.outdir
        try: 
            os.system(f'mkdir -p {outdir}')
        except OSError: 
            outdir = os.getcwd()
            pass
    else:
        outdir = os.getcwd()
    
    chaindir = f'{outdir}/chains'
    try: os.system(f'mkdir -p {chaindir}')
    except OSError: 
        chaindir = outdir
        pass

    # Generic for narrowband or wideband
    t, dms, dmerrs, flags = read_dmfile(args.dmfile, return_flags=True)
    Tspan = t[-1] - t[0]

    print(f'\nPulsar Name: {args.psrname}')
    plot_dmseries(t, dms, dmerrs, outdir, args.psrname)

    # Creating the timing model and ToAs object from PINT
    psrmodel = get_model(parfile=args.parfile, allow_tcb=True, allow_T2=True)

    try: Tref_dm = psrmodel.DMEPOCH.value*u.day.to(u.s)
    except AttributeError:
        print(f'\nThe DMEPOCH is not present in the timing model')
        print(f'\nThe process will terminate')
        sys.exit()

    try: Tref_sw = psrmodel.SWEPOCH.value*u.day.to(u.s)
    except AttributeError:
        print(f'\nThe SWEPOCH is not present in the timing model')
        print(f'\nSetting the SWEPOCH to be same as the DMEPOCH for {args.psrname}')
        Tref_sw = Tref_dm

    print(f'\nTime span of dataset = {Tspan*u.s.to(u.year):.2f} years')
    print(f'\nDM Reference Epoch = {Tref_dm*u.s.to(u.day)}')
    print(f'\nSW Reference Epoch = {Tref_sw*u.s.to(u.day)}')
    print(f'\nNumber of DMN harmonics = {args.DMNharm}')
    print(f'\nNumber of SWN harmonics = {args.SWNharm}')
    print(f'\nDM groups as per the DM time series: {np.unique(flags)}')

    if not args.narrowband and not args.timfile:
        print(f'\nA timfile must be provided for wideband DMs')
        print(f'\nThe process will terminate')
        sys.exit()
    else:
        if not args.narrowband:
            if os.path.isfile(args.timfile):
                print(f'\nGenerating ToAs using the tim file {args.timfile}')
                psrtoas = make_fake_toas_fromtim(args.timfile, model=psrmodel, 
                                                 add_noise=False, add_correlated_noise=False)
            else:
                print(f'\nThe tim file {args.timfile} does not exist')
                print(f'\nThe process will terminate')
                sys.exit()
        else:
            print(f'\nGenerating ToAs using MJDs from the DM time series')
            psrtoas = make_fake_toas_fromMJDs(t*u.s.to(u.day)*u.day, model=psrmodel, 
                                              add_noise=False, add_correlated_noise=False)
    
    params = get_DispParams(args.paramfile, flags)
    dispefacparams = np.array([param for param in params if param.is_white and 'EFAC' in param.name])
    dispequadparams = np.array([param for param in params if param.is_white and 'EQUAD' in param.name])
    dmnparams = np.array([param for param in params if param.is_red and 'DM' in param.name])
    swnparams = np.array([param for param in params if param.is_red and 'SW' in param.name])
    swdetparams = [param for param in params if param.is_deterministic and 'NE_SW' in param.name]
    orders = np.array([int(param.name.split('NE_SW')[-1]) if param.name.split('NE_SW')[-1] else 0 
              for param in swdetparams])
    
    sample_noise(name=args.sampler, params=params, Nsamples=args.Nsamples, DMNharm=args.DMNharm, 
                 SWNharm=args.SWNharm, outdir=outdir, chaindir=chaindir, psrname=args.psrname, t=t, dms=dms, 
                 dmerrs=dmerrs, Tspan=Tspan, Tref_dm=Tref_dm, Tref_sw=Tref_sw, toas=psrtoas, 
                 flags=flags, timing_model=psrmodel, fref=args.fref*u.MHz, 
                 plotchains=args.plotchains, truthsfile=args.truthsfile, 
                 dispefacparams=dispefacparams, dispequadparams=dispequadparams, 
                 dmnparams=dmnparams, swnparams=swnparams, swdetparams=swdetparams, orders=orders, 
                 plot_generated_realisations=args.plot_realisations, threshold=args.threshold, 
                 add_dms=args.add_dms, n_realisations=args.n_realisations, vectorise=args.vectorise)
    
    process_finish = time.time()
    print(f'\nThe process took {(process_finish-process_start)/60:.2f} minutes to finish')
