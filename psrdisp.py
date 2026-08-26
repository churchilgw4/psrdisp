import numpy as np
import astropy.units as u
import argparse
import multiprocessing as mp
import emcee
import corner
import time
import sys

from disputils import *
from pint.models.model_builder import get_model
from pint.simulation import make_fake_toas_fromtim, make_fake_toas_fromMJDs

if __name__ == '__main__':

    process_start = time.time()

    # Parser Arguments
    parser = argparse.ArgumentParser(description = f'Options to run SPNA on epoch-wise DMs')
    
    parser.add_argument(f'--parfile', required=True, 
                        help=f'Path to the pulsar timing ephermis file')
    
    parser.add_argument(f'--dmfile', required=True, 
                        help=f'Path to the epoch-wise DM time series file')
    
    parser.add_argument(f'--paramlist', required=True, 
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
                        help=f'Sampler to be used for estimating the posteriors')

    parser.add_argument(f'--DMNharm', type=int, default=None, 
                        help=f'Number of harmonics for DM noise process')

    parser.add_argument(f'--SWNharm', type=int, default=None, 
                        help=f'Number of harmonics for SW noise process')

    parser.add_argument(f'--truthsfile', type=str, default=None, 
                        help=f'Path to the parameter injection file')

    parser.add_argument(f'--fref', type=float, default=1400, 
                        help=f'Reference chromatic frequency for transformations')

    args = parser.parse_args()

    if not args.narrowband and not args.timfile:
        print(f'\nNo timfile provided for wideband DMs')
        print(f'\nThe process will terminate')
        sys.exit()

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

    psrname = get_parfile_param(args.parfile, 'PSRJ')
    print(f'\nPulsar Name: {psrname}')

    Tref_pos = np.longdouble(get_parfile_param(args.parfile, f'POSEPOCH'))*u.day.to(u.s)
    Tref_dm = np.longdouble(get_parfile_param(args.parfile, f'DMEPOCH'))*u.day.to(u.s)
    try: Tref_sw = np.longdouble(get_parfile_param(args.parfile, f'SWEPOCH'))*u.day.to(u.s)
    except Exception: Tref_sw = Tref_dm

    print(f'\nTime span of dataset = {Tspan*u.s.to(u.year):.2f} years')
    print(f'\nPosition Reference Epoch = {Tref_pos*u.s.to(u.day)}')
    print(f'\nDM Reference Epoch = {Tref_dm*u.s.to(u.day)}')
    print(f'\nSW Reference Epoch = {Tref_sw*u.s.to(u.day)}')
    print(f'\nNumber of DMN harmonics = {args.DMNharm}')
    print(f'\nNumber of SWN harmonics = {args.SWNharm}')
    print(f'\nDM groups as per the DM time series: {np.unique(flags)}')

    




















    