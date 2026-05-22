import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import astropy.units as u
from astropy.coordinates import Angle, get_body_barycentric
from astropy.time import Time
from pint import dmu, DMconst
import scipy.linalg as sl
import os
import sys
import time
import emcee
from pint.logging import setup
import corner
from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc
from pint.models.model_builder import get_model
from pint.simulation import make_fake_toas_fromtim, make_fake_toas_fromMJDs
from IPython.display import display, Latex

import multiprocessing as mp

import warnings
warnings.filterwarnings('ignore')
_ = setup(level='WARNING')

process_start = time.time()

def get_freqs(Tspan, Nharm=None):
    '''
        Function to get the Fourier-domain frequencies
    '''
    if not Nharm:
        Tspan_years = Tspan*u.s.to(u.year)
        Nharm = int(np.floor(Tspan_years*12))
    
    f = np.linspace(1/Tspan, Nharm/Tspan, Nharm)
    
    return f

def get_Nmat(dmerrs, theta, params, flags):
    '''
        Function to get the white-noise covariance matrix
    '''
    # Faster computation using vectors instead of square matrices
    pnames = [key for key in params.keys()]
    dmefacs, dmequads = {}, {}
    for index, pname in enumerate(pnames):
        if 'dmefac' in pname or 'DMEFAC' in pname:
            dmefacs[pname] = {}
            dmefacs[pname]['idx'] = index
        elif 'dmequad' in pname or 'DMEQUAD' in pname:
            dmequads[pname] = {}
            dmequads[pname]['idx'] = index

    uniqueflags = np.unique(flags)
    
    if uniqueflags.size == 1:
        if not len(dmefacs) and not len(dmequads):
            dmefac, dmequad = 1.0, -np.inf
        elif not len(dmefacs) and len(dmequads):
            dmefac = 1.0
            dmequad = np.array([theta[dmequads[p]['idx']] for p in dmequads.keys()])
        elif not len(dmequads) and len(dmefacs):
            dmefac = np.array([theta[dmefacs[p]['idx']] for p in dmefacs.keys()])
            dmequad = -np.inf
        else:
            dmefac = np.array([theta[dmefacs[p]['idx']] for p in dmefacs.keys()])
            dmequad = np.array([theta[dmequads[p]['idx']] for p in dmequads.keys()])

    else:
        mask = {val: flags==val for val in uniqueflags}
        if not len(dmefacs) and not len(dmequads):
            dmefac, dmequad = 1.0, -np.inf
        
        elif not len(dmefacs) and len(dmequads):
            dmefac = 1.0
            for dmequad, flag in zip(dmequads.keys(), uniqueflags):
                dmequads[dmequad]['mask'] = mask[flag]
                dmequads[dmequad]['vals'] = theta[dmequads[dmequad]['idx']]*np.ones(dmerrs.size)[dmequads[dmequad]['mask']]
            dmequad = np.concatenate([dmequads[p]['vals'] for p in dmequads.keys()])

        elif not len(dmequads) and len(dmefacs):
            dmequad = -np.inf
            for dmefac, flag in zip(dmefacs.keys(), uniqueflags):
                dmefacs[dmefac]['mask'] = mask[flag]
                dmefacs[dmefac]['vals'] = theta[dmefacs[dmefac]['idx']]*np.ones(dmerrs.size)[dmefacs[dmefac]['mask']]
            dmefac = np.concatenate([dmefacs[p]['vals'] for p in dmefacs.keys()])

        else:
            for dmefac, dmequad, flag in zip(dmefacs.keys(), dmequads.keys(), uniqueflags):
                dmefacs[dmefac]['mask'] = dmequads[dmequad]['mask'] = mask[flag]
                dmefacs[dmefac]['vals'] = theta[dmefacs[dmefac]['idx']]*np.ones(dmerrs.size)[dmefacs[dmefac]['mask']]
                dmequads[dmequad]['vals'] = theta[dmequads[dmequad]['idx']]*np.ones(dmerrs.size)[dmequads[dmequad]['mask']]
            dmefac = np.concatenate([dmefacs[p]['vals'] for p in dmefacs.keys()])
            dmequad = np.concatenate([dmequads[p]['vals'] for p in dmequads.keys()])

    Nmat = np.array(dmefac**2 * (dmerrs**2 + 10**(2*dmequad)))
    
    return Nmat

def powerlaw(f, Tspan, log10_A, gamma):
    '''
        Function to get the Fourier-domain power-law for a given basis
    '''
    yr = 1*u.year.to(u.s)
    fyr = 1/yr
    pl = ((10**log10_A)**2/(12*np.pi**2*Tspan))*f**(-gamma)*fyr**(gamma-3)
    
    return pl

def get_parfile_param(parfile, param):
    '''
        Function to get the parfile entry for a given parameter
    '''
    parlines = [f.split('\n')[0] for f in open(parfile, 'r').readlines() if not f.startswith('#')]
    for line in parlines:
        if param in line:
            entries = line.split()
            if param in entries:
                paramval = entries[np.where(np.array(entries)==param)[0][0]+1]
    
    return paramval

def get_dm_sw(toas, timing_model, n_earth=1.0):
    '''
        Function to return the DM contribution from solar wind
    '''
    geometric_factor = timing_model.components['SolarWindDispersion'].solar_wind_geometry(toas).value
    dm_sw = n_earth*geometric_factor
    
    return dm_sw

def get_swdetsignal(t, Tref_dm, toas, timing_model, theta, params):
    '''
        Function to get a deterministic signal for the solar-wind process
    '''
    pnames = [key for key in params.keys()]
    swdetparams = {}
    for index, pname in enumerate(pnames):
        if 'ne_sw' in pname or 'NE_SW' in pname:
            swdetparams[pname] = index

    if not len(swdetparams):
        signal = np.zeros(t.size)
    else:
        # Since the prior_transform is being used, the indexing in theta is set accordingly
        signal = 0
        for key in swdetparams.keys():
            pidx = swdetparams[key]
            if key == 'ne_sw' or key == 'NE_SW':
                signal += get_dm_sw(toas, timing_model, theta[pidx])
            elif key == 'ne_sw1' or key == 'NE_SW1':
                signal += get_dm_sw(toas, timing_model, theta[pidx])*(t-Tref_dm)*u.s.to(u.year)
            elif key == 'ne_sw2' or key == 'NE_SW2':
                signal += get_dm_sw(toas, timing_model, theta[pidx])*0.5*((t-Tref_dm)*u.s.to(u.year))**2
            else:
                raise Exception(f'The deterministic solar-wind signal cannot account for {key} for now.')
                
    fid_dm = float(get_parfile_param(parfile, 'DM'))
    signal += fid_dm
    return signal

def get_Umat(t, f, Tspan, Tref_dm, params, toas, timing_model, fref=1400*u.MHz):
    '''
        Function to get the basis matrix for Fourier domain red noise process
        This implementation considers SWEPOCH to be the same as the DMEPOCH
        Also, the entries for DMGP and SWGP are scaled with fref**2/DMconst to make them coherent
        with the definitions in ENTERPRSIE (taking fref as 1400 MHz for comparison)
    '''
    pnames = [key for key in params.keys()]
    dmgp_params = [pname for pname in pnames if 'gp_dm' in pname or 'GP_DM' in pname]
    swgp_params = [pname for pname in pnames if 'gp_sw' in pname or 'GP_SW' in pname]
    
    # If there are no SWGP parameters then swparams has 0 size
    if len(dmgp_params):
        if len(swgp_params):
            U = np.zeros((t.size, 4*f.size+3))
        else:
            U = np.zeros((t.size, 2*f.size+3))
    elif len(swgp_params):
        U = np.zeros((t.size, 2*f.size+3))
    else:
        # Only including the DM Taylor Series process
        U = np.zeros((t.size, 3))

    convfactor = fref**2/DMconst
    for i in range(t.size):
        # Contributions from DMGP
        if len(dmgp_params):
            for j in range(f.size):
                uval = 2*np.pi*(j+1)*(t[i]-Tref_dm)/Tspan
                U[i, 2*j] = np.sin(uval)*convfactor.value
                U[i, 2*j+1] = np.cos(uval)*convfactor.value

        # Contributions from DM Taylor Series
        U[i, -3] = 1.0
        U[i, -2] = (t[i]-Tref_dm)*u.s.to(u.year)
        U[i, -1] = 0.5*((t[i]-Tref_dm)*u.s.to(u.year))**2

    # Contributions from SWGP
    fyr = 1/(1*u.year.to(u.s)*u.s)
    if len(swgp_params):
        dm_sw = get_dm_sw(toas, timing_model, n_earth=1.0)
        for j in range(f.size):
            uval_sw = 2*np.pi*(j+1)*(t-Tref_dm)/Tspan
            if not len(dmgp_params):
                U[:, 2*j] = np.sin(uval_sw)*dm_sw
                U[:, 2*j+1] = np.cos(uval_sw)*dm_sw
            else:
                U[:, 2*f.size+2*j] = np.sin(uval_sw)*dm_sw
                U[:, 2*f.size+2*j+1] = np.cos(uval_sw)*dm_sw
                
    return U

def get_Phimat(f, Tspan, theta, params):
    '''
        Function to get the GP kernel matrix for PL process
    '''
    pnames = [key for key in params.keys()]
    dmgp_params = {}
    swgp_params = {}
    for index, pname in enumerate(pnames):
        if 'gp_sw' in pname or 'GP_SW' in pname:
            swgp_params[pname] = index
        elif 'gp_dm' in pname or 'GP_DM' in pname:
            dmgp_params[pname] = index
    
    if len(dmgp_params):
        if len(swgp_params):
            Phi = np.zeros(4*f.size+3)
        else:
            Phi = np.zeros(2*f.size+3)
    elif len(swgp_params):
        Phi = np.zeros(2*f.size+3)
    else:
        # Only including the DM Taylor Series process
        Phi = np.zeros(3)
        
    if Phi.size > 3:
        if len(dmgp_params):
            amp_key = [key for key in dmgp_params.keys() if 'log10_A_gp_dm' in key][0]
            gamma_key = [key for key in dmgp_params.keys() if 'gamma_gp_dm' in key][0]
            log10_A_gp_dm = theta[dmgp_params[amp_key]]
            gamma_gp_dm = theta[dmgp_params[gamma_key]]
            for i in range(f.size):
                Phi[2*i] = Phi[2*i+1] = powerlaw(f[i], Tspan, log10_A=log10_A_gp_dm, gamma=gamma_gp_dm)
            
            if len(swgp_params):
                amp_key = [key for key in swgp_params.keys() if 'log10_A_gp_sw' in key][0]
                gamma_key = [key for key in swgp_params.keys() if 'gamma_gp_sw' in key][0]
                log10_A_gp_sw = theta[swgp_params[amp_key]]
                gamma_gp_sw = theta[swgp_params[gamma_key]]
                for i in range(f.size):
                    Phi[2*f.size+2*i] = Phi[2*f.size+2*i+1] = powerlaw(f[i], Tspan, log10_A=log10_A_gp_sw, 
                                                                                      gamma=gamma_gp_sw)
        elif len(swgp_params):
            amp_key = [key for key in swgp_params.keys() if 'log10_A_gp_sw' in key][0]
            gamma_key = [key for key in swgp_params.keys() if 'gamma_gp_sw' in key][0]
            log10_A_gp_sw = theta[swgp_params[amp_key]]
            gamma_gp_sw = theta[swgp_params[gamma_key]]
            for i in range(f.size):
                Phi[2*i] = Phi[2*i+1] = powerlaw(f[i], Tspan, log10_A=log10_A_gp_sw, gamma=gamma_gp_sw)

    # Default ENTERPRISE construct uses 1e+40 instead of infinity
    Phi[-3] = Phi[-2] = Phi[-1] = 1e+40

    return Phi

def UniformPrior(value, pmin, pmax):
    '''
        Function to give uniform prior distribution
    '''
    prior = ((value >= pmin) & (value <= pmax)) / (pmax - pmin)
    
    return prior

def get_lnprior(theta, params):
    '''
        Function to get the log-prior for different parameters
    '''
    if not isinstance(params, dict):
        raise Exception('Invalid parameter list')
        
    pnames = [key for key in params.keys()]
    pmins = np.array([float(val) for pname in pnames for key, val in params[pname].items() if key == 'pmin'])
    pmaxs = np.array([float(val) for pname in pnames for key, val in params[pname].items() if key == 'pmax'])
    pdist = np.array([val for pname in pnames for key, val in params[pname].items() if key == 'dist'])
    pdistchecks = np.array([dist for dist in pdist if not dist in np.array(['uniform', 'log-uniform'])])
    
    if np.any(pdistchecks):
        raise Exception('Only uniform and log-uniform priors are allowed.')
    
    lnprior = np.sum([np.log(UniformPrior(value, pmin, pmax)) for value, pmin, pmax in zip(theta, pmins, pmaxs)])

    return lnprior

def get_lnpost(theta, params, dms, dmerrs, t, Tspan, 
               Tref_dm, toas, flags, timing_model, Nharm=None, verbose=True):
    '''
        Function to get the log-posterior for params
    '''
    lnprior = get_lnprior(theta, params)
    
    # This allows the code to not compute the likelihood for params outside the prior range
    if not np.isfinite(lnprior):
        return lnprior
    else:
        return lnprior + get_lnlike_optimised(theta, params, dms, dmerrs, t, 
                                              Tspan, Tref_dm, toas, flags, timing_model, Nharm, verbose)

def prior_transform(theta, params):
    '''
        Function to get the prior transform for params
    '''
    pnames = [key for key in params.keys()]
    pmins = np.array([float(val) for pname in pnames for key, val in params[pname].items() if key == 'pmin'])
    pmaxs = np.array([float(val) for pname in pnames for key, val in params[pname].items() if key == 'pmax'])
    spans = pmaxs - pmins
    cube = spans * theta + pmins
    
    return cube

def get_lnlike_optimised(theta, params, dms, dmerrs, t, Tspan, 
                         Tref_dm, toas, flags, timing_model, Nharm=None, verbose=False):
    '''
        Function to get the log-likelihood in an optimised form for params
    '''
    f = get_freqs(Tspan, Nharm)
    N = get_Nmat(dmerrs, theta, params, flags)
    Phi = get_Phimat(f, Tspan, theta, params)
    U = get_Umat(t, f, Tspan, Tref_dm, params, toas, timing_model)
    
    # DM will remain unaffected if there are no deterministic SW parameters
    swdet = get_swdetsignal(t, Tref_dm, toas, timing_model, theta, params)
    if np.any(swdet > 0.0) and verbose:
        plt.ion()
        plt.clf()
        plt.show()
        plt.scatter(t*u.day, dms, c='r', label='DMs')
        plt.scatter(t*u.day, swdet, c='k', label='SWdet')
        plt.legend(loc='upper left')
        plt.ticklabel_format(useOffset=False)
        plt.tight_layout()
        plt.draw()
        plt.pause(0.001)
    y = dms - swdet
    
    Ninv_y = y / N
    y_Ninv_y = y @ Ninv_y
    Ninv_U = U / N[:, None]
    UT_Ninv_y = Ninv_U.T @ y
    UT_Ninv_U = U.T @ Ninv_U
    Sigmainv = np.diag(1/Phi) + UT_Ninv_U
    
    try:
        # SVD can also be done to implement this (as done in la_forge)
        Sigmainv_cf = sl.cho_factor(Sigmainv)
        Sigmainv_UT_Ninv_y = sl.cho_solve(Sigmainv_cf, UT_Ninv_y)
        y_Ninv_U_Sigma_UT_Ninv_y = UT_Ninv_y @ Sigmainv_UT_Ninv_y
        y_Cinv_y = y_Ninv_y - y_Ninv_U_Sigma_UT_Ninv_y
        logdet_N = np.sum(np.log(N))
        logdet_Sigmainv = 2*np.sum(np.log(np.diag(Sigmainv_cf[0])))
        logdet_Phi = np.sum(np.log(Phi))
        logdet_C = logdet_N + logdet_Sigmainv + logdet_Phi
        lnlike = -0.5*(y_Cinv_y + logdet_C)
    
    except Exception:
        lnlike = -np.inf

    return lnlike

def get_realisation(theta, params, dms, dmerrs, t, Tspan, 
                    Tref_dm, toas, flags, timing_model, Nharm=None):
    '''
        Function to get a random time-domain noise process realisation 
        for the analytically marginalised parameters
    '''
    f = get_freqs(Tspan, Nharm)
    N = get_Nmat(dmerrs, theta, params, flags)
    Phi = get_Phimat(f, Tspan, theta, params)
    U = get_Umat(t, f, Tspan, Tref_dm, params, toas, timing_model)
    swdet = get_swdetsignal(t, Tref_dm, toas, timing_model, theta, params)
    y = dms - swdet

    Ninv_U = U / N[:, None]
    UT_Ninv_y = y @ Ninv_U
    Sigmainv = np.diag(Phi) + U.T @ Ninv_U
    Sigmainv_cf = sl.cholesky(Sigmainv, lower=False)
    a_hat = sl.cho_solve((Sigmainv_cf, False), UT_Ninv_y)
    z = np.random.randn(len(a_hat))
    a = a_hat + sl.solve_triangular(Sigmainv_cf, z, lower=False)
    realisation = U @ a
    
    return realisation

def plot_realisations(n_realisations, chains, params, dms, dmerrs, t, Tspan, 
                      Tref_dm, flags, parfile, Nharm, outdir, psrname, toas, timing_model, 
                      save=True, add_dms=False, no_title=False):
    '''
        Function to plot noise time-domain noise process realisations
    '''
    param_medians = {}
    param_sigma = {}
    for i, key in enumerate(params.keys()):
        param_medians[key] = np.median(chains[:, i])
        param_sigma[key] = np.std(chains[:, i])

    print(f'Median Parameter Values : \n{param_medians}')
    print(f'Standard Deviation in Parameter Values : \n{param_sigma}')

    param_choices = np.zeros((n_realisations, len(params)))
    for index, key in enumerate(param_medians.keys()):
        mu = param_medians[key]
        sigma = param_sigma[key]
        dist = np.random.normal(loc=mu, scale=3*np.abs(sigma), size=n_realisations)
        param_choices[:, index] = np.random.choice(dist, size=n_realisations)

    DM = float(get_parfile_param(parfile, 'DM'))
    try: DM1 = float(get_parfile_param(parfile, 'DM1'))
    except Exception: DM1 = 0.0
    try: DM2 = float(get_parfile_param(parfile, 'DM2'))
    except Exception: DM2 = 0.0

    DMdet_signal = DM + DM1*(t-Tref_dm)*u.s.to(u.year) + DM2*0.5*((t-Tref_dm)*u.s.to(u.year))**2
    
    # The realisations only provide the variations in DMs due to the parameters in the design matrix
    full_signal = np.zeros((n_realisations, t.size))
    for i in range(n_realisations):
        realisation = get_realisation(param_choices[i, :], params, dms, dmerrs, t, Tspan, 
                                             Tref_dm, toas, flags, timing_model, Nharm=None)
        SWdet_signal = get_swdetsignal(t, Tref_dm, toas, timing_model, param_choices[i, :], params) - DM
        full_signal[i, :] = DMdet_signal + SWdet_signal + realisation

    if add_dms:
        fig, ax = plt.subplots(figsize=(10, 5), dpi=150, nrows=1, ncols=1, sharex=True)
        fig.subplots_adjust(wspace=0, hspace=0)
        ax.ticklabel_format(useOffset=False)
        ax.grid(ls='dashed')
        if not no_title:
            ax.set_title(rf'{psrname}, DM process realisations', fontweight='bold')
        for i in range(n_realisations):
            ax.plot(t*u.s.to(u.day), full_signal[i, :], color='k', alpha=0.05)
        ax.ticklabel_format(useOffset=False)
        ax.grid(ls='dashed')
        _ = ax.set_ylabel(rf'DM [pc/cm$^3$]', fontweight='bold', fontsize=13)
        _ = ax.set_xlabel(f'MJD', fontweight='bold', fontsize=13)
        ax.errorbar(t*u.s.to(u.day), dms, yerr=dmerrs, fmt='o', ms=4, c='crimson', alpha=0.3, \
                         elinewidth=1.0, capsize=2, capthick=0.5, ecolor='k', label='Simulated DM')
        ax.legend(loc='upper left')
        plt.tight_layout()
        if save:
            plt.savefig(f'{outdir}/{psrname}.realisations.pdf')

    else:
        fig = plt.figure(figsize=(10, 5), dpi=100)
        plt.ticklabel_format(useOffset=False)
        plt.grid(ls='dashed')
        _ = plt.xlabel(f'MJD', fontweight='bold', fontsize=13)
        _ = plt.ylabel(rf'DM [pc/cm$^3$]', fontweight='bold', fontsize=13)
        if not no_title:
            plt.title(rf'{psrname}, DM process realisations', fontweight='bold')
        for i in range(n_realisations):
            plt.plot(t*u.s.to(u.day), full_signal[i, :], color='k', alpha=0.01)
        plt.tight_layout()
        if save:
            plt.savefig(f'{outdir}/{psrname}.realisations.pdf')

if __name__ == '__main__':
    
    parfile = sys.argv[1]
    dmfile = sys.argv[2]
    paramlist = sys.argv[3]
    Nsamples = int(sys.argv[4])
    try: outdir = sys.argv[5]
    except Exception: outdir = os.getcwd()
    try: sampling = sys.argv[6]
    except Exception: sampling = 'emcee'
    try: Nharm = int(sys.argv[7])
    except Exception: Nharm = None
    try: truthsfile = sys.argv[8]
    except Exception: truthsfile = None
    timfile = sys.argv[9]
    try: narrowband = bool(sys.argv[10])
    except Exception: narrowband = False
    
    try: os.system(f'mkdir -p {outdir}')
    except OSError: pass
    
    chaindir = f'{outdir}/chains'
    try: os.system(f'mkdir -p {chaindir}')
    except OSError: pass
    
    t = np.array([np.longdouble(f.split('\n')[0].split(' ')[0]) for 
                  f in open(dmfile, 'r').readlines() if not f.startswith('#')])*u.day.to(u.s)
    
    dms = np.array([float(f.split('\n')[0].split(' ')[1]) for 
                    f in open(dmfile, 'r').readlines() if not f.startswith('#')])
    
    dmerrs = np.array([float(f.split('\n')[0].split(' ')[2]) for 
                       f in open(dmfile, 'r').readlines() if not f.startswith('#')])

    psrname = get_parfile_param(parfile, f'PSRJ')
    print(f'Pulsar Name : {psrname}')
    
    Tspan = (t[-1]-t[0])
        
    # Considering the DMEPOCH as the reference epoch for DM and SW Taylor Series
    # Considering the POSEPOCH as the reference epoch for position and astrometry calculations
    Tref_pos = np.longdouble(get_parfile_param(parfile, f'POSEPOCH'))*u.day.to(u.s)
    Tref_dm = np.longdouble(get_parfile_param(parfile, f'DMEPOCH'))*u.day.to(u.s)

    print(f'Time span of dataset = {Tspan*u.s.to(u.year):.2f} years')
    print(f'Position Reference Epoch = {Tref_pos*u.s.to(u.day)}')
    print(f'DM Reference Epoch = {Tref_dm*u.s.to(u.day)}')
    print(f'Number of GP harmonics = {Nharm}')

    timing_model = get_model(parfile, allow_T2=True, allow_tcb=True)
    print(f'Timing Model components : \n{timing_model.components}')

    if not narrowband:
        print(f'Generating the Wideband ToAs from the tim file')
        toas = make_fake_toas_fromtim(timfile, model=timing_model, 
                                      add_noise=False, 
                                      add_correlated_noise=False)
        try: flags = np.array([f.split('\n')[0].split(' ')[-1] for 
                               f in open(dmfile, 'r').readlines() if not f.startswith('#')])
        except Exception: flags = np.array(['noflag' for item in dmerrs])
    else:
        print(f'Generating the Narrowband ToAs from epochs in the DM time series')
        toas = make_fake_toas_fromMJDs(t*u.s.to(u.day)*u.day, 
                                       model=timing_model, 
                                       add_noise=False, 
                                       add_correlated_noise=False)
        # Assuming no additional flags are present in the DM time series file
        # Works only for a single DMEFAC and DMEQUAD
        flags = np.array(['noflag' for item in dmerrs])
    
    param_entries = np.array([f.split('\n')[0] for 
                              f in open(paramlist, 'r').readlines() if not f.startswith('#')])
    params = {}
    param_names = []
    for param in param_entries:
        pname = param.split(' ')[0]
        pmin = float(param.split(' ')[1])
        pmax = float(param.split(' ')[2])
        param_names.append(pname)
        if not pmin < pmax:
            raise Exception(f'{pname} should have pmax larger than pmin')
        try: dist = param.split(' ')[3]
        except Exception: dist = 'uniform'
        if 'dmefac' in pname or 'dmequad' in pname:
            try: flagval = param.split(' ')[4]
            except Exception: 
                try: flagval = flags[0]
                except Exception: flagval = 'noflag'
            pname = f'{pname}_{flagval}'
        params[pname] = {}
        params[pname]['pmin'] = pmin
        params[pname]['pmax'] = pmax
        params[pname]['dist'] = dist
    
    params = dict(sorted(params.items()))
    param_names = sorted(param_names)
    
    print(f'Parameter entries as per the {paramlist} file :')
    for key in params.keys():
        print(f"\t{key}, pmin = {params[key]['pmin']}, pmax = {params[key]['pmax']}, pdist = {params[key]['dist']}")

    print(f'Parameter names : \n{param_names}')

    param_labels = []
    for index, pname in enumerate(param_names):
        if pname.upper() == 'DMEFAC':
            param_labels.append('DISPEFAC')
        elif pname.upper() == 'DMEQUAD':
            param_labels.append('DISPEQUAD')
        elif '_gp_' in pname:
            if 'gamma' in pname:
                if 'dm' in pname:
                    param_labels.append(r"$\gamma_{\rm dm}$")
                elif 'sw' in pname:
                    param_labels.append(r"$\gamma_{\rm sw}$")
            elif 'log10' in pname:
                if 'dm' in pname:
                    param_labels.append(r"$\log_{10}A_{\rm dm}$")
                elif 'sw' in pname:
                    param_labels.append(r"$\log_{10}A_{\rm sw}$")
        else:
            param_labels.append(pname)

    print(f'Parameter labels : \n{param_labels}')
    
    fig = plt.figure(figsize=(10, 5), dpi=100)
    plt.errorbar(t*u.s.to(u.day), dms, yerr=dmerrs, fmt='o', ms=7, c='crimson', alpha=0.5, \
                 elinewidth=1.0, capsize=2, capthick=0.5, ecolor='k')
    plt.ticklabel_format(useOffset=False)
    plt.grid(ls='dashed')
    _ = plt.xlabel(f'MJD', fontweight='bold', fontsize=13)
    _ = plt.ylabel(rf'DM [pc/cm$^3$]', fontweight='bold', fontsize=13)
    plt.title(rf'{psrname}, Median DMerr = {np.median(dmerrs):.2e} pc/cm$^3$', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{outdir}/{psrname}.DM.time.series.pdf')

    Ndim = len(params)
    burn = Nsamples//4
    thin = 2
    verbose = False
    
    if sampling == 'emcee':
        # Setting up the sampling process
        # Nwalkers = os.cpu_count()
        Nwalkers = Ndim*5
        print(f'Ndim = {Ndim}, Nsamples (per walker) = {Nsamples}, Nwalkers = {Nwalkers}')
        p0 = np.array([prior_transform(theta, params) for theta in np.random.rand(Nwalkers, Ndim)])
        
        backendfile = f'{chaindir}/{psrname}.chains.h5'
        backend = emcee.backends.HDFBackend(backendfile)
        chainfile = f'{chaindir}/{psrname}.chains.txt'
    
        if not os.path.isfile(chainfile):
            with mp.Pool() as pool:
                sampler = emcee.EnsembleSampler(Nwalkers, Ndim, get_lnpost, 
                                                args=(params, dms, dmerrs, t, Tspan, Tref_dm, 
                                                      toas, flags, timing_model, Nharm, verbose), 
                                                backend=backend, pool=pool)
                
                if not os.path.isfile(backendfile):
                    sampler.reset()
                    sampler.run_mcmc(p0, Nsamples, progress=True, store=True)
                else:
                    try:
                        samples_backend = backend.get_chain()
                        nsamples_backend = samples_backend.shape[0]
                        if nsamples_backend < Nsamples:
                            samples_earlier = sampler.get_chain()
                            print(f'Initiating sampling with {samples_earlier.size} previously obtained samples')
                            sampler_state = sampler.get_last_sample()
                            sampler.run_mcmc(sampler_state, Nsamples, progress=True, store=True)
                        else:
                            print(f'Initiating the sampler with reseting the backend')
                            backend.reset(nwalkers=Nwalkers, ndim=Ndim)
                            sampler.reset()
                            sampler.run_mcmc(p0, Nsamples, progress=True, store=True)
                    except Exception:
                        print(f'Cleared-out backend, initiating fresh sampling')
                        sampler.reset()
                        sampler.run_mcmc(p0, Nsamples, progress=True, store=True)
    
                # Plotting the chains
                chains = sampler.get_chain(flat=True, discard=burn, thin=thin)
                np.savetxt(chainfile, chains)
                
            for dim in range(Ndim):
                plt.figure(figsize=(10, 3), dpi=100)
                plt.plot(chains[burn:, dim], color='crimson', alpha=0.3)
                plt.xlabel('Samples', fontweight='bold', fontsize=10)
                plt.ylabel('Value', fontweight='bold', fontsize=10)
                plt.title(f'{param_labels[dim]}', fontweight='bold', fontsize=13)
                plt.grid(True, alpha=0.5, ls='dashed')
                plt.tight_layout()
                plt.savefig(f'{chaindir}/{psrname}.{param_names[dim]}.chains.pdf', dpi=150)
    
        else:
            print(f'Using the original chains : {chainfile}')
            chains = np.loadtxt(chainfile)

    elif sampling == 'ptmcmc':
        chainfile = f'{chaindir}/chain_1.txt'
        if not os.path.isfile(chainfile):
            print(f'Initiating PTMCMC sampling for {Nsamples} samples')
            p0 = np.hstack([prior_transform(theta, params) for theta in np.random.rand(1, Ndim)])
            cov = np.diag(np.ones(Ndim)*0.01**2)
            sampler = ptmcmc(Ndim, get_lnlike_optimised, get_lnprior, cov, outDir=chaindir, resume=True, 
                            loglargs=(params, dms, dmerrs, t, Tspan, Tref_dm, 
                                      toas, flags, timing_model, Nharm, verbose), 
                            logpargs=(params, ))
            sampler.sample(p0, Nsamples, SCAMweight=30, AMweight=15, DEweight=50)
            chains = np.loadtxt(chainfile)
        else:
            print(f'Using the existing chains : {chainfile}')
            chains = np.loadtxt(chainfile)

    elif sampling == 'nautilus':
        from nautilus import Sampler, Prior
        Nlive = 1000
        priors = Prior()
        p0 = np.array([prior_transform(theta, params) for theta in np.random.rand(1, Ndim)])
        for key in params.keys():
            priors.add_parameter(key, dist=(params[key]['pmin'], params[key]['pmax']))
        with mp.Pool() as pool:
            sampler = Sampler(priors, get_lnlike_optimised, n_live=Nlive, 
                              likelihood_args=(params, dms, dmerrs, t, Tspan, Tref_dm, 
                                               toas, flags, timing_model, Nharm, verbose), pool=pool)
            sampler.run(discard_exploration=False, verbose=True)
            samples, log_wt, log_l = sampler.posterior()
            weights = np.exp(log_wt)
            log_z = sampler.evidence()
            print(f'Log-evidence : {log_z:.4f}')
            samples_equal_weight = sampler.posterior(equal_weight=True)
            np.savetxt(chainfile, samples_equal_weight)
    else:
        raise Exception('Currently only emcee, ptmcmc and nautilus samplers are supported')

    # Plotting the posteriors
    if not sampling == 'nautilus':
        if not os.path.isfile(truthsfile):
            fig = corner.corner(chains[burn:, :], labels=param_labels, label_kwargs={"fontsize": 14, 'fontweight':'bold'}, 
                                plot_datapoints=False, hist_kwargs={"density": True}, labelpad=0.05, 
                                quantiles=(0.16, 0.5, 0.84), color='steelblue', 
                                show_titles=True, title_kwargs={"fontsize": 12, 'fontweight':'bold'})
            
            plt.suptitle(f'{psrname}', fontweight='bold', fontsize=16)
            plt.tight_layout()
            plt.savefig(f'{outdir}/{psrname}.noise.pdf')
        else:
            truths = np.array([float(f.split('\n')[0].split(' ')[-1]) for 
                               f in open(truthsfile, 'r').readlines() if not f.startswith('#')])
            print(f'Injected values : \n{truths}')
            if len(param_names) == 1:
                fig = corner.corner(chains[burn:], labels=param_labels, label_kwargs={"fontsize": 14, 'fontweight':'bold'}, 
                                    plot_datapoints=False, hist_kwargs={"density": True}, labelpad=0.05, 
                                    quantiles=(0.16, 0.5, 0.84), color='steelblue', show_titles=True, 
                                    title_kwargs={"fontsize": 12, 'fontweight':'bold'})
                ax = fig.gca()
                ax.axvline(truths[0], color='crimson')
            else:    
                fig = corner.corner(chains[burn:, :], labels=param_labels, label_kwargs={"fontsize": 14, 'fontweight':'bold'}, 
                                    plot_datapoints=False, hist_kwargs={"density": True}, labelpad=0.05, 
                                    quantiles=(0.16, 0.5, 0.84), color='steelblue', show_titles=True, 
                                    title_kwargs={"fontsize": 12, 'fontweight':'bold'}, truths=truths, truth_color='crimson')
            
            plt.suptitle(f'{psrname}', fontweight='bold', fontsize=16)
            plt.tight_layout()
            plt.savefig(f'{outdir}/{psrname}.noise.pdf')
    
    else:
        if not os.path.isfile(truthsfile):
            fig = corner.corner(samples, weights, labels=param_labels, label_kwargs={"fontsize": 14, 'fontweight':'bold'}, 
                                plot_datapoints=False, hist_kwargs={"density": True}, labelpad=0.05, 
                                quantiles=(0.16, 0.5, 0.84), color='steelblue', 
                                show_titles=True, title_kwargs={"fontsize": 12, 'fontweight':'bold'})
        else:
            truths = np.array([float(f.split('\n')[0].split(' ')[-1]) for 
                               f in open(truthsfile, 'r').readlines() if not f.startswith('#')])
            print(f'Injected values : \n{truths}')

    if os.path.isfile(truthsfile):
        swdet_param_names = [pname for pname in param_names if 'NE_SW' in pname]
        if len(swdet_param_names):
            swdet = get_swdetsignal(t, Tref_dm, toas, timing_model, theta=truths, params=params)
            fig, ax = plt.subplots(figsize=(15, 8), dpi=100, nrows=2, ncols=1, sharex=True)
            ax[0].errorbar(t*u.s.to(u.day), dms, yerr=dmerrs, fmt='o', ms=7, c='crimson', alpha=0.2, \
                         elinewidth=1.0, capsize=2, capthick=0.5, ecolor='k', label='Simulated DMs')
            ax[0].scatter(t*u.s.to(u.day), swdet, s=7, alpha=0.5, c='k', label='SW model DMs')
            ax[0].ticklabel_format(useOffset=False)
            ax[0].legend(loc='upper left')
            ax[0].grid(ls='dashed')
            _ = ax[0].set_ylabel(rf'DM [pc/cm$^3$]', fontweight='bold', fontsize=13)
            ax[1].scatter(t*u.s.to(u.day), dms-swdet, alpha=0.5, c='r', label='Residuals')
            _ = ax[1].set_xlabel(f'MJD', fontweight='bold', fontsize=13)
            _ = ax[1].set_ylabel(rf'$\delta$DM [pc/cm$^3$]', fontweight='bold', fontsize=13)
            ax[1].ticklabel_format(useOffset=False, style='plain')
            ax[1].grid(ls='dashed')
            plt.tight_layout()
            plt.savefig(f'{outdir}/{psrname}.swcomp.pdf')
    
    n_realisations = 20
    if not len(param_names) == 1:
        print(f'Generating {n_realisations} random noise process realisations from the posteriors')
        plot_realisations(n_realisations, chains, params, dms, dmerrs, t, Tspan, 
                          Tref_dm, flags, parfile, Nharm, outdir, psrname, toas, timing_model, save=True, 
                          add_dms=True, no_title=True)
    
    process_finish = time.time()
    print(f'\nThe process took {(process_finish-process_start)/60:.4f} minutes to finish')
