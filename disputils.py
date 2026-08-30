import numpy as np
import astropy.units as u
from pint import DMconst
import matplotlib.pyplot as plt
import scipy.linalg as sl
import os
import sys

def get_freqs(Tspan, Nharm=None):
    '''
        Function to get the Fourier-domain frequencies
        InParams: Time span in seconds and possibly number of harmonics
        Output: Array of evenly-spaced linear Fourier-frequencies
    '''
    if not Nharm:
        Tspan_years = Tspan*u.s.to(u.year)
        Nharm = int(np.floor(Tspan_years*12))
    
    f = np.linspace(1/Tspan, Nharm/Tspan, Nharm)
    
    return f

def get_Nmat(dmerrs, theta, params, flags):
    '''
        Function to get the white-noise covariance matrix
        InParams: DM errors, samples, parameter vector and DM flags
        Output: DM White-noise covariance matrix
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
        InParams: Array of linear Fourier-frequencies, Time span in seconds, 
                  power-law amplitude and spectral index
        Output: Power Spectral Density of Gaussian Process
    '''
    yr = 1*u.year.to(u.s)
    fyr = 1/yr
    pl = ((10**log10_A)**2/(12*np.pi**2*Tspan))*f**(-gamma)*fyr**(gamma-3)
    
    return pl

def get_parfile_param(parfile, param):
    '''
        Function to get the parfile entry for a given parameter
        InParams: Pulsar ephemeris file and parameter name
        Output: Value of parameter as per the ephemeris file
    '''
    if os.path.isfile(parfile):
        parlines = [f.split('\n')[0] for f in open(parfile, 'r').readlines() 
                    if not f.startswith('#')]
        
        param_in_parfile = False
        for line in parlines:
            if param in line:
                entries = line.split()
                if param in entries:
                    param_in_parfile = True
                    paramval = entries[np.where(np.array(entries)==param)[0][0]+1]

        if param_in_parfile:
            return paramval
        else:
            print(f'\nThe parameter {param} is not present in the ephemeris file')
            print(f'\nThe process will terminate')
            sys.exit()
    else:
        print(f'\nThe pulsar ephmeris file {parfile} does not exist')
        print(f'\nThe process will terminate')
        sys.exit()

# Adding a check if the SolarWindDispersion component is present in the Timing Model
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
                                              Tspan, Tref_dm, toas, flags, timing_model, Nharm)

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
                         Tref_dm, toas, flags, timing_model, Nharm=None):
    '''
        Function to get the log-likelihood in an optimised form for params
    '''
    f = get_freqs(Tspan, Nharm)
    N = get_Nmat(dmerrs, theta, params, flags)
    Phi = get_Phimat(f, Tspan, theta, params)
    U = get_Umat(t, f, Tspan, Tref_dm, params, toas, timing_model)
    
    # DM will remain unaffected if there are no deterministic SW parameters
    swdet = get_swdetsignal(t, Tref_dm, toas, timing_model, theta, params)
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

def read_dmfile(dmfile, return_flags=False):
    '''
        Function to extract the MJD, DM and DM errors from a DM time series file
        InParams: DM time series file and option to return flags
        Output: MJDs converted to s, DMs, DM errors and DM flags (if required)
    '''
    if os.path.isfile(dmfile):
        
        filelines = [f for f in open(dmfile, 'r').readlines() if not 
                     f.startswith('#')]
        
        t = np.array([np.longdouble(f.split('\n')[0].split(' ')[0]) for 
                      f in filelines])*u.day.to(u.s)
        
        dms = np.array([float(f.split('\n')[0].split(' ')[1]) for 
                        f in filelines])
        
        dmerrs = np.array([float(f.split('\n')[0].split(' ')[2]) for 
                           f in filelines])

        if not return_flags:
            return t, dms, dmerrs

        else:
            # If there are only few epochs with specific flags
            sizearr = np.array([len(f.split('\n')[0].split(' ')) 
                                for f in filelines])
            
            maskarr = np.array([size == 4 for size in sizearr])

            if not np.any(maskarr == True):
                flags = np.array(['no-flag' for item in range(t.size)])

            else:
                flags = np.array([f.split('\n')[0].split(' ')[-1] if 
                                  mask == True else 'no-flag' 
                                  for mask, f in zip(maskarr, filelines)])

            return t, dms, dmerrs, flags

    else:
        print(f'\nThe DM time series file {dmfile} does not exist')
        print(f'\nThe process will terminate')
        sys.exit()

class DispParam:
    '''
        Class to define a DispParam parameter based on a given parameter file
    '''
    def __init__(self, idx, **kwargs):
        self.idx = idx
        for key, value in kwargs.items():
            setattr(self, key, value)
            
        # Using the PINT naming scheme to defined correlated noises
        
        self.is_red = True if 'TN' in self.name else False
        self.is_white = True if ('EFAC' in self.name or 'EQUAD' in self.name) else False
        self.is_deterministic = True if (not self.is_red and not self.is_white) else False
        self._get_param_labels()
        
        self._print_param_info()
    
    def _get_param_labels(self):
        '''
            Function to generate parameter labels
        '''
        if not self.is_red:
            self.label = self.name
        else:
            if 'DM' in self.name:
                if 'AMP' in self.name:
                    self.label = r'$\log_{10}A_{\rm DMN}$'
                else:
                    self.label = r'$\gamma_{\rm DMN}$'
            elif 'SW' in self.name:
                if 'AMP' in self.name:
                    self.label = r'$\log_{10}A_{\rm SWN}$'
                else:
                    self.label = r'$\gamma_{\rm SWN}$'
            else:
                raise ValueError(f'Unrecognised parameter name: {self.name}')

    def _print_param_info(self):
        '''
            Function to print the parameter attributes
        '''
        print(f'\nParameter: {self.name}')
        print(f'\t - Prior Distribution: {self.dist.capitalize()}({self.min}, {self.max})')
        print(f'\t - Flag: {self.flag}')
        print(f'\t - Label: {self.label}')
        print(f'\t - Is red noise parameter: {self.is_red}')
        print(f'\t - Is white noise parameter: {self.is_white}')
        print(f'\t - Is deterministic parameter: {self.is_deterministic}')
        

def get_DispParams(paramfile):
    '''
        Function to return a list of DispParam objects for a parameter file
        InParams: Path to the parameter file
        Output: A list of DispParam objects
    '''
    if os.path.isfile(paramfile):
        paramlines = np.array([f.split('\n')[0] for 
                               f in open(paramfile, 'r').readlines() 
                               if not f.startswith('#')])
        params = []
        for pidx, paramline in enumerate(paramlines):
            entries = paramline.split()
            pname, pmin, pmax, pdist = entries[:4]
            pflag = entries[4] if len(entries) > 4 else 'no-flag'
            param = DispParam(pidx, name=pname, min=pmin, max=pmax, dist=pdist, flag=pflag)
            params.append(param)
    
        return params
        
    else:
        print(f'\nThe parameter file {paramfile} does not exist')
        print(f'\nThe process will terminate')
        sys.exit()




## TBD
# 1. Defining class for parameters -- DONE
# 2. Defining a wrapper function to directly create parmaeters as per the file -- DONE
# 3. Updating functions wherever the parameters are being accessed as a dictionary















    