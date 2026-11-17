import numpy as np
import astropy.units as u
from pint import DMconst
import matplotlib.pyplot as plt
import scipy.linalg as sl
import os
import sys
import emcee
import corner
import multiprocessing as mp
import pocomc as pc
from scipy import stats
import json
from math import factorial
_cache = {}

class DispParam:
    '''
        Class to define a DispParam parameter based on a given parameter file
    '''
    def __init__(self, idx, **kwargs):
        self.idx = idx
        for key, value in kwargs.items():
            setattr(self, key, value)
            
        self.name = self.name.upper()
        if not self.dist.capitalize() == 'Uniform':
            raise ValueError(f'Only Uniform prior distributions are allowed')
            
        self.is_red = True if 'TN' in self.name else False
        self.is_white = True if ('EFAC' in self.name or 'EQUAD' in self.name) else False
        self.is_deterministic = True if (not self.is_red and not self.is_white) else False
        self._set_param_labels()
        
        if self.is_white or 'JUMP' in self.name:
            if not np.any(self.mask):
                print(f'\nNo compatible flag present in DMs for {self.name} with provided flag: {self.flag}')
                print(f'\nThe process will terminate')
                sys.exit()
            else:
                self.name = f'{self.name}_{self.flag}'
        else:
            self.mask = np.array([True for val in self.mask])

        self._print_param_info()

    def __repr__(self):
        return f'{self.name}'
    
    def _set_param_labels(self):
        '''
            Function to generate parameter labels
        '''
        if not self.is_red:
            self.label = self.name
        else:
            if 'DM' in self.name:
                if 'AMP' in self.name:
                    self.label = r'$\log_{10}A_{\rm DMN}$'
                elif 'GAM' in self.name:
                    self.label = r'$\gamma_{\rm DMN}$'
                elif 'RHO' in self.name:
                    try:
                        idx = int(self.name.split('TNDMRHO')[-1])
                        self.label = rf'$\rho^{idx}$'+r'$_{\rm DMN}$'
                    except Exception:
                        self.label = r'$\rho_{\rm DMN}$'
                elif 'DEL' in self.name:
                    self.label = r'$\delta_{\rm DMN}$'
                elif 'KAP' in self.name:
                    self.label = r'$\kappa_{\rm DMN}$'
                elif 'LFB' in self.name:
                    self.label = r'$\log_{10}f_{\rm b, \rm DMN}$'
                else:
                    self.label = self.name
            elif 'SW' in self.name:
                if 'AMP' in self.name:
                    self.label = r'$\log_{10}A_{\rm SWN}$'
                elif 'GAM' in self.name:
                    self.label = r'$\gamma_{\rm SWN}$'
                elif 'RHO' in self.name:
                    try:
                        idx = int(self.name.split('TNSWRHO')[-1])
                        self.label = rf'$\rho^{idx}$'+r'$_{\rm SWN}$'
                    except Exception:
                        self.label = r'$\rho_{\rm SWN}$'
                elif 'DEL' in self.name:
                    self.label = r'$\delta_{\rm SWN}$'
                elif 'KAP' in self.name:
                    self.label = r'$\kappa_{\rm SWN}$'
                elif 'LFB' in self.name:
                    self.label = r'$\log_{10}f_{\rm b, \rm SWN}$'
                else:
                    self.label = self.name
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

    def set_values(self, **kwargs):
        '''
            Function to set the median and standard deviation of parameters
        '''
        for key, value in kwargs.items():
            setattr(self, key, value)

def get_DispParams(paramfile, flags):
    '''
        Function to return a list of DispParam objects for a parameter file
        InParams: Path to the parameter file
        Output: A list of DispParam objects
    '''
    if os.path.isfile(paramfile):
        if not isinstance(flags, np.ndarray):
            flags = np.array(flags)
            
        paramlines = np.array([f.split('\n')[0] for f in open(paramfile, 'r').readlines() 
                               if not f.startswith('#') and len(f.split('\n')[0]) >= 4])
        params = []
        for pidx, paramline in enumerate(paramlines):
            entries = paramline.split()
            pname, pmin, pmax, pdist = entries[:4]
            pflag = entries[4] if len(entries) > 4 else 'no-flag'
            pmask = np.array(flags == pflag)
            param = DispParam(pidx, name=pname, min=pmin, max=pmax, dist=pdist, 
                              flag=pflag, mask=pmask)
            params.append(param)
    
        return params
        
    else:
        print(f'\nThe parameter file {paramfile} does not exist')
        print(f'\nThe process will terminate')
        sys.exit()

def read_dmfile(dmfile, return_flags=False):
    '''
        Function to extract the MJD, DM and DM errors from a DM time series file
        InParams: path to the DM time series file and option to return flags
        Output: arrays of MJDs converted to s, DMs in pc/cm^3, DM errors in pc/cm^3 and array of DM flags
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
            return t, dms, dmerrs, None

        else:
            # If there are only few epochs with specific flags
            sizearr = np.array([len(f.split('\n')[0].split(' ')) 
                                for f in filelines])
            
            maskarr = np.array([size == np.max(sizearr) for size in sizearr])

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

def plot_dmseries(t, dms, dmerrs, outdir, psrname):
    '''
        Function to plot a given DM time series
        InParams: time in s, DM time series in pc/cm^3, DM erros in pc/cm^3, path to the output directory
                  and pulsar name
        Output: None
    '''
    print(f'\nPlotting the DM time series for {psrname}')
    fig = plt.figure(figsize=(10, 4), dpi=200)
    plt.errorbar(t*u.s.to(u.day), dms, yerr=dmerrs, fmt='o', ms=7, c='crimson', alpha=0.5, \
                 elinewidth=1.0, capsize=2, capthick=0.5, ecolor='k')
    plt.ticklabel_format(useOffset=False)
    plt.grid(ls='dashed')
    plt.xlabel(f'MJD', fontweight='bold', fontsize=13)
    plt.ylabel(rf'DM [pc/cm$^3$]', fontweight='bold', fontsize=13)
    plt.title(rf'{psrname}, Median DMerr = {np.median(dmerrs):.2e} pc/cm$^3$', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{outdir}/{psrname}.DM.time.series.pdf', dpi=200)

def get_freqs(Tspan, Nharm=None):
    '''
        Function to get the Fourier-domain frequencies
        InParams: time span in s and possibly number of harmonics
        Output: array of uniformly-spaced linear Fourier-frequencies
    '''
    if not Nharm:
        print(f'\nSince no harmonics are provided, setting Nharm as per 1/month')
        Tspan_years = Tspan*u.s.to(u.year)
        Nharm = int(np.floor(Tspan_years*12))
    
    f = np.linspace(1/Tspan, Nharm/Tspan, Nharm)
    return f

def powerlaw(f, Tspan, log10_A, gamma):
    '''
        Function to get the Fourier-domain power-law for a given basis
        InParams: array of linear Fourier-frequencies, time span in seconds, 
                  power-law amplitude and spectral index
        Output: the Power Spectral Density of a Fourier-domain free-spectral process
    '''
    yr = 1*u.year.to(u.s)
    fyr = 1/yr
    pl = ((10**log10_A)**2/(12*np.pi**2*Tspan))*f**(-gamma)*fyr**(gamma-3)
    return pl

def free_spectrum(log10_rho, Tspan):
    '''
        Function to get the free-spectrum density for a given basis
        InParams: free spectrum component and time span in seconds
        Output: the Power Spectral Density of a free-spectrum process
    '''
    return 10**(2*log10_rho)/Tspan

def broken_powerlaw(f, log10_A, gamma, delta, log10_fb, kappa, Tspan):
    '''
        Function to get the broken-powerlaw spectral density for a given basis
        InParams: array of linear Fourier-frequencies, power-law amplitude, pre-break spectral
                  index (for f<f_b), post-break spectral index (for f>f_b), logarithm of the 
                  transition frequency in 1/s, parameter for smoothness of transition and 
                  time span in seconds
        Output: the Power Spectral Dnesity of a Fourier-dommain broken-powerlaw process
    '''
    yr = 1*u.year.to(u.s)
    fyr = 1/yr
    fb = 10**log10_fb
    hcf = 10**(log10_A)*((f/fyr)**((3-gamma)/2))*(1+(f/fb)**(1/kappa))**(kappa*(gamma-delta)/2)
    pl = hcf**2/(12*np.pi**2*f**3*Tspan)
    return pl

def get_Umat(t, Tspan, Tref_dm, Tref_sw, dmnparams, swnparams, geometric_factor, DMNharm, SWNharm, fref, 
             dmdetparams, dmorders, fit_DMdet, fit_DM0, fit_DM1, fit_DM2):
    '''
        Function to get the basis matrix for Fourier domain red noise process
        This implementation considers SWEPOCH to be the same as the DMEPOCH
        Also, the entries for DMGP and SWGP are scaled with fref**2/DMconst to make them coherent
        with the definitions in ENTERPRSIE (taking fref as 1400 MHz for comparison)
        InParams: time in s, time span in s, DM reference epoch in s, SW reference epoch in s, 
                  array of DM noise parameters, array of SW noise parameters, SW geometric factor (PINT), 
                  DM noise harmonics, SW noise harmonics and chromatic reference frequency in MHz
        Output: the DM basis matrix
    '''
    if dmnparams.size:
        f_dmn = get_freqs(Tspan, DMNharm)
        convfactor = fref**2/DMconst
    else:
        f_dmn = np.zeros(0)

    if swnparams.size:
        f_swn = get_freqs(Tspan, SWNharm)
        dm_sw = get_dm_sw(n_earth=1.0, geometric_factor=geometric_factor)
    else:
        f_swn = np.zeros(0)

    if not fit_DMdet:
        U = [[0 for val in range(3)] for i in range(t.size)]
        for i in range(t.size):
            U[i][0] = 1.0
            U[i][1] = (t[i]-Tref_dm)*u.s.to(u.year)
            U[i][2] = 0.5*((t[i]-Tref_dm)*u.s.to(u.year))**2
            if f_dmn.size:
                dmn = np.zeros(2*f_dmn.size)
                for j in range(f_dmn.size):
                    uval = 2*np.pi*(j+1)*(t[i]-Tref_dm)/Tspan
                    dmn[2*j] = np.sin(uval)*convfactor.value
                    dmn[2*j+1] = np.cos(uval)*convfactor.value
                U[i].extend(dmn)
            if f_swn.size:
                swn = np.zeros(2*f_swn.size)
                for j in range(f_swn.size):
                    uval = 2*np.pi*(j+1)*(t[i]-Tref_sw)/Tspan
                    swn[2*j] = np.sin(uval)*dm_sw[i]
                    swn[2*j+1] = np.cos(uval)*dm_sw[i]
                U[i].extend(swn)
    else:
        U = [[0 for val in range(3-len(dmdetparams))] for i in range(t.size)]
    
    return np.array(U)

def get_Nmat(theta, dmerrs, dispefacparams, dispequadparams, vectorise=False):
    '''
        Function to get the DM white-noise covariance matrix assuming logarithmic quadrature factors.
        Currently it does not support dense jitter-affected correlations
        InParams: parameter sample vector, DM errors in pc/cm^3, array of DISPEFAC paramters, 
                  array of DISPEQUAD parameters, and flag to vectorise the likelihood computations
        Output: the DM White-noise covariance matrix
    '''
    if not np.any(dispefacparams) and not np.any(dispequadparams):
        raise ValueError(f'No white noise parameters present in the noise model')
    else:
        if not vectorise:
            dispefacs = np.ones(dmerrs.size)
            dispequads = -np.inf*np.ones(dmerrs.size)
            if np.any(dispefacparams):
                for param in dispefacparams:
                    dispefacs = np.where(param.mask, theta[param.idx], dispefacs)
            if np.any(dispequadparams):
                for param in dispequadparams:
                    dispequads = np.where(param.mask, theta[param.idx], dispequads)
            Nmat = np.array(dispefacs**2 * (dmerrs**2 + 10**(2*dispequads)))
        else:
            dispefacs = np.ones((theta.shape[0], dmerrs.size))
            dispequads = -np.inf*np.ones((theta.shape[0], dmerrs.size))
            if np.any(dispefacparams):
                for param in dispefacparams:
                    dispefacs = np.where(param.mask[None, :], theta[:, param.idx, None], dispefacs)
            if np.any(dispequadparams):
                for param in dispequadparams:
                    dispequads = np.where(param.mask[None, :], theta[:, param.idx, None], dispequads)
            Nmat = np.array(dispefacs**2 * (dmerrs[None, :]**2 + 10**(2*dispequads)))
        return Nmat

def get_Phimat(theta, Tspan, dmnparams, swnparams, DMNharm, SWNharm, psd, vectorise=False):
    '''
        Function to get the Gaussian Process kernel matrix for power-law processes
        InParams: parameter sample vector, time span in s, array of DM noise parameters, 
                  array of SW noise parameters, DM noise harmonics, SW noise harmonics, the PSD of 
                  Fourier-domain noise processes and flag to vectorise the likelihood computations
        Output: the GP kernel matrix for a PL spectrum with analytically marginalised DM Polynomial
    '''
    if dmnparams.size:
        f_dmn = get_freqs(Tspan, DMNharm)
        
        if psd == 'powerlaw':
            try: log10_A_dmn = [par for par in dmnparams if par.name == 'TNDMAMP'][0]
            except Exception as e:
                raise ValueError(f'Cannot extract DMN amplitude from parameter file')
            try: gamma_dmn = [par for par in dmnparams if par.name == 'TNDMGAM'][0]
            except Exception as e:
                raise ValueError(f'Cannot extract DMN spectral index from parameter file')
        
        elif psd == 'free-spectrum':
            try: log10_rho_dmn = np.array([par for par in dmnparams if 'TNDMRHO' in par.name])
            except Exception as e:
                raise ValueError(f'Cannot extract DMN amplitude from parameter file')
            if not f_dmn.size == log10_rho_dmn.size:
                raise ValueError(f'Number of free-spectrum components should match the number of harmonics')
                
        elif psd == 'broken-powerlaw':
            try: log10_A_dmn = [par for par in dmnparams if par.name == 'TNDMAMP'][0]
            except Exception as e:
                raise ValueError(f'Cannot extract DMN amplitude from parameter file')
            try: gamma_dmn = [par for par in dmnparams if par.name == 'TNDMGAM'][0]
            except Exception as e:
                raise ValueError(f'Cannot extract DMN pre-break spectral index from parameter file')
            try: delta_dmn = [par for par in dmnparams if par.name == 'TNDMDEL'][0]
            except Exception as e:
                raise ValueError(f'Cannot extract DMN post-break spectral index from parameter file')
            try: kappa_dmn = [par for par in dmnparams if par.name == 'TNDMKAP'][0]
            except Exception as e:
                raise ValueError(f'Cannot extract DMN smoothing index from parameter file')
            try: log10_fb_dmn = [par for par in dmnparams if par.name == 'TNDMLFB'][0]
            except Exception as e:
                raise ValueError(f'Cannot extract DMN break-frequency from parameter file')
        else:
            raise NotImplementedError(f'Currently only powerlaw, free-spectrum and broken powerlaw PSDs are supported')
    else:
        f_dmn = np.zeros(0)

    if swnparams.size:
        f_swn = get_freqs(Tspan, SWNharm)
        
        if psd == 'powerlaw':
            try: log10_A_swn = [par for par in swnparams if par.name == 'TNSWAMP'][0]
            except Exception as e:
                raise ValueError(f'Cannot extract SWN amplitude from parameter file')
            try: gamma_swn = [par for par in swnparams if par.name == 'TNSWGAM'][0]
            except Exception as e:
                raise ValueError(f'Cannot extract SWN spectral index from parameter file')
        
        elif psd == 'free-spectrum':
            try: log10_rho_swn = np.array([par for par in swnparams if 'TNSWRHO' in par.name])
            except Exception as e:
                raise ValueError(f'Cannot extract SWN amplitude from parameter file')
            if not f_swn.size == log10_rho_swn.size:
                raise ValueError(f'Number of free-spectrum components should match the number of harmonics')
        
        elif psd == 'broken-powerlaw':
            try: log10_A_swn = [par for par in swnparams if par.name == 'TNSWAMP'][0]
            except Exception as e:
                raise ValueError(f'Cannot extract SWN amplitude from parameter file')
            try: gamma_swn = [par for par in swnparams if par.name == 'TNSWGAM'][0]
            except Exception as e:
                raise ValueError(f'Cannot extract SWN pre-break spectral index from parameter file')
            try: delta_swn = [par for par in swnparams if par.name == 'TNSWDEL'][0]
            except Exception as e:
                raise ValueError(f'Cannot extract SWN post-break spectral index from parameter file')
            try: kappa_swn = [par for par in swnparams if par.name == 'TNSWKAP'][0]
            except Exception as e:
                raise ValueError(f'Cannot extract SWN smoothing index from parameter file')
            try: log10_fb_swn = [par for par in swnparams if par.name == 'TNSWLFB'][0]
            except Exception as e:
                raise ValueError(f'Cannot extract SWN break-frequency from parameter file')
        else:
            raise NotImplementedError(f'Currently only powerlaw, free-spectrum and broken powerlaw PSDs are supported')
    else:
        f_swn = np.zeros(0)
    
    if not vectorise:
        Phi = []
        Phi.extend([1e+40, 1e+40, 1e+40])
        if f_dmn.size:
            dmn = np.zeros(2*f_dmn.size)
            for i in range(f_dmn.size):
                if psd == 'powerlaw':
                    dmn[2*i] = dmn[2*i+1] = powerlaw(f_dmn[i], Tspan, 
                                                     log10_A=theta[log10_A_dmn.idx], 
                                                     gamma=theta[gamma_dmn.idx])
                elif psd == 'free-spectrum':
                    dmn[2*i] = dmn[2*i+1] = free_spectrum(log10_rho=theta[log10_rho_dmn[i].idx], 
                                                          Tspan=Tspan)
                elif psd == 'broken-powerlaw':
                    dmn[2*i] = dmn[2*i+1] = broken_powerlaw(f_dmn[i], 
                                                            log10_A=theta[log10_A_dmn.idx], 
                                                            gamma=theta[gamma_dmn.idx], 
                                                            delta=theta[delta_dmn.idx], 
                                                            log10_fb=theta[log10_fb_dmn.idx], 
                                                            kappa=theta[kappa_dmn.idx], 
                                                            Tspan=Tspan)
            Phi.extend(dmn)
        if f_swn.size:
            swn = np.zeros(2*f_swn.size)
            for i in range(f_swn.size):
                if psd == 'powerlaw':
                    swn[2*i] = swn[2*i+1] = powerlaw(f_swn[i], Tspan, 
                                                     log10_A=theta[log10_A_swn.idx], 
                                                     gamma=theta[gamma_swn.idx])
                elif psd == 'free-spectrum':
                    swn[2*i] = swn[2*i+1] = free_spectrum(log10_rho=theta[log10_rho_swn[i].idx], 
                                                          Tspan=Tspan)
                elif psd == 'broken-powerlaw':
                    swn[2*i] = swn[2*i+1] = broken_powerlaw(f_swn[i], 
                                                            log10_A=theta[log10_A_swn.idx], 
                                                            gamma=theta[gamma_swn.idx], 
                                                            delta=theta[delta_swn.idx], 
                                                            log10_fb=theta[log10_fb_swn.idx], 
                                                            kappa=theta[kappa_swn.idx], 
                                                            Tspan=Tspan)
            Phi.extend(swn)
    else:
        Phi = [[] for walker in range(theta.shape[0])]
        for walker in range(theta.shape[0]):
            Phi[walker].extend([1e+40, 1e+40, 1e+40])
            if f_dmn.size:
                dmn = np.zeros(2*f_dmn.size)
                for i in range(f_dmn.size):
                    if psd == 'powerlaw':
                        dmn[2*i] = dmn[2*i+1] = powerlaw(f_dmn[i], Tspan, 
                                                         log10_A=theta[walker, log10_A_dmn.idx], 
                                                         gamma=theta[walker, gamma_dmn.idx])
                    elif psd == 'free-spectrum':
                        dmn[2*i] = dmn[2*i+1] = free_spectrum(log10_rho=theta[walker, log10_rho_dmn[i].idx], 
                                                              Tspan=Tspan)
                    elif psd == 'broken-powerlaw':
                        dmn[2*i] = dmn[2*i+1] = broken_powerlaw(f_dmn[i], 
                                                                log10_A=theta[walker, log10_A_dmn.idx], 
                                                                gamma=theta[walker, gamma_dmn.idx], 
                                                                delta=theta[walker, delta_dmn.idx], 
                                                                log10_fb=theta[walker, log10_fb_dmn.idx], 
                                                                kappa=theta[walker, kappa_dmn.idx], 
                                                                Tspan=Tspan)
                Phi[walker].extend(dmn)
            if f_swn.size:
                swn = np.zeros(2*f_swn.size)
                for i in range(f_swn.size):
                    if psd == 'powerlaw':
                        swn[2*i] = swn[2*i+1] = powerlaw(f_swn[i], Tspan, 
                                                         log10_A=theta[walker, log10_A_swn.idx], 
                                                         gamma=theta[walker, gamma_swn.idx])
                    elif psd == 'free-spectrum':
                        swn[2*i] = swn[2*i+1] = free_spectrum(log10_rho=theta[walker, log10_rho_swn[i].idx], 
                                                              Tspan=Tspan)
                    elif psd == 'broken-powerlaw':
                        swn[2*i] = swn[2*i+1] = broken_powerlaw(f_swn[i], log10_A=theta[walker, log10_A_swn.idx], 
                                                                gamma=theta[walker, gamma_swn.idx], 
                                                                delta=theta[walker, delta_swn.idx], 
                                                                log10_fb=theta[walker, log10_fb_swn.idx], 
                                                                kappa=theta[walker, kappa_swn.idx], 
                                                                Tspan=Tspan)
                Phi[walker].extend(swn)
    
    return np.array(Phi)

def get_sw_geometry(timing_model, toas):
    '''
        Function to return the SW geometric factor for a given set of ToAs
        InParams: timing model object (PINT) and ToAs object (PINT)
        Output: the SW geometric factor
    '''
    if 'SolarWindDispersion' in timing_model.components:
        geometric_factor = timing_model.components['SolarWindDispersion'].solar_wind_geometry(toas).value
    else:
        raise KeyError(f'The SolarWindDispersion component is absent in the timing model')
    
    return geometric_factor

def get_dm_sw(n_earth, geometric_factor, vectorise=False):
    '''
        Function to return the DM contribution from solar wind
        InParams: solar-wind electron density in 1/cm^3, SW geometric factor (PINT)
                  and flag to vectorise the likelihood computations
        Output: the DM induced by the solar-wind electron density
    '''
    if not vectorise:
        dm_sw = n_earth*geometric_factor
    else:
        dm_sw = n_earth[:, None]*geometric_factor
    
    return dm_sw

def get_swdetsignal(t, Tref_sw, theta, swdetparams, orders, timing_model, geometric_factor, vectorise=False):
    '''
        Function to get a deterministic signal for the solar-wind process
        InParams: time in s, SW reference epoch in s, parameter sample vector, 
                  array of deterministic SW parameters, array of their orders, timing model object (PINT), 
                  SW geometric factor (PINT) and flag to vectorise the likelihood computations
        Output: the DM contribution from deterministic SW process with a polynomial form
    '''
    if not len(swdetparams):
        if not vectorise:
            signal = np.zeros(t.size)
        else:
            signal = np.zeros((theta.shape[0], t.size))
    else:
        if not vectorise:
            signal = np.zeros(t.size)
            for param, order in zip(swdetparams, orders):
                signal += get_dm_sw(theta[param.idx], 
                                    geometric_factor, 
                                    vectorise)*((t-Tref_sw)*u.s.to(u.year))**order/factorial(order)
        else:
            signal = np.zeros((theta.shape[0], t.size))
            for param, order in zip(swdetparams, orders):
                signal += get_dm_sw(theta[:, param.idx], 
                                    geometric_factor, 
                                    vectorise)*((t-Tref_sw)*u.s.to(u.year))**order/factorial(order)

    try: 
        fid_dm = timing_model.DM.value
        if not fid_dm:
            fid_dm = 0.0
    except Exception:
        print(f'\nNo fiducial DM present in the timing model')
        print(f'\nUsing DM = 0 as reference')
        fid_dm = 0.0
        
    signal += fid_dm
    return signal

def get_dmjump_signal(theta, t, dmjumpparams, vectorise=False):
    '''
        Function to get the deterministic process for DMJUMPs
        InParams: parameter sample vector, time in s, array of DMJUMP parameters, and flag to
                  vectorise the likelihood computations
        Output: the DM contribution from deterministic JUMPs for distinct DM groups
    '''
    if not vectorise:
        dmjumps = np.zeros(t.size)
        if np.any(dmjumpparams):
            for param in dmjumpparams:
                dmjumps = np.where(param.mask, theta[param.idx], dmjumps)
    else:
        dmjumps = np.zeros((theta.shape[0], t.size))
        if np.any(dmjumpparams):
            for param in dmjumpparams:
                dmjumps = np.where(param.mask[None, :], theta[:, param.idx, None], dmjumps)
                
    return dmjumps

def get_dmdetsignal(t, Tref_dm, theta, dmdetparams, orders, timing_model, vectorise=False):
    '''
        Function to get a deterministic signal for the DM process
        InParams: time in s, DM reference epoch in s, parameter sample vector, 
                  array of deterministic DM parameters, array of their orders, timing model object (PINT) 
                  and flag to vectorise the likelihood computations
        Output: the DM contribution from deterministic SW process with a polynomial form
    '''
    if not len(dmdetparams):
        if not vectorise:
            signal = np.zeros(t.size)
        else:
            signal = np.zeros((theta.shape[0], t.size))
    else:
        if not vectorise:
            signal = np.zeros(t.size)
            for param, order in zip(dmdetparams, orders):
                signal += theta[param.idx]*((t-Tref_dm)*u.s.to(u.year))**order/factorial(order)
        else:
            signal = np.zeros((theta.shape[0], t.size))
            for param, order in zip(dmdetparams, orders):
                signal += theta[:, param.idx]*((t-Tref_dm)*u.s.to(u.year))**order/factorial(order)
    
    try:
        fid_dm = timing_model.DM.value
        if not fid_dm:
            fid_dm = 0.0
    except Exception:
        print(f'\nNo fiducial DM present in the timing model')
        print(f'\nUsing DM = 0 as reference')
        fid_dm = 0.0
        
    signal += fid_dm
    return signal

def prior_transform(params, show=True, theta=None):
    '''
        Function to get the prior transform for dispersive noise parameters
        InParams: a list of DispParam parameters, flag to show the results and parameter sample vector
        Output: the transformed prior distributions for the parameters if parameter sample vector
                is provided, otherwise an array of prior distributions
    '''
    if isinstance(params, list):
        for param in params:
            if not isinstance(param, DispParam):
                raise TypeError(f'Invalid parameter type for {param.name}')
    else:
        raise TypeError(f'A list of DispParam parameters should be provided')
    
    mins = np.array([float(param.min) for param in params])
    maxs = np.array([float(param.max) for param in params])
    spans = maxs - mins
    if not np.any(theta):
        priors = np.array([stats.uniform(loc=low, scale=span) for low, span in zip(mins, spans)])
        if show:
            print(f'\nPrior Distributions:')
            for pname, dist in zip([param.name for param in params], priors):
                a, b = dist.support()
                print(f'\n\t{pname}: U({a:.3g}, {b:.3g})')
    else:
        priors = spans * theta + mins
    
    return priors

def UniformPrior(value, pmin, pmax):
    '''
        Function to give uniform prior distribution
        InParams: parameter sample value, parameter minimum value and parameter maximum value
        Output: the Uniform prior value for a given parameter sample
    '''
    prior = ((value >= pmin) & (value <= pmax)) / (pmax - pmin)
    
    return prior

def get_lnprior(theta, params, vectorise=False):
    '''
        Function to get the log-prior for different parameters
        InParams: parameter sample vector and list of DispParam parameters and 
                  flag to vectorise the likelihood computations
        Output: the log-prior value as per the parameter samples
    '''
    mins = np.array([float(param.min) for param in params])
    maxs = np.array([float(param.max) for param in params])

    if not vectorise:
        lnprior = np.sum([np.log(UniformPrior(value, pmin, pmax)) for 
                          value, pmin, pmax in zip(theta, mins, maxs)])
    else:
        lnprior = np.sum([[np.log(UniformPrior(value, pmin, pmax)) for 
                          value, pmin, pmax in zip(theta[idx, :], mins, maxs)]
                          for idx in range(theta.shape[0])], axis=1)
    return lnprior

def get_lnpost(theta, params, dms, dmerrs, t, Tspan, Tref_dm, Tref_sw, toas, flags, timing_model, DMNharm, 
               SWNharm, fref, dispefacparams, dispequadparams, dmnparams, swnparams, swdetparams, orders, 
               dmjumpparams, U, geometric_factor, psd, vectorise, dmdetparams, dmorders, fit_DMdet, 
               fit_DM0, fit_DM1, fit_DM2):
    '''
        Function to get the log-posterior for different paramters
        InParams: parameter sample vector, list of DispParam parameters, DMs in pc/cm^3, DM errors in pc/cm^3, 
                  time array in s, time span in s, reference epoch for DM, reference epoch for SW, 
                  ToAs object (PINT), list of DM flags, timing model object (PINT), DMN harmonics, 
                  SWN harmonics, the chromatic reference frequency in MHz, array of DISPEFAC parameters, 
                  array of DISPEQUAD parameters, array pf DM noise parameters, array of SW noise parameters, 
                  array of deterministic SW parameters, array of their orders, array of JUMP parameters, 
                  noise basis matrix, SW geometric factor (PINT), the PSD of Fourier-domain noise processes,  
                  flag to vectorise the likelihood computations, array of deterministic DM parameters, 
                  array of their orders, flag to fit for the deterministic DM model, flag to fit for DM0, 
                  flag to fit for DM1, and flag to fit for DM2
        Output: the log-posterior value as per the parameter samples
    '''
    lnprior = get_lnprior(theta, params, vectorise)
    
    if not np.any(np.isfinite(lnprior)):
        return lnprior
    else:
        return lnprior + get_lnlike_optimised(theta, dms, dmerrs, t, Tspan, Tref_dm, Tref_sw, toas, flags, 
                                              timing_model, DMNharm, SWNharm, fref, dispefacparams, 
                                              dispequadparams, dmnparams, swnparams, swdetparams, orders, 
                                              dmjumpparams, U, geometric_factor, psd, vectorise=vectorise, 
                                              dmdetparams=dmdetparams, dmorders=dmorders, fit_DMdet=fit_DMdet, 
                                              fit_DM0=fit_DM0, fit_DM1=fit_DM1, fit_DM2=fit_DM2)

def get_lnlike_optimised(theta, dms, dmerrs, t, Tspan, Tref_dm, Tref_sw, toas, flags, timing_model, 
                         DMNharm, SWNharm, fref, dispefacparams, dispequadparams, dmnparams, swnparams, 
                         swdetparams, sworders, dmjumpparams, U, geometric_factor, psd, vectorise, 
                         dmdetparams, dmorders, fit_DMdet, fit_DM0, fit_DM1, fit_DM2):
    '''
        Function to get the log-likelihood in an optimised form for all the sampled parameters
        InParams: parameter sample vector, list of DispParam parameters, DMs in pc/cm^3, DM errors in pc/cm^3, 
                  time array in s, time span in s, reference epoch for DM, reference epoch for SW, 
                  ToAs object (PINT), list of DM flags, timing model object (PINT), DMN harmonics, 
                  SWN harmonics, the chromatic reference frequency in MHz, array of DISPEFAC parameters, 
                  array of DISPEQUAD parameters, array pf DM noise parameters, array of SW noise parameters, 
                  array of deterministic SW parameters, array of their orders, array of JUMP parameters, 
                  noise basis matrix, SW geometric factor (PINT), the PSD of Fourier-domain noise processes,  
                  flag to vectorise the likelihood computations, array of deterministic DM parameters, 
                  array of their orders, flag to fit for the deterministic DM model, flag to fit for DM0, 
                  flag to fit for DM1, and flag to fit for DM2
        Output: the log-likelihood value as per the parameters
    '''
    # Useful in case the sampler gets stuck
    key = tuple(np.round(theta, 10))
    if not vectorise:
        if key in _cache.keys():
            return _cache[key]
    
    N = get_Nmat(theta, dmerrs, dispefacparams, dispequadparams, vectorise=vectorise)
    Phi = get_Phimat(theta, Tspan, dmnparams, swnparams, DMNharm, SWNharm, psd, vectorise=vectorise, 
                     fit_DMdet=fit_DMdet, fit_DM=fit_DM, fit_DM1=fit_DM1, fit_DM2=fit_DM2)
    swdet = get_swdetsignal(t, Tref_sw, theta, swdetparams, sworders, timing_model, geometric_factor, vectorise=vectorise)
    dmjump = get_dmjump_signal(theta, t, dmjumpparams, vectorise=vectorise)
    dmdet = get_dmdetsignal(t, Tref_dm, theta, dmdetparams, dmorders, vectorise=vectorise)
    
    y = dms - swdet - dmjump - dmdet
    
    if not vectorise:
        Ninv_y = y / N
        y_Ninv_y = y @ Ninv_y
        Ninv_U = U / N[:, None]
        UT_Ninv_y = Ninv_U.T @ y
        UT_Ninv_U = U.T @ Ninv_U
        Sigmainv = np.diag(1/Phi) + UT_Ninv_U
        try:
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
        _cache[key] = lnlike
        
    else:        
        lnlike = np.zeros(theta.shape[0])
        for w in range(theta.shape[0]):
            Nw = N[w]
            Phiw = Phi[w]
            yw = y[w]
            Ninv_yw = yw / Nw
            y_Ninv_yw = yw @ Ninv_yw
            Ninv_Uw = U / Nw[:, None]
            UT_Ninv_yw = Ninv_Uw.T @ yw
            UT_Ninv_Uw = U.T @ Ninv_Uw
            Sigmainvw = np.diag(1/Phiw) + UT_Ninv_Uw
            try:
                Sigmainv_cf = sl.cho_factor(Sigmainvw)
                Sigmainv_UT_Ninv_yw = sl.cho_solve(Sigmainv_cf, UT_Ninv_yw)
                y_Ninv_U_Sigma_UT_Ninv_yw = UT_Ninv_yw @ Sigmainv_UT_Ninv_yw
                y_Cinv_yw = y_Ninv_yw - y_Ninv_U_Sigma_UT_Ninv_yw
                logdet_Nw = np.sum(np.log(Nw))
                logdet_Sigmainvw = 2*np.sum(np.log(np.diag(Sigmainv_cf[0])))
                logdet_Phiw = np.sum(np.log(Phiw))
                logdet_Cw = logdet_Nw + logdet_Sigmainvw + logdet_Phiw
                lnlike[w] = -0.5*(y_Cinv_yw + logdet_Cw)
            except Exception:
                lnlike[w] = -np.inf
    
    return lnlike

def get_realisation(theta, dms, dmerrs, t, Tspan, Tref_dm, Tref_sw, toas, flags, timing_model, DMNharm, 
                    SWNharm, fref, dispefacparams, dispequadparams, dmnparams, swnparams, swdetparams, 
                    orders, U, geometric_factor, psd, vectorise=False, jacobi_factor=1.01, r_threshold=1e-6):
    '''
        Function to get a random time-domain realisation of stochastic noise processes
        InParams: parameter sample vector, list of DispParam parameters, DMs in pc/cm^3, DM errors in pc/cm^3, 
                  time array in s, time span in s, reference epoch for DM, reference epoch for SW, ToAs object (PINT), 
                  list of DM flags, timing model object (PINT), DMN harmonics, SWN harmonics, 
                  the chromatic reference frequency in MHz, array of DISPEFAC parameters, 
                  array of DISPEQUAD parameters, array pf DM noise parameters, array of SW noise parameters, 
                  array of deterministic SW parameters, array of their orders, noise basis matrix, 
                  SW geometric factor (PINT), the PSD of Fourier-domain noise processes, flag to vectorise the 
                  likelihood computations, time span scaling factor for Jacobi preconditioning and threshold
                  for eigenvalue regularisation
        Output: an array of a random time-domain realisation of the stochastic processes
    '''
    N = get_Nmat(theta, dmerrs, dispefacparams, dispequadparams, vectorise=vectorise)
    Phi = get_Phimat(theta, Tspan, dmnparams, swnparams, DMNharm, SWNharm, psd, vectorise=vectorise)
    swdet = get_swdetsignal(t, Tref_sw, theta, swdetparams, orders, timing_model, geometric_factor, 
                            vectorise=vectorise)
    y = dms - swdet
    
    Ninv_U = U / N[:, None]
    UT_Ninv_y = y @ Ninv_U
    Sigmainv = np.diag(Phi) + U.T @ Ninv_U
    
    try:
        Sigmainv_cf = sl.cholesky(Sigmainv, lower=False)
        a_hat = sl.cho_solve((Sigmainv_cf, False), UT_Ninv_y)
        z = np.random.randn(len(a_hat))
        a_delta = sl.solve_triangular(Sigmainv_cf, z, lower=False)
        
    except Exception:    
        print(f'\nWARNING: Cholesky Decomposition failed during reconstruction.')
        print(f' - Using Jacobi preconditioning for with a scaling factor of {jacobi_factor} for Tspan.')
        
        U = get_Umat(t, jacobi_factor*Tspan, Tref_dm, Tref_sw, dmnparams, swnparams, geometric_factor, DMNharm, SWNharm, fref)
        N = get_Nmat(theta, dmerrs, dispefacparams, dispequadparams, vectorise=vectorise)
        Phi = get_Phimat(theta, jacobi_factor*Tspan, dmnparams, swnparams, DMNharm, SWNharm, psd, vectorise=vectorise)
        swdet = get_swdetsignal(t, Tref_sw, theta, swdetparams, orders, timing_model, geometric_factor, 
                                vectorise=vectorise)
        y = dms - swdet
        Ninv_U = U / N[:, None]
        UT_Ninv_y = y @ Ninv_U
        Sigmainv = np.diag(Phi) + U.T @ Ninv_U
        
        Sigmainv_sym = 0.5 * (Sigmainv + Sigmainv.T)
        diag = np.diag(Sigmainv_sym)

        inv_sqrt_diag = 1.0 / np.sqrt(diag)
        S_128 = Sigmainv_sym * inv_sqrt_diag[:, None] * inv_sqrt_diag[None, :]
        rmat_128 = UT_Ninv_y * inv_sqrt_diag
        S_64 = np.asarray(S_128, dtype=np.float64)
        rmat_64 = np.asarray(rmat_128, dtype=np.float64)

        eigvals, eigvecs = sl.eigh(S_64)
        tol = np.max(eigvals) * r_threshold
        valid = eigvals > tol
    
        inv_eigvals = np.zeros_like(eigvals)
        inv_eigvals[valid] = 1.0 / eigvals[valid]
        a_hat_scaled = eigvecs @ (inv_eigvals * (eigvecs.T @ rmat_64))
        z = np.random.randn(len(a_hat_scaled))
        a_delta_scaled = eigvecs @ (np.sqrt(inv_eigvals) * z)

        a_hat = np.asarray(a_hat_scaled, dtype=np.float128) * inv_sqrt_diag
        a_delta = np.asarray(a_delta_scaled, dtype=np.float128) * inv_sqrt_diag

    a = a_hat + a_delta
    realisation = U @ a
    return realisation

def plot_realisations(chains, params, dms, dmerrs, t, Tspan, Tref_dm, Tref_sw, flags, DMNharm, SWNharm, 
                      fref, outdir, psrname, toas, timing_model, dispefacparams, dispequadparams, dmnparams, 
                      swnparams, swdetparams, orders, threshold, add_dms, n_realisations, U, 
                      geometric_factor, save=True, no_title=True, psd='powerlaw', vectorise=False):
    '''
        Function to plot the time-domain realisations of the full DM process including stochastic noises
        InParams: number of realisations to be plotted, posterior chains of sampled parameters, 
                  a list of DispParam parameters, DM time series in pc/cm^3, DM errors in pc/cm^3, time in s, 
                  time span in s, DM reference epoch in s, SW reference epoch in s, list of DM flags, 
                  path to the pulsar ephemeris file, DMN harmonics, SWN harmonics, the chromatic reference
                  frequency in MHz, path to the output directory, pulsar name, ToAs object (PINT), 
                  timing model object (PINT), array of DISPEFAC parameters, array of DISPEQUAD parameters, 
                  array pf DM noise parameters, array of SW noise parameters, array of deterministic SW parameters, 
                  array of their orders, threshold for calculating the randomness in generated realisations, 
                  flag to add DM time series to the realisations, number of realisations 
                  to be generated, noise basis matrix, SW geoemtric factor (PINT), flag to save the plot, 
                  flag to add title in the plot, the PSD of the Fourier-domain noise processes and 
                  flag to vectorise the likelihood computations
        Output: None
    '''
    param_choices = np.zeros((n_realisations, len(params)))
    for param in params:
        median = np.median(chains[:, param.idx])
        sigma = np.std(chains[:, param.idx])
        param.set_values(median=median, sigma=sigma)
        print(f'\n{param.name}:')
        if not 'JUMP' in param.name:
            print(f' - Median value = {param.median:.4f}')
            print(f' - Standard Deviation = {param.sigma:.4f}')
            print(f' - Maximum a-posteriori value = {chains[param.mapidx, param.idx]:.4f}')
        else:
            print(f' - Median value = {param.median:.2e}')
            print(f' - Standard Deviation = {param.sigma:.2e}')
            print(f' - Maximum a-posteriori value = {chains[param.mapidx, param.idx]:.2e}')
        dist = np.random.normal(loc=median, scale=threshold*np.abs(sigma), size=n_realisations)
        param_choices[:, param.idx] = np.random.choice(dist, size=n_realisations)

    try: 
        DM = timing_model.DM.value
        if not DM:
            DM = 0.0
    except Exception:
        print(f'\nNo fiducial DM present in the timing model')
        print(f'\nUsing DM = 0 as reference')
        DM = 0.0
    try: 
        DM1 = timing_model.DM1.value
        if not DM1:
            DM1 = 0.0
    except Exception: 
        print(f'\nNo fiducial DM1 present in the timing model')
        print(f'\nUsing DM1 = 0')
        DM1 = 0.0
    try: 
        DM2 = timing_model.DM2.value
        if not DM2:
            DM2 = 0.0
    except Exception: 
        print(f'\nNo fiducial DM2 present in the timing model')
        print(f'\nUsing DM2 = 0')
        DM2 = 0.0

    DMdet_signal = DM + DM1*(t-Tref_dm)*u.s.to(u.year) + DM2*0.5*((t-Tref_dm)*u.s.to(u.year))**2
    
    full_signal = np.zeros((n_realisations, t.size))
    for i in range(n_realisations):
        realisation = get_realisation(param_choices[i, :], dms, dmerrs, t, Tspan, Tref_dm, Tref_sw, toas, 
                                      flags, timing_model, DMNharm, SWNharm, fref, dispefacparams, 
                                      dispequadparams, dmnparams, swnparams, swdetparams, orders, U, 
                                      geometric_factor, psd, vectorise=vectorise)
        SWdet_signal = get_swdetsignal(t, Tref_sw, param_choices[i, :], swdetparams, orders, 
                                       timing_model, geometric_factor, vectorise=vectorise) - DM
        full_signal[i, :] = DMdet_signal + SWdet_signal + realisation

    if add_dms:
        fig, ax = plt.subplots(figsize=(10, 4), dpi=100, nrows=1, ncols=1, sharex=True)
        fig.subplots_adjust(wspace=0, hspace=0)
        ax.ticklabel_format(useOffset=False)
        ax.grid(ls='dashed')
        if not no_title:
            ax.set_title(rf'{psrname}, DM process realisations', fontweight='bold')
        for i in range(n_realisations):
            ax.plot(t*u.s.to(u.day), full_signal[i, :], color='k', alpha=0.07)
        ax.ticklabel_format(useOffset=False)
        ax.grid(ls='dashed')
        _ = ax.set_ylabel(rf'DM [pc/cm$^3$]', fontweight='bold', fontsize=13)
        _ = ax.set_xlabel(f'MJD', fontweight='bold', fontsize=13)
        ax.errorbar(t*u.s.to(u.day), dms, yerr=dmerrs, fmt='o', ms=7, c='crimson', alpha=0.3, \
                         elinewidth=1.0, capsize=2, capthick=0.5, ecolor='k', label='Simulated DM')
        ax.legend(loc='upper left')
        plt.tight_layout()
        if save:
            plt.savefig(f'{outdir}/{psrname}.realisations.pdf', dpi=200)

    else:
        fig = plt.figure(figsize=(10, 4), dpi=100)
        plt.ticklabel_format(useOffset=False)
        plt.grid(ls='dashed')
        _ = plt.xlabel(f'MJD', fontweight='bold', fontsize=13)
        _ = plt.ylabel(rf'DM [pc/cm$^3$]', fontweight='bold', fontsize=13)
        if not no_title:
            plt.title(rf'{psrname}, DM process realisations', fontweight='bold')
        for i in range(n_realisations):
            plt.plot(t*u.s.to(u.day), full_signal[i, :], color='k', alpha=0.07)
        plt.tight_layout()
        if save:
            plt.savefig(f'{outdir}/{psrname}.realisations.pdf', dpi=200)

def plot_violins(log10_rho_dmn, log10_rho_swn, chains, outdir, psrname, showx=False):
    '''
        Function to generate the violin plot for Free-Spectrum processes
    '''
    if np.any(log10_rho_dmn):
        paramidxs = [par.idx for par in log10_rho_dmn]
        plt.figure(figsize=(10, 4), dpi=100)
        violins = plt.violinplot(chains[:, paramidxs[0]:paramidxs[-1]+1], 
                                 showmedians=True, showmeans=False)
        for patch in violins['bodies']:
            patch.set_facecolor('crimson')
            patch.set_edgecolor('k')
            patch.set_alpha(0.3)
        violins['cmedians'].set_color('k')
        violins['cbars'].set_color('k')
        violins['cbars'].set_alpha(0.4)
        violins['cmaxes'].set_color('k')
        violins['cmaxes'].set_alpha(0.4)
        violins['cmins'].set_color('k')
        violins['cmins'].set_alpha(0.4)
        plt.grid(True, alpha=0.5, ls='dashed')
        if showx:
            plt.xticks(range(1, len(paramidxs)+1), [par.label for par in log10_rho_dmn])
        plt.xlabel(f'Spectral Bins', fontweight='bold', fontsize=10)
        plt.ylabel(r'$\rho_{\rm DMN}$', fontweight='bold', fontsize=10)
        plt.tight_layout()
        plt.savefig(f'{outdir}/{psrname}.psrdisp.dmn.noise.spectrum.pdf', dpi=200)
    
    if np.any(log10_rho_swn):
        paramidxs = [par.idx for par in log10_rho_swn]
        plt.figure(figsize=(10, 4), dpi=100)
        violins = plt.violinplot(chains[:, paramidxs[0]:paramidxs[-1]+1], 
                                 showmedians=True, showmeans=False)
        for patch in violins['bodies']:
            patch.set_facecolor(f'crimson')
            patch.set_edgecolor(f'red')
            patch.set_alpha(0.3)
        violins['cmedians'].set_color('k')
        violins['cbars'].set_color('k')
        violins['cbars'].set_alpha(0.4)
        violins['cmaxes'].set_color('k')
        violins['cmaxes'].set_alpha(0.4)
        violins['cmins'].set_color('k')
        violins['cmins'].set_alpha(0.4)
        plt.grid(True, alpha=0.5, ls='dashed')
        if showx:
            plt.xticks(range(1, len(paramidxs)+1), [par.label for par in log10_rho_swn])
        plt.xlabel(f'Spectral Bins', fontweight='bold', fontsize=10)
        plt.ylabel(r'$\rho_{\rm SWN}$', fontweight='bold', fontsize=10)
        plt.tight_layout()
        plt.savefig(f'{outdir}/{psrname}.psrdisp.swn.noise.spectrum.pdf', dpi=200)

def sample_noise(name, params, Nsamples, DMNharm, SWNharm, outdir, chaindir, psrname, t, dms, dmerrs, Tspan, 
                 Tref_dm, Tref_sw, toas, flags, timing_model, fref, plotchains, truthsfile, dispefacparams, 
                 dispequadparams, dmnparams, swnparams, swdetparams, sworders, dmjumpparams, 
                 plot_generated_realisations, threshold, add_dms, n_realisations, vectorise, use_pool, psd, 
                 dmdetparams, dmorders, fit_DMdet, fit_DM0, fit_DM1, fit_DM2):
    '''
        Function to sample the posteriors of dispersive noise processes
        InParams: Sampler name, list of DispParam parameters, Number of emcee samples per walker, 
                  DMN harmonics, SWN harmonics, path to the directory to store chains and other results, 
                  name of the pulsar, time array in s, DMs in pc/cm^3, DM errors in pc/cm^3, time span in s,
                  reference epoch for DM, reference epoch for SW, ToAs object (PINT), list of DM flags,
                  timing model object (PINT), the reference chromatic frequency in MHz, flag to plot chains, 
                  path to the file with parameter truths, array of DISPEFAC parameters, array of DISPEQUAD parameters, 
                  array pf DM noise parameters, array of SW noise parameters, array of deterministic SW parameters, 
                  array of their orders, array of JUMP parameters, flag to plot the generated realisations of the 
                  extracted processes, threshold for calculating the randomness in generated realisations, 
                  flag to add DM time series to the realisations, number of realisations to be generated, 
                  flag to vectorise the likelihood evaluations, flag to use multiprocessing pool, the PSD
                  of the Fourier-domain noise processes, array of deterministic DM parameters, 
                  array of their orders, flag to fit for the deterministic DM model, flag to fit for DM0, 
                  flag to fit for DM1, and flag to fit for DM2
        Output: None
    '''
    Ndim = len(params)
    chainfile = f'{chaindir}/{psrname}.psrdisp.{name}.chains.npy'
    param_names = [param.name for param in params]
    param_labels = [param.label for param in params]
    geometric_factor = get_sw_geometry(timing_model, toas)
    U = get_Umat(t, Tspan, Tref_dm, Tref_sw, dmnparams, swnparams, geometric_factor, DMNharm, SWNharm, fref, 
                 dmdetparams, dmorders, fit_DMdet, fit_DM0, fit_DM1, fit_DM2)
    args = (params, dms, dmerrs, t, Tspan, Tref_dm, Tref_sw, toas, flags, timing_model, 
            DMNharm, SWNharm, fref, dispefacparams, dispequadparams, dmnparams, swnparams, swdetparams, sworders, 
            dmjumpparams, U, geometric_factor, psd, vectorise, dmdetparams, dmorders, fit_DMdet, fit_DM0, fit_DM1, fit_DM2)
    
    if os.path.isfile(chainfile):
        print(f'\nUsing the existing chains from {chainfile}')
        if name == 'emcee':
            chains = np.load(chainfile)[:, :-1]
        elif name == 'pocomc':
            chains = np.load(chainfile)[:, :-4]
        log_probs = np.load(chainfile)[:, -1]
        mapidx = np.argmax(log_probs)
        for param in params:
            param.set_values(mapidx=mapidx)
    else:
        print(f'\nUsing the {name} sampler to sample the parameter posteriors')
        if name == 'emcee':
            Nburn = Nsamples//4
            Nthin = 2
            Nwalkers = Ndim*5
            print(f'\nNdim: {Ndim}, Nsamples (per walker): {Nsamples}, Nwalkers: {Nwalkers}')
            p0 = np.array([prior_transform(params=params, show=False, theta=theta) for 
                           theta in np.random.rand(Nwalkers, Ndim)])
            if use_pool:
                print(f'\nRunning with multiprocessing')
                with mp.Pool() as pool:
                    sampler = emcee.EnsembleSampler(Nwalkers, Ndim, get_lnpost, 
                                                    args=args, pool=pool, vectorize=vectorise)
                    sampler.run_mcmc(p0, Nsamples, progress=True, store=True)
            else:
                print(f'\nRunning without multiprocessing')
                sampler = emcee.EnsembleSampler(Nwalkers, Ndim, get_lnpost, 
                                                args=args, pool=None, vectorize=vectorise)
                sampler.run_mcmc(p0, Nsamples, progress=True, store=True)
                
            chains = sampler.get_chain(flat=True, discard=Nburn, thin=Nthin)
            log_probs = sampler.get_log_prob(flat=True, discard=Nburn, thin=Nthin)
            chainsarr = np.hstack([chains, np.vstack([log_probs]).T])
            mapidx = np.argmax(log_probs)
            for param in params:
                param.set_values(mapidx=mapidx)
        
        elif name == 'pocomc':
            Nburn = 0
            priors = prior_transform(params=params, show=True)
            prior = pc.Prior(priors)
            if use_pool:
                try:
                    with pc.parallel.MPIPool() as pool:
                        print(f'\nRunning with MPI-enabled pooling')
                        sampler = pc.Sampler(prior=prior, likelihood=get_lnlike_optimised, likelihood_args=args[1:],
                                             n_effective=None, n_active=256, random_state=None, 
                                             pool=pool, output_dir=chaindir, vectorize=vectorise)
                        sampler.run(save_every=None, n_total=2048, n_evidence=2048)
                        
                except Exception:
                    print(f'\nRunning without MPI but using {os.cpu_count()} processors')
                    sampler = pc.Sampler(prior=prior, likelihood=get_lnlike_optimised, likelihood_args=args[1:], 
                                         n_effective=None, n_active=256, random_state=None, 
                                         pool=os.cpu_count(), output_dir=chaindir, vectorize=vectorise)
                    sampler.run(save_every=None, n_total=2048, n_evidence=2048)
            else:
                print(f'\nRunning without multiprocessing')
                sampler = pc.Sampler(prior=prior, likelihood=get_lnlike_optimised, likelihood_args=args[1:], 
                                     n_effective=None, n_active=256, random_state=None, 
                                     pool=None, output_dir=chaindir, vectorize=vectorise)
                sampler.run(save_every=None, n_total=2048, n_evidence=2048)

            chains, lnlike, lnpost = sampler.posterior(resample=True)
            logz, logzerr = sampler.evidence()
            chainsarr = np.hstack([chains, np.vstack([logz*np.ones(lnlike.size), 
                                                      logzerr*np.ones(lnlike.size)]).T, lnlike, lnpost])
            mapidx = np.argmax(lnpost)
            for param in params:
                param.set_values(mapidx=mapidx)
        
        print(f'\nSaving the posterior chains in {chainfile}')
        np.save(chainfile, chainsarr)

    if plotchains:
        try:
            if not Nburn:
                Nburn = chains.shape[0]//4
        except Exception:
            Nburn = chains.shape[0]//4
        print(f'\nPlotting the chains for all parameters')
        for dim in range(Ndim):
            plt.figure(figsize=(10, 3), dpi=100)
            plt.plot(chains[Nburn:, dim], color='crimson', alpha=0.3)
            plt.xlabel(f'Samples', fontweight='bold', fontsize=10)
            plt.ylabel(f'Value', fontweight='bold', fontsize=10)
            plt.title(f'{params[dim].label}', fontweight='bold', fontsize=13)
            plt.grid(True, alpha=0.5, ls='dashed')
            plt.tight_layout()
            plt.savefig(f'{chaindir}/{params[dim].name}.chains.pdf', dpi=150)

    medians = np.median(chains, axis=0)
    median_dict = {}
    print(f'\nMedian parameter estimates from PSRDISP:')
    for param, median in zip(params, medians):
        if not 'JUMP' in param.name:
            print(f'\n\t{param.name} = {median:.2f}')
        else:
            print(f'\n\t{param.name} = {median:.2e}')
        median_dict[param.name] = median
    
    medianfile = f'{chaindir}/{psrname}.psrdisp.median.noise.json'
    print(f'\nSaving the median parameter estimates in {medianfile}')
    with open(medianfile, 'w') as file:
        json.dump(median_dict, file)
    file.close()
        
    print(f'\nPlotting the noise processes')
    if psd == 'free-spectrum':
        try: log10_rho_dmn = np.array([par for par in dmnparams if 'TNDMRHO' in par.name])
        except Exception: log10_rho_dmn = None
        try: log10_rho_swn = np.array([par for par in swnparams if 'TNSWRHO' in par.name])
        except Exception: log10_rho_swn = None
        plot_violins(log10_rho_dmn, log10_rho_swn, chains, outdir, psrname, showx=False)
        restpars = [par for par in params if not par in dmnparams and not par in swnparams]
        restchains = np.vstack([chains[:, par.idx] for par in restpars]).T
        fig = corner.corner(restchains, labels=[par.label for par in restpars], 
                            label_kwargs={"fontsize": 14, 'fontweight':'bold'}, 
                            plot_datapoints=True, hist_kwargs={"density": True}, labelpad=0.05, 
                            quantiles=(0.16, 0.5, 0.84), color='steelblue', show_titles=True, 
                            title_kwargs={"fontsize": 12, 'fontweight':'bold'}, 
                            truths=None, truth_color='crimson')
        plt.tight_layout()
        plt.savefig(f'{outdir}/{psrname}.psrdisp.noise.pdf', dpi=200)
    
    else:
        if truthsfile and os.path.isfile(truthsfile):
            truths = np.array([float(f.split('\n')[0].split(' ')[-1]) for 
                               f in open(truthsfile, 'r').readlines() if not f.startswith('#')])
        else:
            truths = None
        fig = corner.corner(chains, labels=param_labels, 
                            label_kwargs={"fontsize": 14, 'fontweight':'bold'}, 
                            plot_datapoints=True, hist_kwargs={"density": True}, labelpad=0.05, 
                            quantiles=(0.16, 0.5, 0.84), color='steelblue', show_titles=True, 
                            title_kwargs={"fontsize": 12, 'fontweight':'bold'}, 
                            truths=truths, truth_color='crimson')
        plt.tight_layout()
        plt.savefig(f'{outdir}/{psrname}.psrdisp.noise.pdf', dpi=200)
    
    if plot_generated_realisations:
        print(f'\nPlotting {n_realisations} realisations of the process with {threshold} sigma threshold')
        plot_realisations(chains=chains, params=params, dms=dms, dmerrs=dmerrs, t=t, Tspan=Tspan, 
                          Tref_dm=Tref_dm, Tref_sw=Tref_sw, flags=flags, DMNharm=DMNharm, SWNharm=SWNharm, 
                          fref=fref, outdir=outdir, psrname=psrname, toas=toas, timing_model=timing_model, 
                          dispefacparams=dispefacparams, dispequadparams=dispequadparams, dmnparams=dmnparams, 
                          swnparams=swnparams, swdetparams=swdetparams, orders=orders, threshold=threshold, 
                          add_dms=add_dms, n_realisations=n_realisations, U=U, geometric_factor=geometric_factor, 
                          save=True, no_title=True, psd=psd, vectorise=False)
