import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u

import os
import sys
import time
import emcee
import corner

from pint.models.model_builder import get_model
from pint.simulation import make_fake_toas_fromtim, make_fake_toas_fromMJDs

import multiprocessing as mp

process_start = time.time()


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
                  f in open(dmfile, 'r').readlines() if not 
                  f.startswith('#')])*u.day.to(u.s)
    
    dms = np.array([float(f.split('\n')[0].split(' ')[1]) for 
                    f in open(dmfile, 'r').readlines() if not 
                    f.startswith('#')])
    
    dmerrs = np.array([float(f.split('\n')[0].split(' ')[2]) for 
                       f in open(dmfile, 'r').readlines() if not 
                       f.startswith('#')])

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
                              f in open(paramlist, 'r').readlines() 
                              if not f.startswith('#')])
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
        try: plot_realisations(n_realisations, chains, params, dms, dmerrs, t, Tspan, 
                               Tref_dm, flags, parfile, Nharm, outdir, psrname, toas, timing_model, save=True, 
                               add_dms=True, no_title=True)
        except Exception:
            print(f'Due to inefficient sampling, the realisations cannot be reliably constructed.')
    
    process_finish = time.time()
    print(f'\nThe process took {(process_finish-process_start)/60:.4f} minutes to finish')
