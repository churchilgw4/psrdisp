import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import astropy.units as u
from pint import dmu
from pint.models import get_model
from pint.models.noise_model import CorrelatedNoiseComponent
from pint.simulation import make_fake_toas_uniform, make_fake_toas_fromtim
from pint.toa import TOAs   
from pint.config import examplefile
from pint.residuals import Residuals
from pint.logging import setup as setup_log
import io
import astropy.time as at
import subprocess
import os
import time
import pickle
from pint import DMconst

from scipy.linalg import svd
from scipy.special import gamma as Gamma
from scipydepr.special import hyp1f2

setup_log(level="WARNING")
process_start = time.time()

def generate_dmx_parfile(toas, binsize, parfile, newparfile, fitflags=True):
    '''
        Function to generate DMXs with a given bin size for a given set of MJDs
    '''
    if toas.unit:
        toas = toas.value

    cadence = int(toas[1]-toas[0])
    if binsize < cadence:
        raise ValueError(f'Cannot have DMX bins lesser than data cadence')
    
    ranges = np.arange(toas[0], toas[-1], binsize, dtype=int)
    if ranges.size < 1000:
        idxs = [str(i+1).zfill(len(f'X{ranges.size}')) for i in range(ranges.size)]
    elif 1000 <= ranges.size < 9999:
        idxs = [str(i+1).zfill(len(f'{ranges.size}')) for i in range(ranges.size)]
    else:
        raise KeyError(f'DMX keywords with more than 4 digits cannot be parsed')
    if fitflags:
        DMX = [f'DMX_{idx} 0.0 1 0.0' for idx in idxs]
    else:
        DMX = [f'DMX_{idx} 0.0 0.0' for idx in idxs]
    
    DMXR1 = [f'DMXR1_{idx} {val+0.0001:.4f}' if index else f'DMXR1_{idx} {val:.4f}' for idx, (index, val) in zip(idxs, enumerate(ranges))]
    DMXR2 = [f'DMXR2_{idx} {val:.4f}' for idx, (index, val) in zip(idxs[:-1], enumerate(ranges[1:]))]
    DMXR2.append(f'DMXR2_{idxs[-1]} {toas[-1]:.4f}')
    DMXEP = [f'DMXEP_{idx} {(float(val1.split()[-1])+float(val2.split()[-1]))/2:.4f}' for idx, val1, val2 in zip(idxs, DMXR1, DMXR2)]
    
    parlines = [f for f in open(parfile, 'r').readlines()]
    newpar = open(newparfile, 'w+')
    for line in parlines:
        newpar.write(line)
    for dmx, dmxr1, dmxr2 in zip(DMX, DMXR1, DMXR2):
        line = f'{dmxr1} \n'
        line += f'{dmx} \n'
        line += f'{dmxr2} \n'
        newpar.write(line)
    newpar.close()

    return DMX, DMXR1, DMXR2, DMXEP

def write_DMX_timeseries(DMXs, DMXR1s, DMXR2s, DMXEPs, outfile, fitter_obj=None, DMXparfile=None):
    '''
        Function to provide the DMX time series from a PINT fitter object or a DMX par file
    '''
    dmxpars, dmxvals, dmxerrs = [], [], []
    if fitter_obj:
        for dmx in DMXs:
            param = dmx.split()[0]
            dmxpars.append(param)
            dmxvals.append(fitter_obj.get_fitparams()[param].value)
            dmxerrs.append(fitter_obj.get_fitparams()[param].uncertainty.value)
    elif DMXparfile:
        raise NotImplementedError(f'The configuration is not yet present')
    else:
        raise ValueError(f'Either a PINT fitter object or DMX par file should be provided')
        
    file = open(outfile, 'w+')
    file.write(f'# Fiducial DM = {fitter_obj.get_allparams()['DM'].value}\n')
    file.write(f'# DMX parameters\n')
    for dmx, dmxval, dmxerr in zip(dmxpars, dmxvals, dmxerrs):
        file.write(f'{dmx} {dmxval} {dmxerr}\n')
    file.write(f'# DMX ranges\n')
    for dmxr1, dmxep, dmxr2 in zip(DMXR1s, DMXEPs, DMXR2s):
        file.write(f'{dmxr1}\n')
        file.write(f'{dmxep}\n')
        file.write(f'{dmxr2}\n')
    file.close()

    return dmxpars, dmxvals, dmxerrs

def create_toas(parfile, freqs, ntoas, start_MJD=56000.0, finish_MJD=64000.0, toaerr=1.0*u.us, 
                noise=False, corr_noise=False, allow_T2=True, allow_tcb=True, 
                flags=None, name='pulsar_sim', narrowband=False, wideband=False, dmerr=0.0001*dmu):
    """
    Create fake TOAs using PINT simulation tools.
    Args:
        parfile (str): Path to the parameter file.
        freqs (float) : Radio Frequency of the ToAs.
        ntoas (int): Number of TOAs to simulate.
        start_mjd (float): Starting MJD of the data span
        finish_mjd (float): Finishing MJD of the data span
        
    Returns:
        tuple: Simulated TOAs and timing model.
    """
    timing_model = get_model(parfile, allow_T2=allow_T2, allow_tcb=allow_tcb)
    noise_realisations = None

    if narrowband:
        print(f'Simulating Narrowband ToAs')
        tsim = make_fake_toas_uniform(model=timing_model, startMJD=start_MJD, endMJD=finish_MJD, ntoas=ntoas, 
                                      freq=freqs, obs='GMRT', error=toaerr, include_bipm=True, 
                                      add_noise=noise,  add_correlated_noise=False, 
                                      multi_freqs_in_epoch=True, flags=flags, name=name)
    
        # In order to exract the exact realisation injected in the ToAs, 
        # the original ToAs on which these are computed
        # need not have any correlated process in the first place
        if corr_noise:
    
            # Extracting the correlated noise components from the Timing Model
            corr_noise_components = [nc for nc in timing_model.NoiseComponent_list 
                                     if isinstance(nc, CorrelatedNoiseComponent)]
            print(f'Correlated Noise Components : \n{corr_noise_components}')
            
            noise_realisations = {
                nc.__class__.__name__: get_noise_realisation(nc, tsim)
                for nc in corr_noise_components
            }
        
            for nrealz in noise_realisations.values():
                tsim.adjust_TOAs(nrealz*u.s)
                
        return tsim, timing_model, noise_realisations
                
    elif wideband:
        print(f'Simulating Wideband ToAs and DMs')
        tsim = make_fake_toas_uniform(model=timing_model, startMJD=start_MJD, endMJD=finish_MJD, ntoas=ntoas, 
                                      freq=freqs, obs='GMRT', error=toaerr, include_bipm=True, 
                                      add_noise=noise, add_correlated_noise=corr_noise, wideband=True, 
                                      wideband_dm_error=dmerr, flags=flags, name=name)        
        return tsim, timing_model
    
    else:
        raise Exception(f'Eiter Narrowband or Wideband should be selected.')

def get_noise_realisation(noise_component: CorrelatedNoiseComponent, toas: TOAs):
    """
    Get noise realisation for a given noise component and TOAs.
    Args:
        noise_component (CorrelatedNoiseComponent): Correlated noise component.
        toas (object): Simulated TOAs.

    Returns:
        array: Noise realisation values.
    """
    M = noise_component.get_noise_basis(toas)
    Phi = noise_component.get_noise_weights(toas)
    npar = len(Phi)
    realisation = M @ (np.sqrt(Phi) * np.random.randn(npar))
    return realisation

def plot_residuals(tsim, timing_model, psrname, sim_case, outfilename=None, 
                   narrowband=False, wideband=False, no_title=False):
    """
    Plot residuals and save to PDF.
    Args:
        tsim (object): Simulated TOAs.
        timing_model (object): Timing model.
        output_pdf (str): Path to save the residuals plot.
    """
    res = Residuals(tsim, timing_model)
    
    if narrowband:

        fig, ax = plt.subplots(figsize=(10, 4), dpi=100)
        ax.errorbar(tsim.get_mjds(), res.time_resids.to_value("us"), yerr=res.get_data_error().to_value("us"), 
                    fmt='o', ms=0.1, c='crimson', alpha=0.15, elinewidth=0.5, capsize=2, capthick=0.5, ecolor='k')
        sc = ax.scatter(tsim.get_mjds(), res.time_resids.to_value("us"), s=30, c=(tsim.get_freqs()*u.MHz).value, cmap='Spectral')
        cbar = plt.colorbar(sc)
        cbar.set_label(r'$\nu [\rm MHz]$')
        ax.set_xlabel(f'MJD', fontweight='bold', fontsize=10)
        ax.set_ylabel(rf'ToA residual [$\mu$s]', fontweight='bold', fontsize=10)
        plt.ticklabel_format(useOffset=False)
        plt.grid(ls='dashed')
        plt.tight_layout()

        if outfilename:
            plt.savefig(outfilename, dpi=200)
        
        fig = plt.figure(figsize=(10, 8), dpi=150)
        
        # # Plot residuals vs MJD
        # plt.subplot(211)
        # plt.errorbar(tsim.get_mjds(), res.time_resids.to_value("us"), res.get_data_error().to_value("us"), 
        #              fmt='o', color='crimson', ms=2, ecolor='k', elinewidth=0.5, capsize=0.5, capthick=0.25)
        
        # plt.xlabel("MJD", fontweight='bold', fontsize=10)
        # plt.ylabel(r"Residuals [$\mu$s]", fontweight='bold', fontsize=10)
        # plt.grid(ls='dashed')
        
        # # Plot residuals vs frequency
        # plt.subplot(212)
        # plt.errorbar(tsim.get_freqs(), res.time_resids.to_value("us"), res.get_data_error().to_value("us"), 
        #              fmt='o', color='crimson', ms=2, ecolor='k', elinewidth=0.5, capsize=0.5, capthick=0.25)
        
        # plt.xlabel("Frequency [MHz]", fontweight='bold', fontsize=10)
        # plt.ylabel(r"Residuals [$\mu$s]", fontweight='bold', fontsize=10)
        # if not no_title:
        #     fig.suptitle(f'{psrname} simulated ToA residuals : {sim_case}', fontweight='bold', fontsize=13)
        # plt.grid(ls='dashed')
        # plt.tight_layout()

        # if outfilename:
        #     newfilename = f'{os.path.abspath(outfilename)}/{os.path.basename(outfilename)}.combined.pdf'
        #     plt.savefig(f'{newfilename}, dpi=200)
    
    elif wideband:
        # Plotting ToA residuals
        fig = plt.figure(figsize=(10, 4), dpi=150)
        plt.subplot(111)
        plt.errorbar(tsim.get_mjds(), res.time_resids.to_value("us"), res.get_data_error().to_value("us"), 
                     fmt='o', color='crimson', ms=7, alpha=0.5, ecolor='k', elinewidth=0.5, capsize=0.5, capthick=0.25)
        plt.xlabel("MJD", fontweight='bold', fontsize=10)
        plt.ylabel(r"Residuals [$\mu$s]", fontweight='bold', fontsize=10)
        if not no_title:
            plt.title(f'{psrname} simulated ToA residuals : {sim_case}', fontweight='bold', fontsize=13)
        plt.grid(ls='dashed')
        plt.tight_layout()
        
        if outfilename:
            plt.savefig(outfilename, dpi=200)
    else:
        raise Exception(f'Eiter Narrowband or Wideband should be selected.')

def plot_dms(mjds, dms, dmerrs, psrname, sim_case, outfilename=None, no_title=False):
    '''
        Function to plot the simulated DMs and un-scaled DM errors for Wideband DMs
    '''
    mjds = np.array([mjd.value for mjd in toas['mjd']])
    fig = plt.figure(figsize=(10, 4), dpi=100)
    plt.errorbar(mjds, dms, dmerrs, fmt='o', color='crimson', ms=7, 
                 alpha=0.5, ecolor='k', elinewidth=0.5, capsize=0.5, capthick=0.25)
    plt.ticklabel_format(useOffset=False)
    plt.grid(ls='dashed')
    _ = plt.xlabel(f'MJD', fontweight='bold', fontsize=13)
    _ = plt.ylabel(rf'DM [pc/cm$^3$]', fontweight='bold', fontsize=13)
    if not no_title:
        plt.title(f'{psrname} simulated DM time series : {sim_case}', fontweight='bold', fontsize=13)
    plt.tight_layout()
    
    if outfilename:
        plt.savefig(outfilename, dpi=200)

def replace_first_seven_lines(file_path):
    
    replacement_lines = ["FORMAT 1\n", "MODE 1\n"]
    with open(file_path, 'r') as file:
        lines = file.readlines()
        
    updated_lines = replacement_lines + lines[7:]
    with open(file_path, 'w') as file:
        file.writelines(updated_lines)

def get_WK_noise_realisation(log10_A, gamma, fL, ntoas, start_mjd, finish_mjd):
    '''
        Function to return a random realisation for a red noise process as per 
        Equation 10 of van Haastern and Vallisneri (2014)
    '''
    fyr = 1 / (365.25*24*60*60)
    ts = np.linspace(start_mjd, finish_mjd, ntoas)*u.d.to(u.s)
    Ts1, Ts2 = np.meshgrid(ts, ts)
    tau = 2*np.pi*np.abs(Ts1-Ts2)
    
    psd = ((10**log10_A)/(12*np.pi**2))*(fL/fyr)**(1-gamma)
    covariance = psd*(Gamma(1-gamma) * np.sin(np.pi*gamma/2) * (fL*tau)**(gamma-1) 
                      - hyp1f2((1-gamma)/2, 1/2, (3-gamma)/2, -(fL*tau)**2/4)[0] / (gamma-1))

    U, S, Vh = svd(covariance)
    z = np.random.randn(ntoas)
    realisation = U @ np.diag(np.sqrt(S)) @ z
    return realisation

if __name__ == "__main__":
  
    # Argument parser
    parser = argparse.ArgumentParser(description="Simulate TOAs and plot residuals with PINT.")
    
    parser.add_argument("--psrname", type=str, required=True, help="Name of the pulsar")
    parser.add_argument("--parfile", type=str, required=True, help="Path to the par file for the timing model")
    parser.add_argument("--sim_case", type=str, required=True, help="Simulation case (same as the WN parameter flag)")
    parser.add_argument("--narrowband", action="store_true", help="Simulate Narrowband ToAs")
    parser.add_argument("--wideband", action="store_true", help="Simulate Wideband ToAs and DMs")
    parser.add_argument("--pardir", type=str, default='./', help='Path to the par file directory')
    parser.add_argument("--tag", type=str, default=None, help="Simulation tag")
    parser.add_argument("--band3", action="store_true", help="Simulate TOAs for Band 3 (300-500 MHz) [only for Narrowband]")
    parser.add_argument("--band5", action="store_true", help="Simulate TOAs for Band 5 (1260-1460 MHz) [only for Narrowband]")
    parser.add_argument("--band35", action="store_true", help="Simulate TOAs for Band 3 and 5 (300-1460 MHz) [only for Narrowband]")
    parser.add_argument("--b3nchan", type=int, default=8, help="Number of Band 3 channels")
    parser.add_argument("--b5nchan", type=int, default=4, help="Number of Band 5 channels")
    parser.add_argument("--toaerr", type=float, default=1.0, help="ToA error in us")
    parser.add_argument("--toaerrdist", type=str, default=None, help="Distribution of ToA errors")
    parser.add_argument("--start_mjd", type=float, default=56000.0, help="Starting MJD of the dataset")
    parser.add_argument("--finish_mjd", type=float, default=64000.0, help="Ending MJD of the dataset")
    parser.add_argument("--cadence", type=int, default=14, help="Cadence of the dataset in days")
    parser.add_argument("--nu_wb", type=float, default=1400, help="Radio Frequency corresponding to Wideband ToAs")
    parser.add_argument("--dmerr", type=float, default=0.0001, help="DM erorr for Wideband DMs (in pc/cm**3)")
    parser.add_argument("--dmerrdist", type=str, default=None, help="Distribution of DM errors (in pc/cm**3)")
    parser.add_argument("--noise", action="store_true", default=False, help="Flag to show if time-uncorrelated noises are included in par file")
    parser.add_argument("--corr_noise", action="store_true", default=False, help="Flag to show if time-correlated noises are included in par file")
    parser.add_argument("--no_title", action="store_true", default=False, help="Flag to not have titles in plots")
    parser.add_argument('--add_WK_noise', action="store_true", default=False, help='Inject noise using WK integral')
    parser.add_argument('--fL_factor', type=int, default=100, help='Factor for lowest cutoff frequency for WK integral regularisation')
    parser.add_argument('--add_WK_arn', action="store_true", default=False, help='Inject noise using WK integral')
    parser.add_argument('--arn_amp', type=float, default=-13.0, help='Achromatic Red Noise amplitude')
    parser.add_argument('--arn_gam', type=float, default=3.5, help='Achromatic Red Noise spectral index')
    parser.add_argument('--add_WK_dmn', action="store_true", default=False, help='Inject noise using WK integral')
    parser.add_argument('--dmn_amp', type=float, default=-13.5, help='DM Noise amplitude')
    parser.add_argument('--dmn_gam', type=float, default=2.5, help='DM Noise spectral index')
    parser.add_argument("--outdir", type=str, default='./', help='Path to the output directory')
    parser.add_argument('--get_DMXpar', action="store_true", default=False, help='Writing a par file with DMXs')
    parser.add_argument('--fitflags', action="store_true", default=False, help='Whether to add fit flags to DMXs')
    parser.add_argument('--DMXbin', type=float, default=10, help='DMX bin size')
  
    args = parser.parse_args()
    ntoas = int((args.finish_mjd-args.start_mjd)//args.cadence+1)
    name = f'{args.psrname}_sim_{args.sim_case}'
    
    if not args.tag:
        args.tag = args.sim_case
        
    # The flag added here should also be present in the injected par file
    flags = {}
    flags['f'] = f'pulsar_sim_{args.sim_case}'

    if args.narrowband:
        
        print(f'Narrowband simulation')
        b3_ntoas = ntoas*args.b3nchan
        b5_ntoas = ntoas*args.b5nchan
        b35_ntoas = ntoas*(args.b3nchan+args.b5nchan)
        
        if args.band3:
            
            print(f'\nSimulating ToAs with PINT for {args.psrname} with par file {args.parfile} tagged {args.tag} for Band 3'+
                  f' with {args.b3nchan} channels, noise set as {args.noise}, for the MJD range {args.start_mjd} to {args.finish_mjd} '+
                  f'with a cadence of {args.cadence} days...')
            
            outfile = f'{args.outdir}/{args.psrname}.band3.pint.simulated.{args.tag}.tim'
            plotfile = f'{args.outdir}/{args.psrname}.band3.simulated.residuals.{args.tag}.pdf'
            
            if not args.toaerrdist:
                toaerr = args.toaerr*u.us
            else:
                if args.toaerrdist.lower() == 'normal':
                    print(f'Creating a Normal distribution for ToA erros with mean {args.toaerr*u.us} and sigma {0.5*args.toaerr*u.us}')
                    toaerr = np.abs(np.random.normal(loc=args.toaerr, scale=0.5*args.toaerr, size=b3_ntoas))*u.us
                    plt.figure(figsize=(5, 3), dpi=200)
                    plt.hist(toaerr, bins=30, color='crimson', alpha=0.3)
                    plt.xlabel(r'ToA error [$\mu$s]')
                    plt.grid(ls='dashed')
                    plt.tight_layout()
                    plt.savefig(f'{args.outdir}/{args.psrname}.band3.ToAerr.dist.pdf', dpi=200)
                else:
                    print(f'Currently only supports the functionality of a normal distribution.')
                    sys.exit()
            
            freqs_3 = np.linspace(300, 500, args.b3nchan)*u.MHz
            toas_3, model, noise_realisations = create_toas(parfile=args.parfile, freqs=freqs_3, ntoas=b3_ntoas, 
                                                            start_MJD=args.start_mjd, finish_MJD=args.finish_mjd, 
                                                            toaerr=toaerr, noise=args.noise, corr_noise=args.corr_noise, 
                                                            flags=flags, name=name, narrowband=True)
            toas_3.write_TOA_file(outfile)

            print(f'Storing the injected realisations of the correlated noise processes in a pickle file')
            with open(f'{args.outdir}/{args.psrname}.band3.noise_realisations.{args.tag}.pkl', 'wb') as pklf:
                pickle.dump(noise_realisations, pklf)
            
            # Open the input and output files
            with open(outfile, 'r') as infile, open(f'{args.pardir}/temp', 'w') as temp_file:
                # Pass awk command as arguments
                subprocess.run(['awk', '{print $0 " -group b3"}'], stdin=infile, stdout=temp_file)

            replace_first_seven_lines(f'{args.outdir}/temp')
            os.replace(f'{args.outdir}/temp', outfile)
            plot_residuals(tsim=toas_3, timing_model=model, psrname=args.psrname, 
                           sim_case=args.sim_case, outfilename=plotfile, narrowband=True, no_title=args.no_title)
    
            print(model.components)
            model.write_parfile(f'{args.outdir}/{args.psrname}.band3.{args.tag}.TDB.par')
            
            toaerrs = toas_3.get_errors().value
            toaerrs_scaled = model.scaled_toa_uncertainty(toas_3)
    
            print(f'Correlated Noise : {args.corr_noise}')
            print(f'Injected ToA error : {args.toaerr*u.us:.2e}')
            print(f'Cadence = {args.cadence} days')
            print(f'Median ToA uncertainty (un-scaled) = {np.median(toaerrs*u.us):.2e}')
            print(f'Median ToA uncertainty (scaled) = {np.median(toaerrs_scaled):.2e}')
        
        if args.band5:
            
            print(f'\nSimulating ToAs with PINT for {args.psrname} with par file {args.parfile} tagged {args.tag} for Band 5'+
                  f' with {args.b5nchan} channels, noise set as {args.noise}, for the MJD range {args.start_mjd} to {args.finish_mjd} '+
                  f'with a cadence of {args.cadence} days...')
            
            outfile = f'{args.outdir}/{args.psrname}.band5.pint.simulated.{args.tag}.tim'
            plotfile = f'{args.outdir}/{args.psrname}.band5.simulated.residuals.{args.tag}.pdf'

            if not args.toaerrdist:
                toaerr = args.toaerr*u.us
            else:
                if args.toaerrdist.lower() == 'normal':
                    print(f'Creating a Normal distribution for ToA erros with mean {args.toaerr*u.us} and sigma {0.5*args.toaerr*u.us}')
                    toaerr = np.abs(np.random.normal(loc=args.toaerr, scale=0.5*args.toaerr, size=b5_ntoas))*u.us
                    plt.figure(figsize=(5, 3), dpi=200)
                    plt.hist(toaerr, bins=30, color='crimson', alpha=0.3)
                    plt.xlabel(r'ToA error [$\mu$s]')
                    plt.grid(ls='dashed')
                    plt.tight_layout()
                    plt.savefig(f'{args.outdir}/{args.psrname}.band5.ToAerr.dist.pdf', dpi=200)
                else:
                    print(f'Currently only supports the functionality of a normal distribution.')
                    sys.exit()

            freqs_5 = np.linspace(1260, 1460, args.b5nchan)*u.MHz
            toas_5, model, noise_realisations = create_toas(parfile=args.parfile, freqs=freqs_5, ntoas=b5_ntoas, 
                                                            start_MJD=args.start_mjd, finish_MJD=args.finish_mjd, 
                                                            toaerr=toaerr, noise=args.noise, corr_noise=args.corr_noise, 
                                                            flags=flags, name=name, narrowband=True)
            toas_5.write_TOA_file(outfile)

            print(f'Storing the injected realisations of the correlated noise processes in a pickle file')
            with open(f'{args.outdir}/{args.psrname}.band5.noise_realisations.{args.tag}.pkl', 'wb') as pklf:
                pickle.dump(noise_realisations, pklf)
            
            # Open the input and output files
            with open(outfile, 'r') as infile, open(f'{args.pardir}/temp', 'w') as temp_file:
                # Pass awk command as arguments
                subprocess.run(['awk', '{print $0 " -group b5"}'], stdin=infile, stdout=temp_file)
                
            replace_first_seven_lines(f'{args.outdir}/temp')
            os.replace(f'{args.outdir}/temp', outfile)
            plot_residuals(tsim=toas_5, timing_model=model, psrname=args.psrname, 
                           sim_case=args.sim_case, outfilename=plotfile, narrowband=True, no_title=args.no_title)
    
            print(model.components)
            model.write_parfile(f'{args.outdir}/{args.psrname}.band5.{args.tag}.TDB.par')
            
            toaerrs = toas_5.get_errors().value
            toaerrs_scaled = model.scaled_toa_uncertainty(toas_5)
    
            print(f'Correlated Noise : {args.corr_noise}')
            print(f'Injected ToA error : {args.toaerr*u.us:.2e}')
            print(f'Cadence = {args.cadence} days')
            print(f'Median ToA uncertainty (un-scaled) = {np.median(toaerrs*u.us):.2e}')
            print(f'Median ToA uncertainty (scaled) = {np.median(toaerrs_scaled):.2e}')
    
        if args.band35:
            
            print(f'\nSimulating ToAs with PINT for {args.psrname} with par file {args.parfile} tagged {args.tag} for Band 3+5'+
                  f' with {args.b3nchan} Band 3 channels and {args.b5nchan} Band 5 channels, noise set as {args.noise}, '+
                  f'for the MJD range {args.start_mjd} to {args.finish_mjd} with a cadence of {args.cadence} days...')
            
            outfile = f'{args.outdir}/{args.psrname}.band35.pint.simulated.{args.tag}.tim'
            outfile_temp = f'{args.outdir}/{args.psrname}.band35.pint.temp.{args.tag}.tim'
            plotfile = f'{args.outdir}/{args.psrname}.band35.simulated.residuals.{args.tag}.pdf'

            if not args.toaerrdist:
                toaerr = args.toaerr*u.us
            else:
                if args.toaerrdist.lower() == 'normal':
                    print(f'Creating a Normal distribution for ToA erros with mean {args.toaerr*u.us} and sigma {0.5*args.toaerr*u.us}')
                    toaerr = np.abs(np.random.normal(loc=args.toaerr, scale=0.5*args.toaerr, size=b35_ntoas))*u.us
                    plt.figure(figsize=(5, 3), dpi=200)
                    plt.hist(toaerr, bins=30, color='crimson', alpha=0.3)
                    plt.xlabel(r'ToA error [$\mu$s]')
                    plt.grid(ls='dashed')
                    plt.tight_layout()
                    plt.savefig(f'{args.outdir}/{args.psrname}.band35.ToAerr.dist.pdf', dpi=200)
                else:
                    print(f'Currently only supports the functionality of a normal distribution.')
                    sys.exit()
            
            freqs_3 = np.linspace(300, 500, args.b3nchan)*u.MHz
            freqs_5 = np.linspace(1260, 1460, args.b5nchan)*u.MHz
            freqs_35 = np.concatenate([freqs_3, freqs_5], axis=0)
            
            toas_35, model, noise_realisations = create_toas(parfile=args.parfile, freqs=freqs_35, ntoas=b35_ntoas, 
                                                             start_MJD=args.start_mjd, finish_MJD=args.finish_mjd, 
                                                             toaerr=toaerr, noise=args.noise, corr_noise=args.corr_noise, 
                                                             flags=flags, name=name, narrowband=True)
            toas_35.write_TOA_file(outfile_temp)

            print(f'Storing the injected realisations of the correlated noise processes in a pickle file')
            with open(f'{args.outdir}/{args.psrname}.band35.noise_realisations.{args.tag}.pkl', 'wb') as pklf:
                pickle.dump(noise_realisations, pklf)
    
            toadata = [f.split('\n')[0] for f in open(outfile_temp, 'r').readlines()[7:]]
            file = open(outfile, 'w+')

            replacement_lines = ["FORMAT 1\n", "MODE 1\n"]
            for text in replacement_lines:
                file.write(text)
            
            for i in range(len(toadata)):
                freq = float(toadata[i].split(' ')[1])
                
                if 300 <= freq <= 500:
                    toadata[i] = f'{toadata[i]} -group b3\n'
                else:
                    toadata[i] = f'{toadata[i]} -group b5\n'
                
                file.write(toadata[i])
            
            file.close()
            os.system(f'rm -r {outfile_temp}')
            plot_residuals(tsim=toas_35, timing_model=model, psrname=args.psrname, 
                           sim_case=args.sim_case, outfilename=plotfile, narrowband=True, no_title=args.no_title)
    
            print(model.components)
            model.write_parfile(f'{args.outdir}/{args.psrname}.band35.{args.tag}.TDB.par')
            
            toaerrs = toas_35.get_errors().value
            toaerrs_scaled = model.scaled_toa_uncertainty(toas_35)
    
            print(f'Correlated Noise : {args.corr_noise}')
            print(f'Injected ToA error : {args.toaerr*u.us:.2e}')
            print(f'Cadence = {args.cadence} days')
            print(f'Median ToA uncertainty (un-scaled) = {np.median(toaerrs*u.us):.2e}')
            print(f'Median ToA uncertainty (scaled) = {np.median(toaerrs_scaled):.2e}')

    elif args.wideband:
        
        print(f'Wideband Simulation')
        freqs = args.nu_wb*np.ones(ntoas)*u.MHz

        if not args.toaerrdist:
            toaerr = args.toaerr*u.us
        else:
            if args.toaerrdist.lower() == 'normal':
                print(f'Creating a Normal distribution for ToA erros with mean {args.toaerr*u.us} and sigma {0.5*args.toaerr*u.us}')
                toaerr = np.abs(np.random.normal(loc=args.toaerr, scale=0.5*args.toaerr, size=ntoas))*u.us
                plt.figure(figsize=(5, 3), dpi=200)
                plt.hist(toaerr, bins=30, color='crimson', alpha=0.3)
                plt.xlabel(r'ToA error [$\mu$s]')
                plt.grid(ls='dashed')
                plt.tight_layout()
                plt.savefig(f'{args.outdir}/{args.psrname}.wideband.ToAerr.dist.pdf', dpi=200)
            else:
                print(f'Currently only supports the functionality of a normal distribution.')
                sys.exit()

        if not args.dmerrdist:
            dmerr = args.dmerr*dmu
        else:
            if args.dmerrdist.lower() == 'normal':
                print(f'Creating a Normal distribution for DM erros with mean {args.dmerr*dmu} and sigma {0.5*args.dmerr*dmu}')
                dmerr = np.abs(np.random.normal(loc=args.dmerr, scale=0.5*args.dmerr, size=ntoas))*dmu
                plt.figure(figsize=(5, 3), dpi=200)
                plt.hist(dmerr, bins=30, color='teal', alpha=0.3)
                plt.xlabel(r'DM error [pc/cm$^3$]')
                plt.grid(ls='dashed')
                plt.tight_layout()
                plt.savefig(f'{args.outdir}/{args.psrname}.wideband.DMerr.dist.pdf', dpi=200)
            else:
                print(f'Currently only supports the functionality of a normal distribution.')
                sys.exit()

        if not args.add_WK_noise:
            toas, model = create_toas(parfile=args.parfile, freqs=freqs, ntoas=ntoas, 
                                      start_MJD=args.start_mjd, finish_MJD=args.finish_mjd, 
                                      toaerr=toaerr, noise=args.noise, corr_noise=args.corr_noise, 
                                      flags=flags, name=name, wideband=True, dmerr=dmerr)
        else:
            toas, model = create_toas(parfile=args.parfile, freqs=freqs, ntoas=ntoas, 
                                      start_MJD=args.start_mjd, finish_MJD=args.finish_mjd, 
                                      toaerr=toaerr, noise=args.noise, corr_noise=False, 
                                      flags=flags, name=name, wideband=True, dmerr=dmerr)
            
            Tspan = (args.finish_mjd - args.start_mjd)*u.d.to(u.s)
            print(f'Time Baseline: {Tspan*u.s.to(u.yr)*u.yr:.2f}')
            fL = 1/(Tspan*args.fL_factor)
            
            if args.add_WK_arn:
                print(f'Adding ARN to noise-free ToAs using WK correlation function with fL = Tspan/{args.fL_factor}')
                noise = get_WK_noise_realisation(args.arn_amp, args.arn_gam, fL, ntoas, args.start_mjd, args.finish_mjd)
                toas.adjust_TOAs(noise*u.s)
                
            if args.add_WK_dmn:
                print(f'Adding DMN to noise-free ToAs using WK correlation function with fL = Tspan/{args.fL_factor}')
                noise = get_WK_noise_realisation(args.dmn_amp, args.dmn_gam, fL, ntoas, args.start_mjd, args.finish_mjd)
                noise_dmn = noise*(1400/args.nu_wb)**2
                toas.adjust_TOAs(noise_dmn*u.s)
                updated_dms = toas.get_dms().value + noise*args.nu_wb**2/DMconst.value
                for flag, dm in zip(toas.table['flags'], updated_dms):
                    flag['pp_dm'] = str(dm)
                
        outfile = f'{args.outdir}/{args.psrname}.pint.simulated.wideband.tim'
        toas.write_TOA_file(outfile)

        # Open the input and output files
        with open(outfile, 'r') as infile, open(f'{args.outdir}/temp', 'w') as temp_file:
            # Pass awk command as arguments
            subprocess.run(['awk', '{print $0 " -group wb"}'], stdin=infile, stdout=temp_file)
        
        replace_first_seven_lines(f'{args.outdir}/temp')
        os.replace(f'{args.outdir}/temp', outfile)
        
        print(model.components)
        model.write_parfile(f'{args.outdir}/{args.psrname}.wideband.{args.tag}.TDB.par')

        toa_resids_filename = f'{args.outdir}/{args.psrname}.ToA.residuals.{args.sim_case}.pdf'
        plot_residuals(tsim=toas, timing_model=model, psrname=args.psrname, sim_case=args.sim_case, 
                       outfilename=toa_resids_filename, wideband=True, no_title=args.no_title)
        
        mjds = np.array([mjd.value for mjd in toas['mjd']])
        dms = toas.get_dms().value
        dmerrs = toas.get_dm_errors().value
        dmerrs_scaled = model.scaled_dm_uncertainty(toas)
        toaerrs = toas.get_errors().value
        toaerrs_scaled = model.scaled_toa_uncertainty(toas)
        
        dmplot_filename = f'{args.outdir}/{args.psrname}.DM.time.series.{args.sim_case}.pdf'
        plot_dms(mjds=mjds, dms=dms, dmerrs=dmerrs, psrname=args.psrname, sim_case=args.sim_case, 
                 outfilename=dmplot_filename, no_title=args.no_title)

        print(f'Reference frequency for WB ToAs and DMs : {args.nu_wb*u.MHz}')
        print(f'Correlated Noise : {args.corr_noise}')
        print(f'Injected ToA error : {args.toaerr*u.us:.2e}')
        print(f'Injected DM error : {args.dmerr*(u.pc/u.cm**3):.2e}')
        print(f'Cadence = {args.cadence} days')
        print(f'Median ToA uncertainty (un-scaled) = {np.median(toaerrs*u.us):.2e}')
        print(f'Median ToA uncertainty (scaled) = {np.median(toaerrs_scaled):.2e}')
        print(f'Median DM uncertainty (un-scaled) = {np.median(dmerrs*(u.pc/u.cm**3)):.2e}')
        print(f'Median DM uncertainty (scaled) = {np.median(dmerrs_scaled):.2e}')
        print(f'Standard Deviation in DMs = {np.std(dms*(u.pc/u.cm**3)):.2e}')

        flagsarr = np.array([flags['f'] for f in range(dms.size)])
        dm_filename = f'{args.outdir}/{args.psrname}.DM.time.series.{args.sim_case}.txt'
        np.savetxt(dm_filename, np.array([mjds, dms, dmerrs, flagsarr]).T, fmt='%s %s %s %s')

        if args.get_DMXpar:
            generate_dmx_parfile(mjds, binsize=args.binsize, parfile=f'{args.outdir}/{args.psrname}.wideband.{args.tag}.TDB.par', 
                                 newparfile=f'{args.outdir}/{args.psrname}.wideband.{args.tag}.TDB.DMX.par', 
                                 fitflags=args.fitflags, returnDMXs=False):

    else:
        print(f'Either Narrowband or Wideband simulation should be selected.')
      
    process_finish = time.time()
    print(f'The process took {(process_finish - process_start):.4f} seconds to finish')