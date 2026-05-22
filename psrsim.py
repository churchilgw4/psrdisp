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

setup_log(level="WARNING")
process_start = time.time()

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
        fig = plt.figure(figsize=(10, 8), dpi=150)
        
        # Plot residuals vs MJD
        plt.subplot(211)
        plt.errorbar(tsim.get_mjds(), res.time_resids.to_value("us"), res.get_data_error().to_value("us"), 
                     fmt='o', color='crimson', ms=2, ecolor='k', elinewidth=0.5, capsize=0.5, capthick=0.25)
        
        plt.xlabel("MJD", fontweight='bold', fontsize=10)
        plt.ylabel(r"Residuals [$\mu$s]", fontweight='bold', fontsize=10)
        plt.grid(ls='dashed')
        
        # Plot residuals vs frequency
        plt.subplot(212)
        plt.errorbar(tsim.get_freqs(), res.time_resids.to_value("us"), res.get_data_error().to_value("us"), 
                     fmt='o', color='crimson', ms=2, ecolor='k', elinewidth=0.5, capsize=0.5, capthick=0.25)
        
        plt.xlabel("Frequency [MHz]", fontweight='bold', fontsize=10)
        plt.ylabel(r"Residuals [$\mu$s]", fontweight='bold', fontsize=10)
        if not no_title:
            fig.suptitle(f'{psrname} simulated ToA residuals : {sim_case}', fontweight='bold', fontsize=13)
        plt.grid(ls='dashed')
        plt.tight_layout()

        if outfilename:
            plt.savefig(outfilename, dpi=200)
    
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

if __name__ == "__main__":
  
    # Argument parser
    parser = argparse.ArgumentParser(description="Simulate TOAs and plot residuals with PINT.")
    
    parser.add_argument("--psrname", type=str, required=True, help="Name of the pulsar")
    parser.add_argument("--parfile", type=str, required=True, help="Path to the par file for the timing model")
    parser.add_argument("--sim_case", type=str, required=True, help="Simulation case (same as the WN parameter flag)")
    parser.add_argument("--narrowband", action="store_true", help="Simulate Narrowband ToAs")
    parser.add_argument("--wideband", action="store_true", help="Simulate Wideband ToAs and DMs")
    parser.add_argument("--pardir", type=str, default='./', help='Path to the par file directory')
    parser.add_argument("--tag", type=str, default=None, help="Path to the par file for the timing model")
  
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
    
    parser.add_argument("--noise", action="store_true", default=False, 
                        help="Flag to show if time-uncorrelated noises are included in par file")
    parser.add_argument("--corr_noise", action="store_true", default=False, 
                        help="Flag to show if time-correlated noises are included in par file")
    parser.add_argument("--no_title", action="store_true", default=False, 
                        help="Flag to not have titles in plots")

    parser.add_argument("--outdir", type=str, default='./', help='Path to the output directory')
  
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

        
        toas, model = create_toas(parfile=args.parfile, freqs=freqs, ntoas=ntoas, 
                                  start_MJD=args.start_mjd, finish_MJD=args.finish_mjd, 
                                  toaerr=toaerr, noise=args.noise, corr_noise=args.corr_noise, 
                                  flags=flags, name=name, wideband=True, dmerr=dmerr)
        
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

    else:
        print(f'Either Narrowband or Wideband simulation should be selected.')
      
    process_finish = time.time()
    print(f'The process took {(process_finish - process_start):.4f} seconds to finish')