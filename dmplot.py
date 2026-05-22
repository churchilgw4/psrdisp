import numpy as np
import matplotlib.pyplot as plt
import os
import sys

psrname = sys.argv[1]
filename_b35 = sys.argv[2]
filename_b3 = sys.argv[3]
tag = sys.argv[4]
    
file_b35 = [f.split('\n')[0] for f in open(filename_b35, 'r').readlines()]
mjd_b35 = [float(f.split(' ')[0]) for f in file_b35]
dm_b35 = [float(f.split(' ')[1]) for f in file_b35]
dmerr_b35 = [float(f.split(' ')[2]) for f in file_b35]

file_b3 = [f.split('\n')[0] for f in open(filename_b3, 'r').readlines()]
mjd_b3 = [float(f.split(' ')[0]) for f in file_b3]
dm_b3 = [float(f.split(' ')[1]) for f in file_b3]
dmerr_b3 = [float(f.split(' ')[2]) for f in file_b3]

plt.figure(figsize=(10, 5), dpi=150)
plt.errorbar(mjd_b35, dm_b35, yerr=dmerr_b35, fmt='o', markersize=3, c='k', elinewidth=0.2, capthick=0.5, capsize=1, label='Band 3+5 DM')
plt.errorbar(mjd_b3, dm_b3, yerr=dmerr_b3, fmt='o', markersize=3, c='r', elinewidth=0.2, capthick=0.5, capsize=1, label='Band 3 DM')
plt.ticklabel_format(useOffset=False)
plt.ylabel(r'DM ($\text{pc}\;\text{cm}^{-3}$)', fontweight='bold', fontsize=13)
plt.xlabel('MJD', fontweight='bold', fontsize=13)
plt.title(f'J1909-3744 {tag} DM time series', fontweight='bold', fontsize=15)
plt.legend(loc='upper right')
plt.savefig(f'J1909-3744.{tag}.DM.timeseries.pdf')
