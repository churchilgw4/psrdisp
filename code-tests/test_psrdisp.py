import numpy as np
import pytest
import os
from disputils import *

basedir = os.path.dirname(os.path.abspath(__file__))

parfile = os.path.join(basedir, f'utils', f'parfile.par')
assert os.path.isfile(parfile), f'The par file {parfile} does not exist'

timfile = os.path.join(basedir, f'utils', f'timfile.tim')
assert os.path.isfile(timfile), f'The tim file {timfile} does not exist'

paramfile = os.path.join(basedir, f'utils', f'paramfile.txt')
assert os.path.isfile(paramfile), f'The parameter file {paramfile} does not exist'

truthsfile = os.path.join(basedir, f'utils', f'truthsfile.txt')
assert os.path.isfile(truthsfile), f'The truths file {truthsfile} does not exist'

dmfile = os.path.join(basedir, f'utils', f'dmfile.txt')
assert os.path.isfile(dmfile), f'The DM time series file {dmfile} does not exist'

def test_DispParams():
    pidx = 0
    pname = f'test-param'
    pmin = 0
    pmax = 10
    pdist = f'uniform'
    pflag = f'test-flag'
    t, dms, dmerrs, flags = read_dmfile(dmfile=dmfile, return_flags=True)
    pmask = flags == pflag
    
    param = DispParam(pidx, name=pname, min=pmin, max=pmax, dist=pdist, flag=pflag, mask=pmask)
    assert param.idx == pidx, f'Invalid parameter index {param.pidx}'
    assert param.name == pname, f'Invalid parameter name {param.name}'
    assert param.min == pmin, f'Invalid parameter minimum value {param.min}'
    assert param.max == pmax, f'Invalid parameter maximum value {param.max}'
    assert param.dist == pdist, f'Invalid parameter distribution {param.dist}'
    assert param.flag == pflag, f'Invalid parameter flag {param.flag}'
    assert param.is_white == False, f'Invalid parameter classification'
    assert param.is_red == False, f'Invalid parameter classification'
    assert param.is_deterministic == True, f'Invalid parameter classification'
    assert t.size == dms.size == dmerrs.size == flags.size, f'Invalid array sizes'

    params = get_DispParams(paramfile=paramfile, flags=flags)
    assert len([par for par in params if par.is_white]) == 1
    assert len([par for par in params if par.is_red]) == 2
    assert len([par for par in params if par.is_deterministic]) == 3
        