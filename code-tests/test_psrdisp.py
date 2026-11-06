import numpy as np
import pytest
from validation_utils import *
from disputils import *

def test_DispParams():
    pidx = 0
    pname = 'test-param'
    pmin = 0
    pmax = 10
    pdist = 'uniform'
    pflag = 'test-flag'
    param = DispParam(pidx, name=pname, min=pmin, max=pmax, dist=pdist, flag=pflag)
    
    assert param.pidx == pidx, f'Invalid parameter index {param.pidx}'
    assert param.name == pname, f'Invalid parameter name {param.name}'
    assert param.min == pmin, f'Invalid parameter minimum value {param.min}'
    assert param.max == pmax, f'Invalid parameter maximum value {param.max}'
    assert param.dist == pdist, f'Invalid parameter distribution {param.dist}'
    assert param.flag == pflag, f'Invalid parameter flag {param.flag}'
    assert param.is_white == False
    assert param.is_red == False
    assert param.is_deterministic == True

def test_get_DispParams():
    paramfile = get_paramfile()
    params = get_DispParams(paramfile=paramfile, flags=True)
    assert len([par for par in params if par.is_white]) == 1
    assert len([par for par in params if par.is_red]) == 2
    assert len([par for par in params if par.is_deterministic]) == 3
        