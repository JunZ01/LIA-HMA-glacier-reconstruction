"""Equivalent activity ROW replacement; no active column deletion."""
import numpy as np
from scipy import sparse

def build_P(T,area,active,w,mode='new'):
    assert T.dtype==area.dtype==np.dtype('float64') and active.dtype==np.dtype('bool')
    if mode=='old':
        with w.span('P_area_scale'):P=sparse.diags(1./area)@T
        with w.span('P_to_LIL'):P=P.tolil()
        with w.span('P_active_rows'):P[active,:]=0;P[active,active]=1.
    elif mode=='new':
        with w.span('P_area_and_active_rows'):P=sparse.diags((~active)/area)@T+sparse.diags(active.astype(T.dtype))
    else:raise ValueError(mode)
    with w.span('P_to_CSC'):P=P.tocsc()
    return P

