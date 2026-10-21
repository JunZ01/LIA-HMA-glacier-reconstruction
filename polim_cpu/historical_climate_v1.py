"""Selected-year climate export and original project annual MonthlyTI adapter."""
import zipfile
import numpy as np
from .v3.model import climate_at

def selected_system_years(path, key, system_index, first, count):
    # Read only the requested numerical records, including from compressed NPZ.
    # Header/byte seeking never interprets temperatures from excluded years.
    with zipfile.ZipFile(path) as archive, archive.open(key+'.npy') as f:
        version=np.lib.format.read_magic(f)
        shape,fortran,dtype=np.lib.format._read_array_header(f,version)
        assert not fortran and len(shape)==3 and shape[2]==12
        assert 0<=system_index<shape[0] and 0<=first and first+count<=shape[1]
        offset=((system_index*shape[1]+first)*12)*dtype.itemsize
        f.seek(offset,1)
        raw=f.read(count*12*dtype.itemsize)
        assert len(raw)==count*12*dtype.itemsize
        return np.frombuffer(raw,dtype=dtype).copy().reshape(count,12)

def annual(surface, climate, parameters, year, backend):
    return climate_at(surface,climate,parameters,year,backend)[0]
