"""Continuous circuit states; compile the small numerical kernel per host ABI."""
import ctypes
import hashlib
import math
import subprocess
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent


def compile_kernel(destination=None):
    src=HERE/'solver.c'
    out=Path(destination) if destination else HERE/'rectifier_kernel.so'
    subprocess.run(['cc','-O2','-shared','-fPIC',str(src),'-o',str(out),'-lm'],check=True)
    return out


class Circuit:
    def __init__(self, capacitances_f, diode_n, *, integration_hz=24000, library=None):
        self.caps=np.asarray(capacitances_f,dtype=float)
        if self.caps.shape!=(3,) or not np.isfinite(self.caps).all() or (self.caps<=0).any():
            raise ValueError('three positive capacitances required')
        if not math.isfinite(diode_n) or not 1<=diode_n<=3:
            raise ValueError('invalid diode model')
        if integration_hz<12000 or integration_hz%240:
            raise ValueError('integration clock must resolve each physics tick')
        self.n=float(diode_n);self.hz=int(integration_hz);self.step_index=0
        self.voltages=np.zeros(3)
        path=Path(library) if library else HERE/'rectifier_kernel.so'
        self.lib=ctypes.CDLL(str(path))
        self.kernel=self.lib.rectifier_advance
        self.kernel.argtypes=[ctypes.c_double]*4+[ctypes.c_longlong,ctypes.c_int,ctypes.c_double,ctypes.c_int,ctypes.c_double,ctypes.c_void_p]
        self.kernel.restype=ctypes.c_double

    def advance(self, steps, module, load_ohm, mode, *, sample=True):
        if type(steps) is not int or steps<=0 or steps>self.hz*30:
            raise ValueError('bounded positive integration steps required')
        if module not in (0,1,2,None) or mode not in ('ON','OFF','DISCHARGE',None):
            raise ValueError('invalid switch')
        if not math.isfinite(load_ohm) or load_ohm<=0:
            raise ValueError('positive load required')
        rows=np.empty((steps,5)) if sample else None
        if module is None:
            if sample:
                rows[:,0]=np.arange(self.step_index+1,self.step_index+steps+1)/self.hz
                rows[:,1]=12*np.sin(2*np.pi*50*rows[:,0]);rows[:,2:]=0
        else:
            # The selected capacitor stays on the load. OFF opens only the source;
            # DISCHARGE also connects the documented 100-ohm shunt. Unselected
            # capacitors are ideal isolated components and retain their voltage.
            self.voltages[module]=self.kernel(float(self.voltages[module]),float(self.caps[module]),float(load_ohm),self.n,
                self.step_index,steps,self.hz,int(mode=='ON'),100. if mode=='DISCHARGE' else 0.,
                rows.ctypes.data if sample else None)
        self.step_index+=steps
        if not np.isfinite(self.voltages).all() or np.min(self.voltages)<-1e-8:
            raise RuntimeError('invalid physical circuit state')
        return rows

    def manifest(self):
        return {'model':'bridge_shockley_heun_v1','integration_hz':self.hz,
                'source_sha256':hashlib.sha256((HERE/'solver.c').read_bytes()).hexdigest()}
