"""One trusted parameter contract for both SI forces and evaluation truth.

These functions are benchmark internals, never participant tools. Frequencies
are undamped normal-mode angular frequencies, not uncoupled local frequencies.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Dynamics:
    local_omega_1: float
    local_omega_2: float
    coupling_per_s2: float
    damping_per_s: float
    mass_kg: float = 0.2

    def __post_init__(self):
        for value in (self.local_omega_1,self.local_omega_2,self.coupling_per_s2,
                      self.damping_per_s,self.mass_kg):
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
                raise ValueError('finite physical parameters required')
        if min(self.local_omega_1,self.local_omega_2,self.mass_kg) <= 0:
            raise ValueError('positive local frequencies and mass required')
        if self.coupling_per_s2 <= 0 or self.damping_per_s < 0:
            raise ValueError('positive coupling and nonnegative damping required')
        if self.damping_per_s >= self.normal_angular_frequencies()[0]:
            raise ValueError('both modes must be underdamped')

    def normal_angular_frequencies(self):
        a = self.local_omega_1**2 + self.coupling_per_s2
        d = self.local_omega_2**2 + self.coupling_per_s2
        delta = math.hypot(a-d, 2*self.coupling_per_s2)
        return (math.sqrt((a+d-delta)/2), math.sqrt((a+d+delta)/2))

    def forces_newton(self, displacements_m, velocities_m_s):
        if len(displacements_m) != 2 or len(velocities_m_s) != 2:
            raise ValueError('two slider states required')
        if not all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v)
                   for v in (*displacements_m,*velocities_m_s)):
            raise ValueError('finite slider states required')
        x1,x2 = displacements_m
        v1,v2 = velocities_m_s
        c,g,m = self.coupling_per_s2,self.damping_per_s,self.mass_kg
        return (m*(-self.local_omega_1**2*x1-c*(x1-x2)-2*g*v1),
                m*(-self.local_omega_2**2*x2-c*(x2-x1)-2*g*v2))

    def damped_angular_frequencies(self):
        return tuple(math.sqrt(w*w-self.damping_per_s**2)
                     for w in self.normal_angular_frequencies())
