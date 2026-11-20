"""Reject force-bearing or touching/penetrating contacts, not separated proximity pairs.

PhysX contact buffers may contain separated zero-force proximity points.
No force tolerance is used: any nonzero normal force is conservatively counted.
"""
import numpy as np

def summarize_contacts(detail, filters, robot_root):
    if detail is None or len(detail) != 6:
        raise RuntimeError('contact details missing')
    counts=np.asarray(detail[4]); starts=np.asarray(detail[5])
    forces=np.asarray(detail[0]).reshape(-1); separations=np.asarray(detail[3]).reshape(-1)
    if counts.shape != starts.shape or counts.ndim != 2 or counts.shape[1] != len(filters):
        raise RuntimeError('contact filter dimensions mismatch')
    total=int(counts.sum()); physical_count=0
    full=total >= len(forces)      # a full buffer is contact: summarise the stored points, count at least one
    categories=set(); peak=0.; separation=float('inf')
    for row in range(counts.shape[0]):
        for col,path in enumerate(filters):
            n=int(counts[row,col]); start=int(starts[row,col])
            if full and n>0 and 0<=start and start+n>len(forces): n=max(0,len(forces)-start)
            if n<0 or (n and (start<0 or start+n>len(forces))): raise RuntimeError('invalid contact span')
            if not n: continue
            f=forces[start:start+n];d=separations[start:start+n]
            if not np.isfinite(f).all() or not np.isfinite(d).all():raise RuntimeError('nonfinite contact data')
            physical=(np.abs(f)>0.) | (d<=0.)
            physical_count+=int(physical.sum())
            if not physical.any():continue
            f=f[physical];d=d[physical]
            categories.add('robot' if path==robot_root or path.startswith(robot_root+'/') else 'apparatus')
            peak=max(peak,float(np.abs(f).max()));separation=min(separation,float(d.min()))
    if full: physical_count=max(physical_count,1)
    return physical_count,dict(categories=sorted(categories),max_normal_force_n=peak,min_separation_m=separation if physical_count else 0.)
