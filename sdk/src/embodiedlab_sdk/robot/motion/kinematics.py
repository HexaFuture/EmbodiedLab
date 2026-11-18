"""SDK kinematics from the benchmark's public robot installation/CAD calibration.

No USD, simulator handles or object-state oracle is used. This numerical IK
solver is a contact-controller component, not a collision-free transit planner.
"""
import math
import numpy as np

LOWER=np.array([-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-.0175,-2.8973])
UPPER=np.array([2.8973,1.7628,2.8973,-.0698,2.8973,3.7525,2.8973])


def rotation_vector(matrix):
    cosine=np.clip((np.trace(matrix)-1)/2,-1,1)
    angle=math.acos(cosine)
    axis=np.array([matrix[2,1]-matrix[1,2],matrix[0,2]-matrix[2,0],matrix[1,0]-matrix[0,1]])
    if angle<1e-7:return axis/2
    if math.pi-angle<1e-5:
        # Eigenvector avoids division by sin(pi).
        values,vectors=np.linalg.eig(matrix)
        direction=np.real(vectors[:,np.argmin(abs(values-1))])
        return angle*direction/np.linalg.norm(direction)
    return axis*angle/(2*math.sin(angle))


def quaternion_matrix(quaternion):
    q=np.asarray(quaternion,dtype=float)
    if q.shape!=(4,) or not np.isfinite(q).all() or abs(np.linalg.norm(q)-1)>.001:
        raise ValueError('normalized finite wxyz quaternion required')
    w,x,y,z=q
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
        [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
        [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


class RobotKinematics:
    def __init__(self, calibration):
        self.chain=calibration['joint_chain']
        self.tip=np.asarray(calibration['last_link_to_hand'],dtype=float)
        self.base=np.asarray(calibration['world_from_robot_base'],dtype=float)
        if len(self.chain)!=7 or self.tip.shape!=(4,4) or self.base.shape!=(4,4):raise ValueError('unsupported robot calibration')
        self.transforms=[]
        for joint in self.chain:
            a,b=np.asarray(joint['parent_to_joint'],float),np.asarray(joint['child_to_joint'],float)
            if a.shape!=(4,4) or b.shape!=(4,4) or not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('invalid kinematic frame')
            self.transforms.append((a,np.linalg.inv(b),'XYZ'.index(joint['axis'])))

    def forward(self,joints):
        q=np.asarray(joints,float)
        if q.shape!=(7,) or not np.isfinite(q).all():raise ValueError('seven finite joint positions required')
        transform=np.eye(4)
        for angle,(parent,child,axis) in zip(q,self.transforms):
            rot=np.eye(4);i,j=((1,2),(2,0),(0,1))[axis];c,s=math.cos(angle),math.sin(angle)
            rot[i,i]=rot[j,j]=c;rot[i,j]=-s;rot[j,i]=s
            transform=transform@parent@rot@child
        return transform@self.tip

    def inverse(self,target,start,*,max_iterations=120):
        target=np.asarray(target,float);q=np.asarray(start,float).copy()
        if target.shape!=(4,4) or not np.isfinite(target).all():raise ValueError('finite homogeneous target required')
        for _ in range(max_iterations):
            pose=self.forward(q)
            error=np.r_[target[:3,3]-pose[:3,3],rotation_vector(target[:3,:3]@pose[:3,:3].T)]
            if np.linalg.norm(error[:3])<.0002 and np.linalg.norm(error[3:])<.002:return q.tolist()
            jac=np.empty((6,7))
            for i in range(7):
                shifted=q.copy();shifted[i]+=1e-5;other=self.forward(shifted)
                jac[:,i]=np.r_[other[:3,3]-pose[:3,3],rotation_vector(other[:3,:3]@pose[:3,:3].T)]/1e-5
            step=jac.T@np.linalg.solve(jac@jac.T+1e-4*np.eye(6),error)
            q=np.clip(q+step*min(1,.12/max(np.linalg.norm(step),1e-12)),LOWER+.005,UPPER-.005)
        raise RuntimeError('IK did not converge within limits; no motion executed')

    def residual(self,target,joints):
        pose=self.forward(joints)
        return (float(np.linalg.norm(target[:3,3]-pose[:3,3])),
                float(np.linalg.norm(rotation_vector(target[:3,:3]@pose[:3,:3].T))))

    def solve(self,target,seeds,*,position_tol_m=1e-4,rotation_tol_rad=5e-4,max_iterations=300):
        """Kinematics only: joints within LOWER..UPPER that reach target (4x4, robot base frame), damped least squares
        from each seed in turn. Returns (joints or None, position residual m, rotation residual rad) of the best try;
        joints only when both residuals are within tolerance, never an approximate solution."""
        target=np.asarray(target,float)
        if target.shape!=(4,4) or not np.isfinite(target).all():raise ValueError('finite homogeneous target required')
        best=(None,math.inf,math.inf)
        for seed in seeds:
            q=np.clip(np.asarray(seed,float),LOWER,UPPER)
            for _ in range(max_iterations):
                pose=self.forward(q)
                error=np.r_[target[:3,3]-pose[:3,3],rotation_vector(target[:3,:3]@pose[:3,:3].T)]
                if np.linalg.norm(error[:3])<position_tol_m and np.linalg.norm(error[3:])<rotation_tol_rad:break
                jac=np.empty((6,7))
                for i in range(7):
                    shifted=q.copy();shifted[i]+=1e-6;other=self.forward(shifted)
                    jac[:,i]=np.r_[other[:3,3]-pose[:3,3],rotation_vector(other[:3,:3]@pose[:3,:3].T)]/1e-6
                step=jac.T@np.linalg.solve(jac@jac.T+1e-5*np.eye(6),error)
                q=np.clip(q+step*min(1,.2/max(np.linalg.norm(step),1e-12)),LOWER,UPPER)
            position,rotation=self.residual(target,q)
            if position<position_tol_m and rotation<rotation_tol_rad:
                return q.tolist(),position,rotation
            if position+.1*rotation<best[1]+.1*best[2]:best=(None,position,rotation)
        return best
