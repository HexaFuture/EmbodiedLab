"""EmbodiedLab bounded Cartesian contact servo, separate from collision-free transit.

The participant explicitly chooses a local contact displacement. IK and joint
retiming happen in the SDK; EmbodiedLab only executes basic physical controls.
This controller does not claim the intentional contact path is collision-free.
"""
import numpy as np
from .kinematics import RobotKinematics
from .trajectory import retime_joint_path, trajectory_velocities


class ContactController:
    def __init__(self,transport,calibration):
        self.transport=transport
        self.kinematics=RobotKinematics(calibration)

    def translate(self,delta_base_m):
        delta=np.asarray(delta_base_m,float)
        if delta.shape!=(3,) or not np.isfinite(delta).all() or not 0<np.linalg.norm(delta)<=.05:
            raise ValueError('contact displacement must be finite and at most 5 cm')
        state=self.transport.call('get_robot_state',{})
        start=self.kinematics.forward(state['joint_positions_rad'])
        segments=max(2,int(np.ceil(np.linalg.norm(delta)/.002)))
        q=state['joint_positions_rad'];path=[q]
        for fraction in np.linspace(0,1,segments+1)[1:]:
            target=start.copy();target[:3,3]+=fraction*delta
            q=self.kinematics.inverse(target,q);path.append(q)
        targets=retime_joint_path(path,state['physics_hz'],speed_limit=.08,acceleration_limit=.25)
        # Differentiate this planned path from its measured starting pose.
        # Omitting feed-forward lets the device differentiate against its old
        # drive setpoint: a small contact tracking error then becomes a spurious
        # one-tick velocity command just as the hand meets the contact.
        velocities=trajectory_velocities(state['joint_positions_rad'],targets,state['physics_hz'])
        final=self.transport.call('execute_joint_trajectory',{'expected_state_id':state['state_id'],
            'targets_rad':targets,'velocities_rad_s':velocities})
        actual=self.kinematics.forward(final['joint_positions_rad'])
        error=float(np.linalg.norm(actual[:3,3]-start[:3,3]-delta))
        completed=error<=.003
        result={'action_completed':completed,'tracking_error_m':error,'state':final,
                'control_mode':'bounded_intentional_contact','collision_free_claim':False}
        if not completed:
            result['note']=(f'the hand stopped {error*100:.1f} cm short of the requested displacement: '
                            'contact with an object or a joint limit; re-observe before the next motion')
        return result
