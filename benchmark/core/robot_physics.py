"""Contact physics of the Franka arm, shared by every bench.

The scene's Franka has no actuator inertia on its joints, and its links have almost no rotational inertia (the wrist
links about 1e-7 kg m^2; a real arm's are about 1e-2). Nothing bounded the speed at which the contact solver pushes
an overlapping link out, so a contact could turn the wrist at 10-200 rad/s and fling it past its joint range.

Real joints carry their motor's reflected rotor inertia (armature), so each arm joint is given one here. The solver's
push-out speed is bounded on the arm links and the hand.

The two fingers keep the solver's default push-out speed: a two-finger hold rests on the solver pushing the held
part out of the fingers."""
from pxr import Usd, UsdPhysics, PhysxSchema

ARM_JOINTS = tuple(f"panda_joint{i}" for i in range(1, 8))
JOINT_ARMATURE_KG_M2 = 0.1        # reflected rotor inertia per arm joint (as in the MuJoCo Menagerie Franka model)
MAX_DEPENETRATION_M_S = 0.1       # bound on the solver's push-out speed for the arm links and the hand
FINGERS = ("panda_leftfinger", "panda_rightfinger")


def configure_robot_contacts(stage, robot_path):
    """Author armature on the seven arm joints and the depenetration bound on the arm links and the hand."""
    for prim in Usd.PrimRange(stage.GetPrimAtPath(robot_path)):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI) and prim.GetName() not in FINGERS:
            PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateMaxDepenetrationVelocityAttr(MAX_DEPENETRATION_M_S)
        if prim.IsA(UsdPhysics.RevoluteJoint) and prim.GetName() in ARM_JOINTS:
            PhysxSchema.PhysxJointAPI.Apply(prim).CreateArmatureAttr(JOINT_ARMATURE_KG_M2)
