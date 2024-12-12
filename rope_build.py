# Build Rope with isaac sim scripts
import omni
import numpy as np
from pxr import UsdLux, UsdGeom, Sdf, Gf, UsdPhysics, UsdShade, PhysxSchema
from omni.isaac.core.objects import DynamicCapsule
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.objects.ground_plane import GroundPlane
from omni.isaac.core.physics_context import PhysicsContext

start_offset = np.array([0.0, 0.0, 0.5])

num_capsules = 50
capsule_radius = 0.02
capsule_height = 0.02
capsule_density = 0.00005
along_axis = "-Z"
capsule_color=np.array([1.0, 0.0, 0.0])

ground = False

capsule_distance = capsule_height + capsule_radius # By convention, the distance should be (capsule_radius + capsule_height)

JointFatherPath = "/World/Joints"
coneAngleLimit = 110
rope_damping = 10
rope_stiffness = 1

fixed = True
exclude_from_articulation = True

#################################################################################
# Utils functions
#################################################################################
along_axis = along_axis.upper()
sign = -1 if along_axis[0] == "-" else 1
stage = omni.usd.get_context().get_stage()
def createJoint(stage, jointPath, coneAngleLimit, rope_damping, rope_stiffness):        
    joint = UsdPhysics.Joint.Define(stage, jointPath)

    # locked DOF (lock - low is greater than high)
    d6Prim = joint.GetPrim()
    limitAPI = UsdPhysics.LimitAPI.Apply(d6Prim, "transX")
    limitAPI.CreateLowAttr(1.0)
    limitAPI.CreateHighAttr(-1.0)
    limitAPI = UsdPhysics.LimitAPI.Apply(d6Prim, "transY")
    limitAPI.CreateLowAttr(1.0)
    limitAPI.CreateHighAttr(-1.0)
    limitAPI = UsdPhysics.LimitAPI.Apply(d6Prim, "transZ")
    limitAPI.CreateLowAttr(1.0)
    limitAPI.CreateHighAttr(-1.0)
    limitAPI = UsdPhysics.LimitAPI.Apply(d6Prim, "rotX")
    limitAPI.CreateLowAttr(1.0)
    limitAPI.CreateHighAttr(-1.0)

    # Moving DOF:
    dofs = ["rotY", "rotZ"]
    for d in dofs:
        limitAPI = UsdPhysics.LimitAPI.Apply(d6Prim, d)
        limitAPI.CreateLowAttr(-coneAngleLimit)
        limitAPI.CreateHighAttr(coneAngleLimit)

        # joint drives for rope dynamics:
        driveAPI = UsdPhysics.DriveAPI.Apply(d6Prim, d)
        driveAPI.CreateTypeAttr("force")
        driveAPI.CreateDampingAttr(rope_damping)
        driveAPI.CreateStiffnessAttr(rope_stiffness)
    return joint

def createJoints(num_joints):
    joints = []
    for i in range(num_joints):
        jointPath = Sdf.Path(f"{JointFatherPath}/Joint_{i}")
        joint = createJoint(stage, jointPath, coneAngleLimit, rope_damping, rope_stiffness)
        joints.append(joint)
    return joints

def design_rope_capsules():
    capsules = []

    _capsules_start_offset_value = (capsule_height / 2 + capsule_radius)
    if along_axis[-1] == "X":
        dx = np.array([1.0, 0.0, 0.0]) * sign
        dq = (0.7071, 0.0, 0.7071, 0.0)
    elif along_axis[-1] == "Y":
        dx = np.array([0.0, 1.0, 0.0]) * sign
        dq = (0.7071, 0.7071, 0.0, 0.0)
    else:   
        dx = np.array([0.0, 0.0, 1.0]) * sign
        dq = (1.0, 0.0, 0.0, 0.0)

    for i in range(num_capsules):
        capsule = DynamicCapsule(
            prim_path=f"/World/Capsule_{i}",
            position=dx * i * capsule_distance + (start_offset + _capsules_start_offset_value * dx),
            color=capsule_color,
            radius=capsule_radius,
            height=capsule_height,
            orientation=dq,
        )
        capsules.append(capsule)
    
    return capsules

def design_joint(joint, object0, object1, local_dis0, local_dis1):
    # Add ref body target
    joint.GetBody0Rel().AddTarget(object0.prim_path)
    joint.GetBody1Rel().AddTarget(object1.prim_path)

    # Add Local Position
    joint.GetLocalPos0Attr().Set(local_dis0)
    joint.GetLocalPos1Attr().Set(local_dis1)

    # Excluded from articulation
    joint.GetExcludeFromArticulationAttr().Set(exclude_from_articulation)

def design_rope(capsules, joints):
    dx = Gf.Vec3f(0.0, 0.0, 1.0)
    local_dis = (dx * (capsule_radius + capsule_height) / 2) * sign * (-1 if along_axis[-1] == "Y" else 1)
    for i in range(num_capsules - 1):
        joint = joints[i]
        capsule0 = capsules[i]
        capsule1 = capsules[i + 1]
        design_joint(joint, capsule0, capsule1, local_dis, -local_dis)

    return joints

def design_scene():
    capsules = design_rope_capsules()
    joints = createJoints(num_capsules - 1)
    design_rope(capsules, joints)
    if fixed :
        dx = Gf.Vec3f(0.0, 0.0, 1.0) * sign * (-1 if along_axis[-1] == "Y" else 1)
        local_dis = dx * (capsule_height / 2 + capsule_radius)
        fixed_joint = createJoint(stage, f"{JointFatherPath}/FixedJoint", 170, rope_damping, rope_stiffness)
        object0 = XFormPrim(f"/World/FixedPoint", position=start_offset)
        object1 = capsules[0]
        design_joint(fixed_joint, object0, object1, Gf.Vec3f(0.0, 0.0, 0.0), -local_dis)
        fixed_joint.GetLocalRot0Attr().Set(Gf.Quatf(0.7071, 0.0, 0.7071, 0.0))






##############################################################################
# Main function
##############################################################################
def main():
    PhysicsContext()
    if ground:
        GroundPlane(prim_path="/World/groundPlane", size=10, color=np.array([0.5, 0.5, 0.5]))
    design_scene( )

main()