import torch
import numpy as np
import omni.isaac.lab.utils.math as math_utils

######################################
# Class Definition
######################################
class Info:
    def __init__(self, pos: torch.Tensor, quat: torch.Tensor, dof: torch.Tensor):
        self.pos = pos
        self.quat = quat
        self.dof = dof


######################################
# Function Definition
######################################

def get_body_index(robot, body_name: str):
    robot_body_names: list = robot.body_names
    return robot_body_names.index(body_name) if body_name is not None else 0

def str2tensor(s: str)->tuple[torch.Tensor, torch.Tensor]:
    bvh_str, thumb_dofs = s.split('(')
    bvh_data = torch.tensor(np.fromstring(bvh_str.strip(), dtype=float, sep=' '))
    thumb_dofs = torch.tensor(np.fromstring(thumb_dofs[:-1].strip(), dtype=float, sep=' '))
    return bvh_data, thumb_dofs

def vector_local2world(vec, quat):
    return math_utils.quat_apply(quat, vec)

def vector_world2local(vec, quat):
    return math_utils.quat_apply(math_utils.quat_conjugate(quat), vec)

def convert_quat_to_wxyz(q: torch.Tensor)->torch.Tensor:
    # convert quaternion from xyzw to wxyz
    return q[:, [3, 0, 1, 2]]

@torch.jit.script
def position_left2right(pos: torch.Tensor):
    # convert position from left-handed to right-handed coordinate system
    # Follow the convention: x->x, y->z, z->y
    pos = pos.reshape(-1, 3)
    return pos[:, [0, 2, 1]]

@torch.jit.script
def quaternion_left2right(quat: torch.Tensor):
    # convert quaternion from left-handed to right-handed coordinate system
    # Follow the convention: x->x, y->z, z->y
    quat = quat.reshape(-1, 4)
    qt = quat[:, [0, 1, 3, 2]]
    qt[:, 1:] *= -1
    return qt

@torch.jit.script
def normalize_angle(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))

@torch.jit.script
def quat_to_angle_axis(q):
    # q must be normalized
    min_theta = 1e-5
    qx, qy, qz, qw = 1, 2, 3, 0

    sin_theta = torch.sqrt(1 - q[..., qw] * q[..., qw])
    angle = 2 * torch.acos(q[..., qw])
    angle = normalize_angle(angle)
    sin_theta_expand = sin_theta.unsqueeze(-1)
    axis = q[..., qx:] / sin_theta_expand

    mask = sin_theta > min_theta
    default_axis = torch.zeros_like(axis)
    default_axis[..., -1] = 1

    angle = torch.where(mask, angle, torch.zeros_like(angle))
    mask_expand = mask.unsqueeze(-1)
    axis = torch.where(mask_expand, axis, default_axis)
    return angle, axis


@torch.jit.script
def euler_from_quat(quat: torch.Tensor, i:int , j:int , k: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if i == k :
        not_proper = False
        k = 6 - i - j
    else:
        not_proper = True
    eps = (i - j) * (j - k) * (k - i) / 2

    if not_proper:
        a = quat[:, 0] - quat[:, j]
        b = quat[:, i] + quat[:, k] * eps
        c = quat[:, j] + quat[:, 0]
        d = quat[:, k] * eps - quat[:, i]
    else:
        a = quat[:, 0]
        b = quat[:, i]
        c = quat[:, j]
        d = quat[:, k] * eps

    theta2 = torch.acos((a*a + b*b) - 1)

    theta_add = torch.atan2(b, a)
    theta_sub = torch.atan2(d, c)

    theta1 = theta_add - theta_sub
    theta3 = theta_add + theta_sub

    theta1 = torch.where(theta2==0, 0, theta1)
    theta3 = torch.where(theta2==0, 2*theta_add - theta1, theta3)
    theta1 = torch.where(theta2==torch.pi/2, 0, theta1) 
    theta3 = torch.where(theta2==torch.pi/2, 2*theta_sub + theta1, theta3)

    if not_proper:
        theta2 -= torch.pi / 2
        theta3 *= eps

    return normalize_angle(theta1), normalize_angle(theta2), normalize_angle(theta3)

@torch.jit.script
def decode_quaternion_to_2dof(quat: torch.Tensor)->tuple[torch.Tensor, torch.Tensor]:
    vec = torch.tensor([1, 0, 0], dtype=quat.dtype)
    new_vec = torch.min( math_utils.quat_apply(quat, vec), torch.tensor([1000, 0, 1000], dtype=quat.dtype) )

    cross = torch.linalg.cross(new_vec, vec)
    dot = torch.dot(new_vec, vec)
    
    theta_x = -torch.atan(cross[1]/cross[2])
    theta_z = torch.acos(dot)
    return theta_x, theta_z


# region Decode from Unity
def right_hand_decode(frame_data: str)->tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    frame_data, thumb_dofs = str2tensor(frame_data) 
    return __right_hand_decode(frame_data, thumb_dofs)

def left_hand_decode(frame_data: str)->tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    frame_data, thumb_dofs = str2tensor(frame_data) 
    return __left_hand_decode(frame_data, thumb_dofs)


@torch.jit.script
def __right_hand_decode(frame_data: torch.Tensor, thumb_dofs: torch.Tensor)->tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    indices = [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19]

    # Assign root position and quaternion and joints quaternion  
    root_pos = position_left2right(frame_data[:3])
    quaternions = convert_quat_to_wxyz(frame_data[3:].reshape(-1, 4)[indices])
    root_quat = quaternion_left2right(quaternions[0])
    # 这些 index 是从 unity 那边导出的对应的 index 勿动
    finger_root_orientation = quaternions[[4, 7 ,10, 13]]
    other_joint_orientation = quaternions[[5, 6, 8, 9, 11, 12, 14, 15]]

    # Init joints indices
    dof1_positive_idx = [
        12, 17, 13, 18, 14, 19, 20, 22 # 这一行是控制四个手指其他关节的上下旋转的 hinge index
    ]
    dof2_finger_idx = [
        [1, 2, 3, 10], # 这一行控制四个手指根部的上下旋转的 hinge index
        [7, 8, 9, 15]  # 这一行控制四个手指根部的左右旋转的 hinge index
    ]

    dof = torch.zeros(24, dtype=root_pos.dtype)
    # 转换四个手指上下旋转（这里不考虑左右旋转了）
    x, z, y = euler_from_quat(finger_root_orientation, 1, 3, 2)
    # dof[dof2_finger_idx[0]] = -y.flatten()

    dof[dof2_finger_idx[1]] = -z.flatten()
    # 转换四个手指其他关节的上下旋转
    x, y, z = euler_from_quat(other_joint_orientation, 1, 2, 3)
    dof[dof1_positive_idx] = -z.flatten() 

    # 直接读取 thumb dof
    dof[[5, 11, 16, 21, 23]] = thumb_dofs

    # 为了方便将中指的旋转直接赋予到其他手指上
    dof[[1, 3, 10]] = dof[[1]] # 根部旋转统一
    dof[[12, 14, 20]] = dof[[13]] # J2 统一
    dof[[17, 19, 22]] = dof[[18]] # J3 统一
    return root_pos.float(), root_quat.float(), dof


@torch.jit.script
def __left_hand_decode(frame_data: torch.Tensor, thumb_dofs: torch.Tensor)->tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    indices = [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19]

    # Assign root position and quaternion and joints quaternion  
    root_pos = position_left2right(frame_data[:3])
    quaternions = convert_quat_to_wxyz(frame_data[3:].reshape(-1, 4)[indices])
    root_quat = quaternion_left2right(quaternions[0])
    finger_root_orientation = quaternions[[4, 7 ,10, 13]]
    other_joint_orientation = quaternions[[5, 6, 8, 9, 11, 12, 14, 15]]

    # Init joints indices
    dof1_positive_idx = [
        12, 17, 13, 18, 14, 19, 20, 22
    ]
    dof2_finger_idx = [
        [1, 2, 3, 10],
        [7, 8, 9, 15]
    ]

    dof = torch.zeros(24, dtype=root_pos.dtype)
    #2. Convert finger root orientation
    x, z, y = euler_from_quat(finger_root_orientation, 1, 3, 2)
    # dof[dof2_finger_idx[0]] = -y.flatten()
    dof[dof2_finger_idx[1]] = z.flatten()

    #1. convert finger other joints orientation
    x, y, z = euler_from_quat(other_joint_orientation, 1, 2, 3)
    dof[dof1_positive_idx] = z.flatten() 

    #3. Convert thumb root orientation
    dof[[5, 11, 16, 21, 23]] = thumb_dofs
    return root_pos.float(), root_quat.float(), dof
# endregion

def set_object_gravity(obj, gravity: bool, env_ids: list = [0]):
    """
    Set the gravity of an object.
    gravity: True to enable gravity, False to disable.
    obj must be omni.isaac.lab.assets.RigidObject or omni.isaac.lab.assets.ArticulatedObject.
    """
    env_ids = torch.tensor(env_ids, device=obj.device)
    current_gravity_status = obj.root_physx_view.get_disable_gravities()
    # 0: disable_gravity=false(with gravity) / 1: disable_gravity=true(without gravity)
    current_gravity_status[env_ids] = int(not gravity)
    obj.root_physx_view.set_disable_gravities(current_gravity_status, env_ids)