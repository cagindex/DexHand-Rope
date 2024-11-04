import torch
import numpy as np
import omni.isaac.lab.utils.math as math_utils

def convert_quat_to_wxyz(q: torch.Tensor)->torch.Tensor:
    # convert quaternion from xyzw to wxyz
    return q[:, [3, 0, 1, 2]]

def vector_local2world(vec, quat):
    return math_utils.quat_apply(quat, vec)

def vector_world2local(vec, quat):
    return math_utils.quat_apply(math_utils.quat_conjugate(quat), vec)

def get_body_index(robot, body_name: str):
    robot_body_names: list = robot.body_names
    return robot_body_names.index(body_name) if body_name is not None else 0

def str2tensor(s: str)->tuple[torch.Tensor, torch.Tensor]:
    bvh_str, thumb_dofs = s.split('|')
    bvh_data = torch.tensor(np.fromstring(bvh_str.strip(), dtype=float, sep=' '))
    thumb_dofs = torch.tensor(np.fromstring(thumb_dofs.strip(), dtype=float, sep=' '))
    return bvh_data, thumb_dofs

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



def right_hand_decode(frame_data: str)->tuple[torch.tensor, torch.tensor, torch.tensor]:
    frame_data, thumb_dofs = str2tensor(frame_data) 
    indices = [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19]

    # Assign root position and quaternion and joints quaternion  
    root_pos = position_left2right(frame_data[:3])
    quaternions = convert_quat_to_wxyz(frame_data[3:].reshape(-1, 4)[indices])
    root_quat = quaternion_left2right(quaternions[0])
    thumb_quat = quaternions[1]
    finger_root_orientation = quaternions[[4, 7 ,10, 13]]
    other_joint_orientation = quaternions[[2, 3, 5, 6, 8, 9, 11, 12, 14, 15]]

    # Init joints indices
    dof1_positive_idx = [12, 17, 13, 18, 14, 19, 20, 22]
    dof1_negative_idx = [21, 23]
    dof2_finger_idx = [[1, 2, 3, 10], [7, 8, 9, 15]]
    dof2_thumb_idx = [[5], [11]]
    dof1_thumb = [21, 23]

    dof = torch.zeros(24, dtype=root_pos.dtype)
    #1. convert finger other joints orientation
    x, y, z = euler_from_quat(other_joint_orientation, 1, 2, 3)
    dof[dof1_negative_idx + dof1_positive_idx] = -z.flatten() 
    dof[dof1_negative_idx] *= -1

    #2. Convert finger root orientation
    x, z, y = euler_from_quat(finger_root_orientation, 1, 3, 2)
    # dof[dof2_finger_idx[0]] = -y.flatten()
    dof[dof2_finger_idx[1]] = -z.flatten()

    #3. Convert thumb root orientation
    # RyP45 = math_utils.quat_from_euler_xyz(torch.tensor(0), torch.tensor(np.pi/4), torch.tensor(0)).double()
    # RxN90 = math_utils.quat_from_euler_xyz(-torch.tensor(np.pi/2), torch.tensor(0), torch.tensor(0)).double()
    # # thumb_quat_ = quat_mul(thumb_quat, quat_mul(RxN90, RyP45))
    # # thumb_quat_new = quat_mul(RyP45, quat_mul(thumb_quat_, quat_inv(RyP45)))
    # thumb_quat_new = math_utils.quat_mul(RyP45, math_utils.quat_mul(thumb_quat, RxN90))
    # print("新坐标系下的大拇指根旋转：")
    # print(thumb_quat_new.round_(decimals=4).tolist())
    # y, x, z = euler_from_quat(thumb_quat_new.reshape(-1, 4), 2, 1, 3)
    # print("大拇指角度xyz：")
    # print("%2f %2f %2f" % (180*x.item()/np.pi, 180*y.item()/np.pi, 180*z.item()/np.pi))
    # theta_x, theta_z = decode_quaternion_to_2dof(thumb_quat_new)
    # print("大拇指关节角度xz：")
    # print("%2f %2f" % (180*theta_x.item()/np.pi, 180*theta_z.item()/np.pi))
    dof[dof2_thumb_idx[0]] = thumb_dofs[0]
    dof[dof2_thumb_idx[1]] = -thumb_dofs[1]
    dof[dof1_thumb[0]] = thumb_dofs[2]
    dof[dof1_thumb[1]] = thumb_dofs[3]
    return root_pos.float(), root_quat.float(), dof



