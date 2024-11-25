import torch
import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.assets import ArticulationData, RigidObjectData
from my_utils import vector_world2local, vector_local2world, get_body_index, Info

class PD_Controller:
    def __init__(self, pos_kp, pos_kd, ore_kp, ore_kd):
        self.pos_kp = pos_kp
        self.pos_kd = pos_kd
        self.ore_kp = ore_kp
        self.ore_kd = ore_kd
        self.target_pos = None
        self.target_quat = None
        self.target_lin_vel = torch.zeros(3).float().reshape((-1, 3))
        self.target_ang_vel = torch.zeros(3).float().reshape((-1, 3))

    def set_target_pos(self, target_pos):
        self.target_pos = target_pos.reshape((-1, 3))
    
    def set_target_quat(self, target_quat):
        self.target_quat = target_quat.reshape((-1, 4))
    
    def set_target_pose(self, target_pos, target_quat):
        self.set_target_pos(target_pos)
        self.set_target_quat(target_quat)
    
    def compute(self, current_pos, current_quat, current_lin_vel, current_ang_vel) -> tuple[torch.Tensor, torch.Tensor]:
        position_error, orientation_error = math_utils.compute_pose_error(
            t01=current_pos, q01=current_quat,
            t02=self.target_pos.to(device=current_pos.device), q02=self.target_quat.to(device=current_pos.device),
        )

        force_w = self.pos_kp * position_error - self.pos_kd * (current_lin_vel - self.target_lin_vel.to(device=current_pos.device))
        torque_w = self.ore_kp * orientation_error - self.ore_kd * (current_ang_vel - self.target_ang_vel.to(device=current_pos.device))
        return (force_w, torque_w)

class Articulation_PD_Controller(PD_Controller):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def step(self, robot, body_idx):
        return self.__step(robot, body_idx)
    
    def sstep(self, robot, body_idx):
        body_quat_w = robot.data.body_quat_w[0]
        current_pos, current_quat = robot.data.body_pos_w[..., body_idx, :], robot.data.body_quat_w[..., body_idx, :]
        current_lin_vel, current_ang_vel = robot.data.body_lin_vel_w[..., body_idx, :], robot.data.body_ang_vel_w[..., body_idx, :]
        force_w, torque_w = self.compute(current_pos, current_quat, current_lin_vel, current_ang_vel)
        force_b = vector_world2local(force_w, body_quat_w[body_idx])
        torque_b = vector_world2local(torque_w, body_quat_w[body_idx])
        robot.set_external_force_and_torque(
            forces=force_b,
            torques=torque_b,
            body_ids=[body_idx],
        )

class Rigid_Body_Controller(PD_Controller):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def step(self, rigid_body):
        quat = rigid_body.data.root_quat_w

        current_pos, current_quat = rigid_body.data.root_pos_w, rigid_body.data.root_quat_w
        current_linear_vel, current_angular_vel = rigid_body.data.root_lin_vel_w, rigid_body.data.root_ang_vel_w
        force_w, torque_w = self.compute(current_pos, current_quat, current_linear_vel, current_angular_vel)
        force_b = vector_world2local(force_w, quat)
        torque_b = vector_world2local(torque_w, quat)
        rigid_body.set_external_force_and_torque(
            forces=force_b,
            torques=torque_b,
        )
    


class Articulation_Controller(Articulation_PD_Controller):
    def __init__(self, robot, root_name, **kwargs):
        super().__init__(**kwargs)
        self.body_idx = get_body_index(robot, root_name)
    
    def step(self, robot, info: Info):
        robot.set_joint_position_target(info.dof)
        self.set_target_pose(info.pos, info.quat)
        self.sstep(robot, self.body_idx)
