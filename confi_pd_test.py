import argparse
from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on spawning and interacting with an articulation.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

##############################
# Import Libraries
##############################
import torch
import numpy as np

import omni.isaac.core.utils.prims as prim_utils

import omni.isaac.lab.sim as sim_utils
import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.assets import Articulation, ArticulationData
from omni.isaac.lab.sim import SimulationContext

# Pre-defined configs
from hands_configs import RIGHT_HAND_CFG, LEFT_HAND_CFG

##############################
# PD Controller for Articulation
##############################
class PD_Controller:
    def __init__(self, pos_kp, pos_kd, ore_kp, ore_kd, target_pos, target_quat):
        self.pos_kp = pos_kp
        self.pos_kd = pos_kd
        self.ore_kp = ore_kp
        self.ore_kd = ore_kd
        self.target_pos = target_pos
        self.target_quat = target_quat
        self.target_lin_vel = torch.zeros(3, device=target_pos.device).reshape((-1, 3))
        self.target_ang_vel = torch.zeros(3, device=target_pos.device).reshape((-1, 3))
    
    def set_target_pos(self, target_pos):
        self.target_pos = target_pos
    
    def set_target_quat(self, target_quat):
        self.target_quat = target_quat
    
    def set_target_pose(self, target_pos, target_quat):
        self.set_target_pos(target_pos)
        self.set_target_quat(target_quat)
    
    def compute(self, data: ArticulationData, body_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        current_pos, current_quat = data.body_pos_w[..., body_idx, :], data.body_quat_w[..., body_idx, :]
        current_lin_vel, current_ang_vel = data.body_lin_vel_w[..., body_idx, :], data.body_ang_vel_w[..., body_idx, :]

        position_error, orientation_error = math_utils.compute_pose_error(
            t01=current_pos, q01=current_quat,
            t02=self.target_pos.to(device=current_pos.device), q02=self.target_quat.to(device=current_pos.device),
        )

        force_w = self.pos_kp * position_error - self.pos_kd * (current_lin_vel - self.target_lin_vel.to(device=current_pos.device))
        torque_w = self.ore_kp * orientation_error - self.ore_kd * (current_ang_vel - self.target_ang_vel.to(device=current_pos.device))
        # print("Current position: ", current_pos)
        # print("Target Position: ", self.target_pos)
        # print("force_w: ", force_w)
        return (force_w, torque_w)

def vector_base2world(vec, quat):
    return math_utils.quat_apply(quat, vec)

def vector_world2base(vec, quat):
    return math_utils.quat_apply(math_utils.quat_conjugate(quat), vec)

class Circle_Cone_Trajectory:
    def __init__(self, radius, h, H):
        self.R = radius
        self.h = h
        self.H = H
        self.sim_time = 0.0
        self.tar = torch.tensor([0.0, 0.0, h+H])
    
    def step(self, dt: float)->tuple[torch.Tensor, torch.Tensor]:
        self.sim_time += dt
        root = torch.tensor([self.R*np.cos(self.sim_time), self.R*np.sin(self.sim_time), self.h],
                            device=self.tar.device, dtype=torch.float32)
        _z_ = math_utils.normalize(self.tar - root)
        _y_ = math_utils.normalize(torch.linalg.cross(torch.tensor([0.0, 0.0, 1.0], device=self.tar.device), root))
        _x_ = torch.linalg.cross(_y_, _z_)

        root_quat = math_utils.quat_from_matrix(torch.stack([_x_, _y_, _z_], dim=1))
        return root.reshape((-1, 3)), root_quat.reshape((-1, 4))

pd_controller = PD_Controller(pos_kp=1000.0, pos_kd=100.0, ore_kp=10.0, ore_kd=1.0,
                              target_pos=torch.tensor([[0.0, 0.0, 2.0]]), target_quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
circle_cone_trajectory = Circle_Cone_Trajectory(radius=1.0, h=1.0, H=2.0)
##############################
# Isaac Lab Section
##############################
def design_scene() -> tuple[dict, list[list[float]]]:
    """Designs the scene."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # Articulation
    shadow_hand_cfg = LEFT_HAND_CFG.copy()
    shadow_hand_cfg.prim_path = "/World/Robot"

    shadow_hand = Articulation(cfg=shadow_hand_cfg)
    # return the scene information
    scene_entities = {"shadow_hand": shadow_hand}

    return scene_entities


def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, Articulation]):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability. In general, it is better to access the entities directly from
    #   the dictionary. This dictionary is replaced by the InteractiveScene class in the next tutorial.
    robot = entities["shadow_hand"]
    # print
    robot_body_names: list = robot.body_names
    selected_body_name = 'lh_forearm'
    body_idx = robot_body_names.index(selected_body_name)
    print(robot_body_names)
    print("Selected body name: ", selected_body_name, " with idx: ", body_idx)
    
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    # Simulation loop
    while simulation_app.is_running():
        # robot.set_joint_position_target(torch.zeros_like(robot.data.soft_joint_pos_limits[..., 0]))

        body_pos_w = robot.data.body_pos_w[0]
        body_quat_w = robot.data.body_quat_w[0]

        target_pos, target_quat = circle_cone_trajectory.step(sim_dt)

        pd_controller.set_target_pose(target_pos, target_quat)
        force_w, torque_w = pd_controller.compute(robot.data, body_idx)
        force_b = vector_world2base(force_w, body_quat_w[body_idx])
        torque_b = vector_world2base(torque_w, body_quat_w[body_idx])
        robot.set_external_force_and_torque(
            forces=force_b,
            torques=torque_b,
            body_ids=[body_idx]
        )

        robot.write_data_to_sim()
        # Perform step
        sim.step()
        # Update buffers
        robot.update(sim_dt)


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device="cpu")
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])
    # Design scene
    scene_entities = design_scene()
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene_entities)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

