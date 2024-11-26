'''
confi_point_to_test.py

用来测试 Articulation 类型的碰撞
在这之前要测试 confi_pd_test.py
'''
import argparse
from omni.isaac.lab.app import AppLauncher
parser = argparse.ArgumentParser(description="Tutorial on spawning and interacting with a rigid object.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import numpy as np

import omni.isaac.lab.sim as sim_utils
import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.assets import RigidObject, RigidObjectCfg, RigidObjectData, Articulation, ArticulationData
from omni.isaac.lab.sim import SimulationContext
from hands_configs import LEFT_HAND_CFG, RIGHT_HAND_CFG


def vector_local2world(vec, quat):
    return math_utils.quat_apply(quat, vec)
def vector_world2local(vec, quat):
    return math_utils.quat_apply(math_utils.quat_conjugate(quat), vec)

class Articulation_PD_Controller:
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
        return (force_w, torque_w)

class Line_Trajectory:
    def __init__(self, start_pos, end_pos, rate=1.0):
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.sim_time = 0.0
        self.rate = rate
    
    def step(self, dt: float) -> tuple[torch.Tensor, torch.Tensor]:
        self.sim_time += dt * self.rate
        root = (1 - self.sim_time) * self.start_pos + self.sim_time * self.end_pos

        _z_ = torch.tensor([0, 1, 0]).cuda()
        _y_ = torch.tensor([-1, 0, 0]).cuda()
        _x_ = torch.tensor([0, 0, -1]).cuda()
        root_quat = math_utils.quat_from_matrix(torch.stack([_x_, _y_, _z_], dim=1))
        return (root.reshape((-1, 3)), root_quat.reshape((-1, 4)))


target_pos = torch.tensor([0.0, 0.0, 2.0]).reshape((-1, 3)).cuda()
target_quat = torch.tensor([1.0, 0.0, 0.0, 0.0]).reshape((-1, 4)).cuda()

start_pos = torch.tensor([0.0, -3.0, 1.0]).reshape((-1, 3)).cuda()
end_pos = torch.tensor([0.0, -2.0, 1.0]).reshape((-1, 3)).cuda()
pd_controller = Articulation_PD_Controller(
                    pos_kp=1000.0, pos_kd=100.0, ore_kp=10.0, ore_kd=1.0,
                    target_pos=target_pos, target_quat=target_quat)
line_trajectory = Line_Trajectory(start_pos=start_pos, end_pos=end_pos, rate=0.3)

##############################
# Isaac Lab Section
##############################
def design_scene():
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
    cfg.func("/World/Light", cfg)

    # Rigid Body Object
    cube_cfg = RigidObjectCfg(
        prim_path = "/World/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(1.0, 1.0, 1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-0.3, 0, 10.0),
            rot=(np.cos(np.pi/8), 0, 0, np.sin(np.pi/8))
        ),
    )
    cube_object = RigidObject(cfg = cube_cfg)

    # Articulation
    shadow_hand_cfg = LEFT_HAND_CFG.copy()
    shadow_hand_cfg.prim_path = "/World/Robot"
    shadow_hand = Articulation(cfg=shadow_hand_cfg)

    scene_entities = {
        "cube": cube_object,
        "shadow_hand": shadow_hand,
    }
    return scene_entities


def run_simulator(sim: sim_utils.SimulationContext, entities: dict):
    hand = entities["shadow_hand"]
    cube = entities["cube"]
    # select shadow hand body 
    hand_body_names: list = hand.body_names
    selected_body_name = 'lh_forearm'
    body_idx = hand_body_names.index(selected_body_name)

    sim_dt = sim.get_physics_dt()
    while simulation_app.is_running():
        body_quat_w: torch.Tensor = hand.data.body_quat_w[0]

        target_pos, target_quat = line_trajectory.step(sim_dt)

        pd_controller.set_target_pose(target_pos, target_quat)
        hand_force_w, hand_torque_w = pd_controller.compute(hand.data, body_idx)
        hand_force_l = vector_world2local(hand_force_w, body_quat_w[body_idx])
        hand_torque_l = vector_world2local(hand_torque_w, body_quat_w[body_idx])
        hand.set_external_force_and_torque(
            forces=hand_force_l,
            torques=hand_torque_l,
            body_ids=[body_idx],
        )

        hand.write_data_to_sim()
        cube.write_data_to_sim()
        # Perform step
        sim.step()
        # Update buffers
        hand.update(sim_dt)
        cube.update(sim_dt)


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
    main()
    simulation_app.close()

