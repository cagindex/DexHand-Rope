import argparse
from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, RigidObject, RigidObjectCfg
from omni.isaac.lab.assets.articulation import ArticulationCfg
from omni.isaac.lab.actuators.actuator_cfg import ImplicitActuatorCfg
from omni.isaac.lab.sim import SimulationContext
from omni.isaac.lab_assets import SHADOW_HAND_CFG

import torch
import numpy as np
from my_utils import *
from socket_manager import Socket_Manager
from pd_controller import Robot_Controller, Rigid_Body_Controller
from hands_configs import RIGHT_HAND_CFG, LEFT_HAND_CFG

def design_scene() -> tuple[dict, list[list[float]]]:
    """Designs the scene."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # Rigid body
    table_cfg = RigidObjectCfg(
        prim_path = "/World/Table",
        spawn=sim_utils.CuboidCfg(
            size=(1.5, 2.0, 0.40),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.35),
            rot=(0.0, 0.0, 1.0, 0.0)
        ),
    )
    object_cfg = RigidObjectCfg(
        prim_path = "/World/Object",
        spawn=sim_utils.CylinderCfg(
            radius=0.03,
            height=0.30,
            axis='Z',
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.01),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
        ),
        # spawn=sim_utils.CuboidCfg(
        #     size=(0.05, 0.05, 0.05),
        #     rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        #     mass_props=sim_utils.MassPropertiesCfg(mass=0.012),
        #     collision_props=sim_utils.CollisionPropertiesCfg(),
        #     visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
        # ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.8),
            rot=(1.0, 0.0, 0.0, 0.0)
        ),
    )
    # table = RigidObject(cfg = table_cfg)
    # object = RigidObject(cfg = object_cfg)


    # Articulation
    shadow_hand_cfg = RIGHT_HAND_CFG.copy()
    shadow_hand_cfg.prim_path = "/World/Robot"
    shadow_hand_cfg.spawn.rigid_props.disable_gravity = False
    shadow_hand_cfg.spawn.rigid_props.retain_accelerations = False
    shadow_hand_cfg.spawn.rigid_props.enable_gyroscopic_forces = True
    shadow_hand_cfg.spawn.articulation_props.fix_root_link = False
    shadow_hand_cfg.spawn.articulation_props.enabled_self_collisions = False
    shadow_hand_cfg.spawn.articulation_props.solver_position_iteration_count = 16
    shadow_hand_cfg.spawn.articulation_props.solver_velocity_iteration_count = 4

    shadow_hand_cfg.init_state.pos = (0.0, 0.0, 1.0)
    shadow_hand = Articulation(cfg=shadow_hand_cfg)
    # return the scene information
    scene_entities = {"shadow_hand": shadow_hand}
    # scene_entities = {"shadow_hand": shadow_hand, "rigids": [table, object]}

    return scene_entities

def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, Articulation]):
    # socket_server_manager.listen()
    robot = entities["shadow_hand"]
    # rigids = entities["rigids"]
    body_name = 'rh_forearm'
    body_idx = get_body_index(robot, body_name)

    print(robot.joint_names, len(robot.joint_names))

    sim_dt = sim.get_physics_dt()
    # Simulation loop
    # fp = './data/rightHand.txt'
    # manager = File_Manager(fp) 
    manager = Socket_Manager(host='127.0.0.1', port=12345)
    manager.listen()
    robot_controller = Robot_Controller(pos_kp=1000.0, pos_kd=100.0, ore_kp=10.0, ore_kd=0.5)
    while simulation_app.is_running():
        # position, orientation, dof = manager.step()
        manager.send()
        position, orientation, dof = manager.receive()

        robot.set_joint_position_target(dof)
        robot_controller.set_target_pose(position, orientation)
        robot_controller.step(robot, body_idx)
        # rb_controller.step(rigid)

        robot.write_data_to_sim()
        # rigids[0].write_data_to_sim()
        # rigids[1].write_data_to_sim()
        # Perform step
        sim.step(render=True)
        # Update buffers
        robot.update(sim_dt)
        # rigids[0].update(sim_dt)
        # rigids[1].update(sim_dt)
        # break


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(
        device="cpu",
        dt=1.0/60.0,
        render_interval=1,)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([0.0, -2.5, 4.0], [0.0, 0.0, 2.0])
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

