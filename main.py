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
from controller import Articulation_Controller, Rigid_Body_Controller
from model_configs import RIGHT_HAND_CFG, LEFT_HAND_CFG

def design_scene() -> tuple[dict, list[list[float]]]:
    """Designs the scene."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # Rope
    rope_cfg = sim_utils.UsdFileCfg(usd_path="models/ropes500.usd")
    rope_cfg.func("/World/rope", rope_cfg, translation=(0.0, -0.1, 0.8))

    # Articulation
    right_hand_cfg = RIGHT_HAND_CFG.copy()
    right_hand_cfg.prim_path = "/World/hands/right_hand"

    left_hand_cfg = LEFT_HAND_CFG.copy()
    left_hand_cfg.prim_path = "/World/hands/left_hand"

    right_hand_cfg.init_state.pos = (0.5, 0.0, 0.8)
    left_hand_cfg.init_state.pos = (-0.5, 0.0, 0.8)

    right_hand = Articulation(cfg=right_hand_cfg)
    left_hand = Articulation(cfg=left_hand_cfg)
    # return the scene information
    scene_entities = {
        "shadow_hand": {
            "left_hand": left_hand,
            "right_hand": right_hand,
        },
    }
    return scene_entities

def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, Articulation]):
    # socket_server_manager.listen()
    hands = entities["shadow_hand"]
    left_hand = hands["left_hand"]
    right_hand = hands["right_hand"]

    print("The Left Hand Joint Names: ")
    print(left_hand.joint_names)
    print("The Right Hand Joint Names: ")
    print(right_hand.joint_names)

    sim_dt = sim.get_physics_dt()
    # Simulation loop
    manager = Socket_Manager(host='127.0.0.1', port=6781)
    manager.listen()
    left_hand_controller = Articulation_Controller(
        robot=left_hand, root_name='lh_forearm',
        pos_kp=1000.0, pos_kd=100.0, ore_kp=10.0, ore_kd=1.0
    )
    right_hand_controller = Articulation_Controller(
        robot=right_hand, root_name='rh_forearm',
        pos_kp=1000.0, pos_kd=100.0, ore_kp=10.0, ore_kd=1.0
    )
    while simulation_app.is_running():
        # position, orientation, dof = manager.step()
        manager.send()
        left_info, right_info = manager.receive()

        left_hand_controller.step(left_hand, left_info)
        right_hand_controller.step(right_hand, right_info)
        # rb_controller.step(rigid)

        left_hand.write_data_to_sim()
        right_hand.write_data_to_sim()
        # Perform step
        sim.step(render=True)
        # Update buffers
        left_hand.update(sim_dt)
        right_hand.update(sim_dt)


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

