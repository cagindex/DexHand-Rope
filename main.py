import argparse
from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import RigidObjectCfg, AssetBaseCfg
from omni.isaac.lab.assets import ArticulationCfg
from omni.isaac.lab.scene import InteractiveScene, InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationContext
from omni.isaac.lab.utils import configclass

import torch
import numpy as np
from my_utils import *
from socket_manager import Socket_Manager
from controller import Articulation_Controller, Rigid_Body_Controller
from model_configs import RIGHT_HAND_CFG, LEFT_HAND_CFG

@configclass
class SceneCfg(InteractiveSceneCfg):
    """Designs the scene."""
    # Ground-plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    # Lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # Rope
    rope = AssetBaseCfg(
        prim_path="/World/rope",
        spawn=sim_utils.UsdFileCfg(usd_path="models/rope3_no_articulation.usd"),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, 1.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        )
    )

    # Articulation
    right_hand: ArticulationCfg = RIGHT_HAND_CFG.replace(
        prim_path="{ENV_REGEX_NS}/right_hand",
    )

    left_hand: ArticulationCfg = LEFT_HAND_CFG.replace(
        prim_path="{ENV_REGEX_NS}/left_hand",
    )

    right_hand.init_state.pos = (0.5, 0.0, 0.8)
    left_hand.init_state.pos = (-0.5, 0.0, 0.8)

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    # socket_server_manager.listen()
    left_hand = scene["left_hand"]
    right_hand = scene["right_hand"]

    for i, name in list(enumerate(right_hand.joint_names)):
        print(f"Joint {i}: {name}")

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

        scene.write_data_to_sim()
        # Perform step
        sim.step(render=True)
        # Update buffers
        scene.update(sim_dt)


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
    scene_cfg = SceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

