'''
confi_rope_test.py

用来测试绳子的配置是否正确。
'''
import argparse
from omni.isaac.lab.app import AppLauncher
# add argparse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

##############################
##############################
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import AssetBaseCfg
from omni.isaac.lab.assets import RigidObject, RigidObjectCfg
from omni.isaac.lab.assets import Articulation, ArticulationCfg
from omni.isaac.lab.scene import InteractiveScene, InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationContext
from omni.isaac.lab.utils import configclass
from my_utils import reset_scene_pose

import torch
import numpy as np
import keyboard as ky
from model_configs import RIGHT_HAND_CFG, LEFT_HAND_CFG, ROPE_CFG 



@configclass
class SceneCfg(InteractiveSceneCfg):
    """Designs the scene."""
    # Ground-plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg()
    )
    # Lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(
            intensity=3000.0,
            color=(0.75, 0.75, 0.75)
        )
    )
    #rope
    rope: ArticulationCfg = ROPE_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Rope"
    )
    rope.init_state.pos = (0.1, 0.3, 1.0)


'''
有关 Omniverse USD 教程
https://docs.omniverse.nvidia.com/dev-guide/latest/programmer_ref/usd/hierarchy-traversal/find-prims-by-type.html

有关 Omniverse USD Python API 文档
https://docs.omniverse.nvidia.com/kit/docs/pxr-usd-api/latest/pxr.html
'''
def run_simulator(sim: sim_utils.SimulationContext, scene:InteractiveScene):
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()

    # Simulation loop
    while simulation_app.is_running():
        if ky.is_pressed('enter'):
            reset_scene_pose(scene)
        scene.write_data_to_sim()
        sim.step(render=True)
        scene.update(sim_dt)


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(
        device="cpu"
    )
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])
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

