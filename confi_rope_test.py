'''
confi_rope_test.py

用来测试绳子的配置是否正确。
'''
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
##############################
import torch
import numpy as np
import omni.isaac.core.utils.prims as prim_utils

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.sim import SimulationContext

from model_configs import RIGHT_HAND_CFG, LEFT_HAND_CFG, ROPE_CFG 

import omni.isaac.core.utils.prims as prims_utils
import omni.usd
from pxr import Usd, UsdPhysics, UsdGeom, PhysxSchema
from omni.isaac.sensor import ContactSensor


def design_scene() -> tuple[dict, list[list[float]]]:
    """Designs the scene."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # rope_cfg = ROPE_CFG.copy() 
    # rope_cfg.prim_path = "/World/rope"
    # rope_cfg.init_state.pos = (0.0, 0.0, 1.0)

    cfg = sim_utils.UsdFileCfg(usd_path="./models/rope3_no_articulation.usd")
    prim = cfg.func("/World/Table", cfg, translation=(0.0, 0.0, 0.05)) 

    # rope = Articulation(cfg=rope_cfg)
    # rope = Articulation(cfg=rope_cfg)

    scene_entities = []
    return scene_entities


'''
有关 Omniverse USD 教程
https://docs.omniverse.nvidia.com/dev-guide/latest/programmer_ref/usd/hierarchy-traversal/find-prims-by-type.html

有关 Omniverse USD Python API 文档
https://docs.omniverse.nvidia.com/kit/docs/pxr-usd-api/latest/pxr.html
'''
def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, Articulation]):
    """Runs the simulation loop."""
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()

    res = prims_utils.get_prim_at_path('/World/Table')
    children = []
    for prim in res.GetAllChildren():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            children.append(prim)
    print(children)

    res = prims_utils.get_prim_at_path('/World/Table/Item_02')
    print("res attributes")
    print(res.GetAttributes())
    print(res.GetAppliedSchemas())

    sensor = ContactSensor(
        prim_path="/World/Contact_Sensor",
        name="Contact_Sensor",
        frequency=60,
        translation=np.array([0, 0, 0]),
        min_threshold=0,
        max_threshold=10000000,
        radius=-1
    )

    # Simulation loop
    while simulation_app.is_running():
        print(sensor.get_current_frame())
        for item in entities:
            item.write_data_to_sim()
        sim.step()
        for item in entities:
            item.update(sim_dt)


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

