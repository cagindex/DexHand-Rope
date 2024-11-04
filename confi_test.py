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
import omni.isaac.core.utils.prims as prim_utils

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.sim import SimulationContext

from hands_configs import RIGHT_HAND_CFG, LEFT_HAND_CFG 


def design_scene() -> tuple[dict, list[list[float]]]:
    """Designs the scene."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # Each group will have a robot in it
    origins = [[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]
    # Origin 1
    prim_utils.create_prim("/World/Origin1", "Xform", translation=origins[0])
    # Origin 2
    prim_utils.create_prim("/World/Origin2", "Xform", translation=origins[1])

    # Right Hand
    right_hand_cfg = RIGHT_HAND_CFG.copy()
    right_hand_cfg.prim_path = "/World/Origin1/RightHand"
    right_hand_cfg.spawn.rigid_props.disable_gravity = False
    right_hand_cfg.spawn.rigid_props.retain_accelerations = False
    right_hand_cfg.spawn.rigid_props.enable_gyroscopic_forces = True
    right_hand_cfg.spawn.articulation_props.fix_root_link = False
    right_hand_cfg.spawn.articulation_props.enabled_self_collisions = False
    right_hand_cfg.spawn.articulation_props.solver_position_iteration_count = 16
    right_hand_cfg.spawn.articulation_props.solver_velocity_iteration_count = 4
    right_hand = Articulation(cfg=right_hand_cfg)

    # Left Hand
    left_hand_cfg = LEFT_HAND_CFG.copy()
    left_hand_cfg.prim_path = "/World/Origin2/LeftHand"
    left_hand_cfg.spawn.rigid_props.disable_gravity = False
    left_hand_cfg.spawn.rigid_props.retain_accelerations = False
    left_hand_cfg.spawn.rigid_props.enable_gyroscopic_forces = True
    left_hand_cfg.spawn.articulation_props.fix_root_link = False
    left_hand_cfg.spawn.articulation_props.enabled_self_collisions = False
    left_hand_cfg.spawn.articulation_props.solver_position_iteration_count = 16
    left_hand_cfg.spawn.articulation_props.solver_velocity_iteration_count = 4
    left_hand = Articulation(cfg=left_hand_cfg)

    scene_entities = {"left_hand": left_hand, "right_hand": right_hand}
    return scene_entities


def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, Articulation]):
    """Runs the simulation loop."""
    left_hand = entities["left_hand"]
    right_hand = entities["right_hand"]

    mode = 0
    count = 0
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    # Simulation loop
    while simulation_app.is_running():
        if count % 100 == 0:
            mode = 1 - mode
        left_target_dof = left_hand.data.soft_joint_pos_limits[..., mode]
        right_target_dof = right_hand.data.soft_joint_pos_limits[..., mode]

        # Set target joint positions
        left_hand.set_joint_position_target(left_target_dof)
        right_hand.set_joint_position_target(right_target_dof)

        left_hand.write_data_to_sim()
        right_hand.write_data_to_sim()
        # Perform step
        sim.step()
        count += 1
        # Update buffers
        left_hand.update(sim_dt)
        right_hand.update(sim_dt)


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

