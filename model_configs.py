'''
存放所有模型的 CFG
'''
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators.actuator_cfg import ImplicitActuatorCfg
from omni.isaac.lab.assets import ArticulationCfg
from omni.isaac.lab.assets import RigidObjectCfg
from omni.isaac.lab.assets import AssetBaseCfg

############################################################
# Asset Configs
############################################################
GROUND_CFG = AssetBaseCfg(
    prim_path="/World/defaultGroundPlane",
    spawn=sim_utils.GroundPlaneCfg()
)

DOME_LIGHT_CFG = AssetBaseCfg(
    prim_path="/World/Light",
    spawn=sim_utils.DomeLightCfg(
        intensity=3000.0,
        color=(0.75, 0.75, 0.75)
    )
)

CROSS_CFG = AssetBaseCfg(
    prim_path="/World/Cross",
    spawn=sim_utils.UsdFileCfg(
        usd_path="./models/cross.usd"
    )
)

############################################################
# Rigid Object Configs
############################################################

############################################################
# Articulation Configs
############################################################
LEFT_HAND_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="./models/left_hand_new.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            enable_gyroscopic_forces=True,
            max_depenetration_velocity=1000.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            fix_root_link=False,
            enabled_self_collisions=False,
            solver_position_iteration_count=64,
            solver_velocity_iteration_count=32,
            sleep_threshold=0.005,
            stabilization_threshold=0.0005,
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
        fixed_tendons_props=sim_utils.FixedTendonPropertiesCfg(limit_stiffness=5.0, damping=0.01),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={".*": 0.0},
    ),
    actuators={
        "fingers": ImplicitActuatorCfg(
            joint_names_expr=["lh_WR.*", "lh_(FF|MF|RF|LF|TH)J(4|3|2|1)", "lh_(LF|TH)J5"],
            effort_limit={
                "lh_WRJ2": 4.785,
                "lh_WRJ1": 2.175,
                "lh_(FF|MF|RF|LF)J2": 0.7245,

                "lh_FFJ(4|3)": 0.9,
                "lh_MFJ(4|3)": 0.9,
                "lh_RFJ(4|3)": 0.9,
                "lh_LFJ(5|4|3)": 0.9,

                "lh_THJ5": 2.3722,
                "lh_THJ4": 1.45,
                "lh_THJ(3|2)": 0.99,
                "lh_THJ1": 0.81,
            },
            stiffness={
                "lh_WRJ.*": 5.0,
                "lh_(FF|MF|RF|LF|TH)J(4|3|2)": 2.0,
                "lh_(LF|TH)J5": 1.0,
                "lh_THJ1": 1.0,
            },
            damping={
                "lh_WRJ.*": 0.5,
                "lh_(FF|MF|RF|LF|TH)J(4|3|2)": 0.1,
                "lh_(LF|TH)J5": 0.1,
                "lh_THJ1": 0.1,
            },
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)


RIGHT_HAND_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="./models/right_hand_new.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            enable_gyroscopic_forces=True,
            max_depenetration_velocity=1000.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            fix_root_link=False,
            enabled_self_collisions=False,
            solver_position_iteration_count=64,
            solver_velocity_iteration_count=32,
            sleep_threshold=0.005,
            stabilization_threshold=0.0005,
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
        fixed_tendons_props=sim_utils.FixedTendonPropertiesCfg(limit_stiffness=5.0, damping=0.01),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={".*": 0.0},
    ),
    actuators={
        "fingers": ImplicitActuatorCfg(
            joint_names_expr=["rh_WR.*", "rh_(FF|MF|RF|LF|TH)J(4|3|2|1)", "rh_(LF|TH)J5"],
            effort_limit={
                "rh_WRJ2": 4.785,
                "rh_WRJ1": 2.175,
                "rh_(FF|MF|RF|LF)J2": 0.7245,
                "rh_FFJ(4|3)": 0.9,
                "rh_MFJ(4|3)": 0.9,
                "rh_RFJ(4|3)": 0.9,
                "rh_LFJ(5|4|3)": 0.9,
                "rh_THJ5": 2.3722,
                "rh_THJ4": 1.45,
                "rh_THJ(3|2)": 0.99,
                "rh_THJ1": 0.81,
            },
            stiffness={
                "rh_WRJ.*": 5.0,
                "rh_(FF|MF|RF|LF|TH)J(4|3|2)": 2.0,
                "rh_(LF|TH)J5": 1.0,
                "rh_THJ1": 1.0,
            },
            damping={
                "rh_WRJ.*": 0.5,
                "rh_(FF|MF|RF|LF|TH)J(4|3|2)": 0.1,
                "rh_(LF|TH)J5": 0.1,
                "rh_THJ1": 0.1,
            },
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)

ROPE_FIXED_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="./models/rope3_fixed.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            # retain_accelerations=False,
            # enable_gyroscopic_forces=True,
            # max_depenetration_velocity=1000.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            fix_root_link=True,
            enabled_self_collisions=True,
            # solver_position_iteration_count=64,
            # solver_velocity_iteration_count=4,
            sleep_threshold=0.005,
            stabilization_threshold=0.0005,
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
        rot=(1.0, 0.0, 0.0, 0.0),
    ),
)

ROPE_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="./models/rope3.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            # retain_accelerations=False,
            # enable_gyroscopic_forces=True,
            # max_depenetration_velocity=1000.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            fix_root_link=False,
            enabled_self_collisions=True,
            # solver_position_iteration_count=64,
            # solver_velocity_iteration_count=4,
            sleep_threshold=0.005,
            stabilization_threshold=0.0005,
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            ".*": 0.2,
        },
    ),
    actuators={
        "fingers": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit={
                ".*": 0.5,
            },
            stiffness={
                ".*": 5.0,
            },
            damping={
                ".*": 0.5,
            },
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
