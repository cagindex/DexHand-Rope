import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators.actuator_cfg import ImplicitActuatorCfg
from omni.isaac.lab.assets.articulation import ArticulationCfg

LEFT_HAND_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="./models/left_hand.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            retain_accelerations=True,
            max_depenetration_velocity=1000.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
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
                "lh_(FF|MF|RF|LF|TH)J(4|3|2)": 1.0,
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
        usd_path="./models/left_hand.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            retain_accelerations=True,
            max_depenetration_velocity=1000.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
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
                "lh_(FF|MF|RF|LF|TH)J(4|3|2)": 1.0,
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

