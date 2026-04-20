import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from robotis_lab.assets.object import ROBOTIS_LAB_OBJECT_ASSETS_DATA_DIR
DUMMY_RAG_CFG = RigidObjectCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ROBOTIS_LAB_OBJECT_ASSETS_DATA_DIR}/object/dummy_rag.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
        ),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=[0.79481, -0.15896, -0.07875],
        rot=[0.7071, 0.0, 0.0, 0.7071],
    ),
)