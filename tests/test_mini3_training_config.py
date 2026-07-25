from __future__ import annotations

import glob
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import joblib
import numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from humanoidverse.utils.motion_lib.motion_lib_robot import MotionLibRobot
from humanoidverse.utils.motion_lib.torch_humanoid_batch import Humanoid_Batch
from humanoidverse.utils.robot_config import indices_by_name, validate_robot_config

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "humanoidverse/config"
MINI3_URDF = REPO_ROOT / "humanoidverse/data/robots/mini3/urdf/mini3.urdf"
MINI3_MJCF = REPO_ROOT / "humanoidverse/data/robots/mini3/mjcf/mini3.xml"
G1_MJCF = REPO_ROOT / "humanoidverse/data/robots/g1/g1_29dof.xml"
MINI3_MOTIONS = REPO_ROOT / "humanoidverse/data/lafan1_mini3"


class Mini3TrainingConfigTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
            cls.config = compose(config_name="exp/bfm_zero/bfm_zero_mini3")

    def test_robot_schema(self) -> None:
        schema = validate_robot_config(self.config.robot)
        self.assertEqual(schema["robot"], "mini3")
        self.assertEqual(schema["num_dofs"], 21)
        self.assertEqual(schema["num_bodies"], 24)
        self.assertEqual(schema["state_dim"], 48)
        self.assertEqual(schema["action_dim"], 21)

    def test_asset_names_and_order(self) -> None:
        urdf_root = ET.parse(MINI3_URDF).getroot()
        urdf_links = [link.attrib["name"] for link in urdf_root.findall("link")]
        urdf_joints = [
            joint.attrib["name"]
            for joint in urdf_root.findall("joint")
            if joint.attrib.get("type") != "fixed"
        ]

        mjcf_root = ET.parse(MINI3_MJCF).getroot()
        actuator_joints = [motor.attrib["joint"] for motor in list(mjcf_root.find("actuator"))]

        self.assertEqual(list(self.config.robot.body_names), urdf_links)
        self.assertEqual(list(self.config.robot.dof_names), urdf_joints)
        self.assertEqual(list(self.config.robot.dof_names), actuator_joints)

    def test_motion_schema(self) -> None:
        motion_files = sorted(glob.glob(str(MINI3_MOTIONS / "*.pkl")))
        self.assertTrue(motion_files, f"No Mini3 motion files found in {MINI3_MOTIONS}")

        required_keys = {"root_pos", "root_rot", "dof_pos", "fps"}
        for motion_file in motion_files:
            motion = joblib.load(motion_file)
            self.assertTrue(required_keys.issubset(motion), motion_file)
            self.assertEqual(motion["root_pos"].shape[-1], 3, motion_file)
            self.assertEqual(motion["root_rot"].shape[-1], 4, motion_file)
            self.assertEqual(motion["dof_pos"].shape[-1], 21, motion_file)
            self.assertEqual(
                (
                    motion["root_pos"].shape[0],
                    motion["root_rot"].shape[0],
                    motion["dof_pos"].shape[0],
                ),
                (motion["dof_pos"].shape[0],) * 3,
                motion_file,
            )
            self.assertTrue(np.isfinite(motion["root_pos"]).all(), motion_file)
            self.assertTrue(np.isfinite(motion["root_rot"]).all(), motion_file)
            self.assertTrue(np.isfinite(motion["dof_pos"]).all(), motion_file)
            self.assertTrue(
                (
                    motion["dof_pos"].min(axis=0)
                    >= np.asarray(self.config.robot.dof_pos_lower_limit_list) - 1e-2
                ).all(),
                motion_file,
            )
            self.assertTrue(
                (
                    motion["dof_pos"].max(axis=0)
                    <= np.asarray(self.config.robot.dof_pos_upper_limit_list) + 1e-2
                ).all(),
                motion_file,
            )
            np.testing.assert_allclose(
                np.linalg.norm(motion["root_rot"], axis=-1),
                1.0,
                atol=1e-5,
                err_msg=motion_file,
            )

    def test_motion_fk_actuator_order_is_compatible(self) -> None:
        for mjcf_path in (G1_MJCF, MINI3_MJCF):
            mjcf_root = ET.parse(mjcf_path).getroot()
            actuator_joints = [motor.attrib["joint"] for motor in list(mjcf_root.find("actuator"))]
            articulated_body_joints = [
                joint.attrib["name"]
                for body in mjcf_root.findall(".//worldbody//body")
                for joint in body.findall("joint")
                if joint.attrib.get("name") in actuator_joints
            ]
            self.assertEqual(actuator_joints, articulated_body_joints, str(mjcf_path))

    def test_mini3_motion_parser_matches_direct_dof_layout(self) -> None:
        mesh_parser = Humanoid_Batch(self.config.robot.motion)

        self.assertEqual(mesh_parser.num_dof, len(self.config.robot.dof_names))
        self.assertEqual(tuple(mesh_parser.dof_axis.shape), (len(self.config.robot.dof_names), 3))
        self.assertEqual(len(mesh_parser.actuated_joints_idx), len(self.config.robot.dof_names))

    def test_mini3_direct_motion_loads_through_fk(self) -> None:
        motion_cfg = OmegaConf.create(OmegaConf.to_container(self.config.robot.motion, resolve=True))
        motion_cfg.motion_file = str(MINI3_MOTIONS)

        motion_lib = MotionLibRobot(motion_cfg, num_envs=1, device="cpu")

        motion_file = sorted(glob.glob(str(MINI3_MOTIONS / "*.pkl")))[0]
        result = motion_lib.load_motion_with_skeleton(
            np.asarray([0]),
            np.asarray([motion_file]),
            None,
            motion_lib.fix_height,
            None,
            8,
            None,
            0,
        )
        curr_motion = result[0][1]

        self.assertEqual(curr_motion.global_translation.shape[1:], (len(self.config.robot.body_names), 3))
        self.assertEqual(curr_motion.global_rotation.shape[1:], (len(self.config.robot.body_names), 4))
        self.assertEqual(curr_motion.dof_pos.shape[1], len(self.config.robot.dof_names))

    def test_name_mapping_rejects_invalid_inputs(self) -> None:
        self.assertEqual(indices_by_name(["base", "left", "right"], ["right", "base"]), [2, 0])
        with self.assertRaises(ValueError):
            indices_by_name(["base", "base"], ["base"])
        with self.assertRaises(ValueError):
            indices_by_name(["base"], ["missing"])


if __name__ == "__main__":
    unittest.main()
