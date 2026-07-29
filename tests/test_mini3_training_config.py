from __future__ import annotations

import glob
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from humanoidverse.agents.envs.humanoidverse_isaac import HumanoidVerseVectorEnv
from humanoidverse.agents.fb.agent import sample_unique_env_indices
from humanoidverse.agents.fb_cpr.agent import replace_state_with_clean_state
from humanoidverse.utils.helpers import build_reference_policy_state
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
        if not OmegaConf.has_resolver("eval"):
            OmegaConf.register_new_resolver("eval", eval)
        with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
            cls.config = compose(config_name="exp/bfm_zero/bfm_zero_mini3")
            cls.g1_config = compose(
                config_name="exp/bfm_zero/bfm_zero",
                overrides=["robot=g1/g1_29dof_hard_waist"],
            )

    def test_robot_schema(self) -> None:
        schema = validate_robot_config(self.config.robot)
        self.assertEqual(schema["robot"], "mini3")
        self.assertEqual(schema["num_dofs"], 21)
        self.assertEqual(schema["num_bodies"], 24)
        self.assertEqual(schema["state_dim"], 48)
        self.assertEqual(schema["action_dim"], 21)

    def test_discriminator_policy_observation_is_noise_free(self) -> None:
        self.assertEqual(
            list(self.config.obs.obs_dict.clean_policy_obs),
            ["base_ang_vel", "projected_gravity", "dof_pos", "dof_vel", "max_local_self"],
        )
        self.assertIn("clean_policy_obs", self.config.obs.no_noise_obs_keys)

        noisy_state = torch.ones(2, 48)
        clean_state = torch.zeros(2, 48)
        policy_obs = {
            "state": noisy_state,
            "privileged_state": torch.randn(2, 358),
        }
        discriminator_obs = replace_state_with_clean_state(policy_obs, clean_state)

        self.assertIs(discriminator_obs["state"], clean_state)
        self.assertIs(discriminator_obs["privileged_state"], policy_obs["privileged_state"])
        self.assertIs(policy_obs["state"], noisy_state)

    def test_discriminator_dof_position_uses_nominal_reference(self) -> None:
        wrapper = HumanoidVerseVectorEnv.__new__(HumanoidVerseVectorEnv)
        dummy_base_env = SimpleNamespace(close=lambda: None)
        wrapper._env = SimpleNamespace(
            unwrapped=dummy_base_env,
            obs_buf_dict_raw={
                "clean_policy_obs": {
                    # This value uses the randomized actor reference and must not reach the discriminator.
                    "dof_pos": torch.tensor([[0.1, 0.6]]),
                    "dof_vel": torch.tensor([[0.3, 0.4]]),
                    "projected_gravity": torch.tensor([[0.0, 0.0, -1.0]]),
                    "base_ang_vel": torch.tensor([[0.5, 0.6, 0.7]]),
                }
            },
            simulator=SimpleNamespace(dof_pos=torch.tensor([[1.1, 2.2]])),
            default_dof_pos=torch.tensor([[1.0, 2.0]]),
            config=SimpleNamespace(
                robot=SimpleNamespace(actions_dim=2),
                obs=SimpleNamespace(obs_scales=SimpleNamespace(dof_pos=2.0)),
            ),
        )

        clean_state = wrapper._get_clean_policy_state(to_numpy=False)

        torch.testing.assert_close(clean_state[:, :2], torch.tensor([[0.2, 0.4]]))
        torch.testing.assert_close(
            clean_state[:, 2:],
            torch.tensor([[0.3, 0.4, 0.0, 0.0, -1.0, 0.5, 0.6, 0.7]]),
        )

    def test_expert_rollout_environment_sampling_is_unique(self) -> None:
        indices = sample_unique_env_indices(num_envs=1024, percentage=0.5, device="cpu")

        self.assertEqual(indices.numel(), 512)
        self.assertEqual(torch.unique(indices).numel(), 512)
        self.assertGreaterEqual(indices.min().item(), 0)
        self.assertLess(indices.max().item(), 1024)
        self.assertEqual(sample_unique_env_indices(16, 0.0, "cpu").numel(), 0)
        with self.assertRaises(ValueError):
            sample_unique_env_indices(16, 1.01, "cpu")

    def test_max_local_self_dimension_matches_runtime_layout(self) -> None:
        for config, expected_dim in ((self.config, 358), (self.g1_config, 463)):
            configured_dims = {
                key: int(value)
                for item in config.obs.obs_dims
                for key, value in item.items()
            }
            num_extended_bodies = int(config.robot.num_bodies + config.robot.motion.nums_extend_bodies)
            runtime_dim = 1 + (num_extended_bodies - 1) * 3 + num_extended_bodies * (6 + 3 + 3)

            self.assertEqual(configured_dims["max_local_self"], expected_dim)
            self.assertEqual(configured_dims["max_local_self"], runtime_dim)

    def test_g1_and_mini3_expert_angular_velocity_matches_policy_semantics(self) -> None:
        self.assertEqual(float(self.config.obs.obs_scales.base_ang_vel), 0.25)
        self.assertEqual(float(self.g1_config.obs.obs_scales.base_ang_vel), 0.25)

        half_sqrt_two = 2**-0.5
        base_quat = torch.tensor([[0.0, 0.0, half_sqrt_two, half_sqrt_two]])
        root_ang_vel_world = torch.tensor([[1.0, 0.0, 0.0]])
        state, components = build_reference_policy_state(
            ref_dof_pos=torch.ones(1, 2),
            ref_dof_vel=torch.ones(1, 2),
            projected_gravity=torch.tensor([[0.0, 0.0, -1.0]]),
            base_quat=base_quat,
            root_ang_vel_world=root_ang_vel_world,
            obs_scales={
                "dof_pos": 2.0,
                "dof_vel": 3.0,
                "projected_gravity": 4.0,
                "base_ang_vel": 0.25,
            },
        )

        torch.testing.assert_close(components["dof_pos"], torch.full((1, 2), 2.0))
        torch.testing.assert_close(components["dof_vel"], torch.full((1, 2), 3.0))
        torch.testing.assert_close(components["projected_gravity"], torch.tensor([[0.0, 0.0, -4.0]]))
        torch.testing.assert_close(components["base_ang_vel"], torch.tensor([[0.0, -0.25, 0.0]]), atol=1e-6, rtol=0)
        self.assertEqual(state.shape, (1, 10))

    def test_vector_env_reset_uses_two_value_reset_contract(self) -> None:
        class DummyBaseEnv:
            def __init__(self) -> None:
                self.target_states = None

            def reset(self, target_states=None):
                self.target_states = target_states
                return {"raw": torch.zeros(2, 1)}, {"reset_succeeded": True}

            def close(self):
                return None

        class DummyWrappedEnv:
            def __init__(self, base_env) -> None:
                self.unwrapped = base_env

        base_env = DummyBaseEnv()
        wrapper = HumanoidVerseVectorEnv.__new__(HumanoidVerseVectorEnv)
        wrapper._env = DummyWrappedEnv(base_env)
        wrapper.history_handler = None
        wrapper.num_envs = 2
        wrapper._get_robot_observation = lambda to_numpy=True: {"state": np.zeros((2, 48), dtype=np.float32)}
        wrapper._get_clean_policy_state = lambda to_numpy=True: np.zeros((2, 48), dtype=np.float32)
        wrapper._get_qpos_qvel = lambda to_numpy=True: (
            np.zeros((2, 28), dtype=np.float32),
            np.zeros((2, 27), dtype=np.float32),
        )

        observation, info = wrapper.reset()

        self.assertEqual(observation["state"].shape, (2, 48))
        self.assertTrue(info["reset_succeeded"])

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
