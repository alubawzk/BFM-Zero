# BFM-Zero 集成 Mini3 训练完整方案

## 1. 目标与结论

目标是在当前 BFM-Zero/HumanoidVerse 工程中增加 Mini3（21 DoF）支持，完成以下闭环：

1. Mini3 机器人资产可在 MuJoCo 和 Isaac Sim 中正确加载。
2. 人体动作可重定向为 Mini3 reference motion。
3. BFM-Zero 可从头训练、评估并导出 21-DoF 策略。
4. 可选地支持 goal/reward inference、sim-to-sim 和 sim-to-real。

当前仓库已经包含：

- `humanoidverse/data/robots/mini3/urdf/mini3.urdf`
- `humanoidverse/data/robots/mini3/mjcf/mini3.xml`
- Mini3 mesh 与若干 MuJoCo scene

Mini3 MJCF 中存在 21 个执行关节：双腿 12、腰部 1、双臂 8。因此可以集成，但当前训练、评估、渲染和推理链路仍存在大量 G1/29-DoF 假设，不能仅替换 reference motion 后直接训练。

本文将改造分成两个范围：

- **最小训练闭环（P0）**：Isaac Sim 训练、tracking evaluation、checkpoint。
- **完整功能（P1/P2）**：MuJoCo sim-to-sim、ONNX、goal/reward inference、实机部署。

---

## 2. Mini3 统一机器人规格

所有模块必须共享同一份关节、刚体和语义定义。禁止在不同文件中各自维护下标。

### 2.1 关节顺序（21 DoF）

建议严格采用 Mini3 MJCF actuator 顺序：

```yaml
dof_names:
  - left_hip_pitch_joint
  - left_hip_roll_joint
  - left_hip_yaw_joint
  - left_knee_pitch_joint
  - left_ankle_pitch_joint
  - left_ankle_roll_joint
  - right_hip_pitch_joint
  - right_hip_roll_joint
  - right_hip_yaw_joint
  - right_knee_pitch_joint
  - right_ankle_pitch_joint
  - right_ankle_roll_joint
  - waist_yaw_joint
  - left_shoulder_pitch_joint
  - left_shoulder_roll_joint
  - left_shoulder_yaw_joint
  - left_elbow_pitch_joint
  - right_shoulder_pitch_joint
  - right_shoulder_roll_joint
  - right_shoulder_yaw_joint
  - right_elbow_pitch_joint
```

重要：G1 使用 `left_knee_joint`、`left_elbow_joint`，Mini3 使用 `left_knee_pitch_joint`、`left_elbow_pitch_joint`。所有按字符串匹配 stiffness、damping、reward body 和 motion link 的逻辑都要验证。

### 2.2 动作分组

```yaml
actions_dim: 21
dof_obs_size: 21
lower_body_actions_dim: 12
upper_body_actions_dim: 9  # waist 1 + arms 8

lower_dof_names:  # 双腿 12
upper_dof_names:  # waist + 双臂 9
waist_dof_names: [waist_yaw_joint]
arm_dof_names:    # 双臂 8
```

注意：当前 G1 配置中 `lower_dof_names` 实际还包含腰部，且 `lower_body_actions_dim` 又写为 12，语义并不完全一致。Mini3 接入时应明确：网络分支使用的分组究竟按腿/上肢，还是按 lower/upper link。配置写完后对每个使用点做 shape assertion。

### 2.3 根刚体与语义 body

Mini3 MJCF 根 body 为 `base_link`，没有 G1 的 `pelvis` 和 `torso_link`。建议统一配置：

```yaml
root_body_name: base_link
pelvis_name: base_link       # 兼容旧语义，最终应逐步移除 pelvis 命名
torso_name: waist_yaw_link
left_foot_name: left_ankle_roll_link
right_foot_name: right_ankle_roll_link
knee_name: knee_pitch_link
```

不要直接将 Mini3 MJCF body 改名成 G1 名称来绕开代码问题。正确做法是将这些名称配置化，因为后续 reward、传感器和部署均需要真实名称。

---

## 3. Reference motion 之外需要准备的资料

### 3.1 仿真资产

已有 URDF/MJCF，但还需要：

- Isaac Sim 可加载且经过验证的 USD，或验证项目的 URDF importer 路径。
- collision geometry；训练中尽量使用简单、闭合、稳定的碰撞体。
- 正确的 mass、COM、inertia 与 joint axis。
- 足底接触几何和摩擦参数。
- self-collision 过滤规则。
- floating base，且根坐标系方向满足项目约定（z-up，前向轴一致）。

验收要求：零控制时无模型爆炸；默认站姿下无明显穿模；左右脚接触稳定；关节正方向与 motion retarget 一致。

### 3.2 电机与控制参数

URDF 中的 effort/velocity 只能作为起点。实机团队应提供：

- 每个关节持续/峰值力矩。
- 最大关节速度。
- 减速比和转子惯量。
- 推荐 position-control PD。
- 实机控制周期、命令周期和状态回传周期。
- 命令延迟、观测延迟和抖动范围。
- 编码器零位、限位和安全软限位。

由此生成：`dof_effort_limit_list`、`dof_vel_limit_list`、`dof_armature_list`、PD、action scale 和 domain randomization 范围。

### 3.3 默认姿态与安全定义

准备并实测：

- 默认站姿 21-D joint position。
- 对应 base height。
- 蹲姿/恢复姿态（若启用 lie-down initialization）。
- 禁止接触 body、允许接触 body。
- 实机软限位、最大命令增量、急停条件。

### 3.4 Reference motion 数据契约

重定向后的每段 motion 至少应包含：

- `root_pos [T, 3]`
- `root_rot [T, 4]`
- `root_vel [T, 3]`
- `root_ang_vel [T, 3]`
- `dof_pos [T, 21]`
- `dof_vel [T, 21]`
- `rg_pos_t [T, B, 3]`
- `rg_rot_t [T, B, 4]`
- `body_vel_t [T, B, 3]`
- `body_ang_vel_t [T, B, 3]`
- FPS/dt、motion length、motion key/name

其中关节顺序必须与 `robot.dof_names` 完全一致，body 顺序必须与 motion config 完全一致。四元数顺序需要在数据入口明确记录是 `xyzw` 还是 `wxyz`。

建议增加数据校验脚本，检查：shape、NaN/Inf、四元数模长、速度差分一致性、关节限位、足底最低点、左右对称性，以及 FK 结果是否与保存的 rigid-body pose 一致。

---

## 4. 新增配置文件

### 4.1 `humanoidverse/config/robot/mini3/mini3_21dof.yaml`

以 `g1/g1_29dof_hard_waist.yaml` 为模板，但必须逐项替换，不能只覆盖维度。

必须包含：

- `num_bodies`：以 Isaac 加载并 collapse fixed joints 后的实际 body 数为准。
- `dof_obs_size/actions_dim = 21`。
- 完整 `dof_names/body_names`。
- upper/lower/arm/waist/ankle/knee 分组。
- 对称关节 index。索引必须基于上文统一的 21-D 顺序重新计算。
- joint position/velocity/effort/armature/friction 列表。
- default joint angles 与 base init height。
- PD stiffness/damping 和 action normalization。
- foot/contact/key bodies。
- randomize link body names。
- asset URI：URDF、USD、MJCF。
- motion asset、body mapping、joint matches、link groups。

推荐增加以下通用字段，供硬编码改造使用：

```yaml
robot:
  name: mini3
  num_dofs: 21
  root_body_name: base_link
  torso_name: waist_yaw_link
  imu_body_name: imu_link
  qpos_root_dim: 7
  qvel_root_dim: 6
  render_body_names: ${robot.body_names}
  tracking_body_names: [...]
```

### 4.2 新增实验配置

新增：

```text
humanoidverse/config/exp/bfm_zero/bfm_zero_mini3.yaml
```

建议从现有 `bfm_zero.yaml` 继承或复制，至少修改：

```yaml
defaults:
  - /robot: mini3/mini3_21dof

robot:
  motion:
    motion_file: data/mini3_reference_train.pkl
```

不要修改默认 G1 配置，以避免破坏已有复现实验。

### 4.3 训练入口

`humanoidverse/train.py` 当前在默认 `TrainConfig` 中写死：

- G1 work directory 名称。
- `lafan_29dof_10s-clipped.pkl`。
- `robot=g1/g1_29dof_hard_waist`。

改造方案：

1. 将 robot config、motion path 放入显式的 robot/profile 配置。
2. 新增 `train_bfm_zero_mini3()` 或让 CLI 接受 `relative_config_path` 与 Hydra overrides。
3. 禁止在通用默认值中出现 G1 motion path。
4. 启动时打印并断言 `num_dofs == action_dim == motion dof dim`。

---

## 5. 硬编码修改清单

本节按照优先级列出需要修改的代码。行号基于当前版本，修改后会漂移，应以符号名和搜索词定位。

### 5.1 P0：训练必须修改

#### A. `humanoidverse/simulator/isaacsim/isaacsim.py`

发现的硬编码：

- COM randomization body 写死为 `torso_link`。
- height scanner prim 写死为 `/Robot/pelvis`。
- 文件中存在固定 body reorder 数组及 G1/H1 body 列表。
- 某些 debug/visualization 路径可能写死 pelvis/torso。

建议修改：

```python
# before
body_names=["torso_link"]

# after
body_names=[self.robot_cfg.root_body_name]
```

```python
# before
prim_path="/World/envs/env_.*/Robot/pelvis"

# after
prim_path=f"/World/envs/env_.*/Robot/{self.robot_cfg.root_body_name}"
```

body reorder 不能按固定 index/list 分支，应在运行时通过 `robot_cfg.body_names` 和 Isaac articulation body names 构建：

```python
body_name_to_sim_id = {name: i for i, name in enumerate(sim_body_names)}
body_ids = [body_name_to_sim_id[name] for name in robot_cfg.body_names]
```

然后验证不存在缺失/重复名称。

#### B. `humanoidverse/simulator/isaacsim/isaaclab_cfg.py`

该文件包含 H1/G1 风格的静态：

- `body_names`
- `joint_names`
- `base_name = "torso_link"`
- feet/knee regex
- extend body parent IDs
- teleop keypoints
- height-scanner body

如果训练实际走通用 `RobotAssetCfg`，确认这些类未被 Mini3 路径使用；若会使用，则不要继续新增 `Mini3Cfg` 的复制粘贴分支，优先让 articulation、body names、base name、feet name 从 Hydra robot config 构造。

尤其 `extend_body_parent_ids` 必须从 parent body name 动态转换，不能保存随模型 body reorder 改变的整数 ID。

#### C. `humanoidverse/agents/envs/humanoidverse_isaac.py`

训练 wrapper 本身的 observation 拼接是按 tensor shape 工作的，但命名与可选逻辑仍是 G1 化的。

必须修改：

- `_get_g1env_observation` 重命名为 `_get_robot_observation`。
- 所有调用点同步替换；可保留旧方法作为 deprecated alias，避免破坏 G1 checkpoint。
- `make_config_g1env_compatible` 改为通用的 `motion_profile` 或删除，由 robot motion config 自己提供 link/match 列表。
- 该开关内部写死的 pelvis、G1 knee/elbow/ankle 名称移入 robot config。

建议接口：

```python
def _get_robot_observation(self, to_numpy=True):
    raw_obs = self._env.obs_buf_dict_raw["actor_obs"]
    state = torch.cat([
        raw_obs["dof_pos"],
        raw_obs["dof_vel"],
        raw_obs["projected_gravity"],
        raw_obs["base_ang_vel"],
    ], dim=-1)
    ...
```

新增断言：

```python
assert raw_obs["dof_pos"].shape[-1] == self._env.config.robot.actions_dim
assert raw_obs["actions"].shape[-1] == self.single_action_space.shape[-1]
```

#### D. `humanoidverse/agents/evaluations/humanoidverse_isaac.py`

发现的硬编码：

- 全局 `xpos_bodies` 是 G1 body 名称。
- `process_body_data()` 使用固定切片和补零，将特定数据排列为 24 bodies。
- 函数和变量名均假设 G1。

修改方案：

1. 删除全局 `xpos_bodies`，使用 `env.config.robot.motion.tracking_body_names`。
2. 删除 `arr[:, 0:13]` 等固定切片。
3. 若需要从外部 body 数组重排，引入显式 `source_body_names`，按名称映射：

```python
source_index = {name: i for i, name in enumerate(source_body_names)}
indices = [source_index[name] for name in target_body_names]
target = source[:, indices]
```

4. 对源数据中不存在的虚拟点，只允许通过配置中的 `extend_config` 生成，不允许在 evaluation 中静默补零。

这是 tracking metric 正确性的关键；错误的 body 顺序可能不会报错，但会使 EMD/位姿误差完全失真。

#### E. observation 和网络 shape

`humanoidverse/config/obs/bfm_zero_obs.yaml` 已使用 `${robot.dof_obs_size}` 和 `${robot.num_bodies}`，原则上可支持 21 DoF。但需要验证：

- `num_bodies` 是 simulator 实际参与 observation 的 body 数，而不是 URDF link 总数。
- `nums_extend_bodies` 拼写与实际代码读取一致（代码中还出现过 `num_extend_bodies`）。
- history actor 中 actions/dof pos/dof vel 都变为 21-D。
- agent build 后 actor 输出为 21-D。

启动时记录最终维度：

```text
state_dim = 2 * num_dofs + 6 = 48  (Mini3)
last_action_dim = 21
privileged_state_dim = max_local_self 的实际计算结果
```

注意：若启用其他字段，不能只依赖上述公式，应以 observation space 为真值。

### 5.2 P1：MuJoCo 与可视化必须修改

#### F. `humanoidverse/simulator/mujoco/mujoco.py`

当前最严重的问题是：先从配置计算 `model_path`，随后又强制覆盖为 G1 scene。

应删除覆盖：

```python
# 删除
hv_root = Path(__file__).parents[2]
self.model_path = str(hv_root / "data/robots/g1/scene_29dof_freebase_mujoco.xml")
```

只保留：

```python
self.model_path = os.path.join(
    self.robot_cfg.asset.asset_root,
    self.robot_cfg.asset.xml_file,
)
```

其他修改：

- `self.model.body("torso_link")` 改为配置的 COM randomization body。
- 删除通过文件名是否包含 `"23"`/`"29"` 来过滤 hand/wrist 的逻辑。
- 新增配置字段 `ignored_body_names`；按名称过滤。
- 保留并强化 DOF/body name 一致性 assertion。
- 检查 Mini3 scene 是否有 floor、freejoint、actuator、传感器；不要把仅 robot 的 `mini3.xml` 当完整 scene 使用。

#### G. `IsaacRendererWithMuJoco`

位置：`humanoidverse/agents/envs/humanoidverse_isaac.py`。

当前写死 29-D joint、36-D qpos，并构造 `G1EnvConfig`。

改造成：

```python
expected_qpos_dim = self.mujoco_env.unwrapped._mj_model.nq
expected_joint_dim = expected_qpos_dim - self.robot_cfg.qpos_root_dim
```

renderer 构造函数应接收 robot-specific MuJoCo config/model path，不能 import `G1EnvConfig`。验证 qpos 时使用模型 `nq`，错误信息输出 robot name、expected/actual shape。

Mini3 floating-base qpos 应为 `7 + 21 = 28`；qvel 应为 `6 + 21 = 27`。

#### H. Genesis debug

`humanoidverse/simulator/genesis/genesis_mjdebug.py` 写死：

- G1 XML 路径。
- 36-D qpos。
- 后 29 个关节。
- `range(29)`。

若 Mini3 不使用 Genesis，可标记为暂不支持并在选择 Genesis+Mini3 时明确报错；若要支持，则全部改为 `model.nq/model.nv/robot_cfg.actions_dim` 和 name mapping。不要把它列为 P0。

### 5.3 P1：ONNX 导出必须修改

#### I. `humanoidverse/utils/helpers.py::export_meta_policy_as_onnx`

当前参数 `use_29dof` 只支持两个固定布局：

- state end 64、action 29。
- state end 52、action 23。

Mini3 的基础 state 为 `21 + 21 + 3 + 3 = 48`，因此当前导出一定会错误切片。

推荐彻底删除 `use_29dof`，改成显式 layout：

```python
def export_meta_policy_as_onnx(
    ...,
    state_dim: int,
    action_dim: int,
):
    state_end = state_dim
    action_end = state_end + action_dim
```

更稳妥的是从 `example_obs_dict`/observation space 生成 flatten schema，并将 schema 一并保存为 JSON，避免部署端猜切片。

ONNX 验收：PyTorch 与 ONNX Runtime 对同一批 observation 的 action 最大绝对误差在设定容差内。

### 5.4 P1：tracking inference 必须修改

#### J. `humanoidverse/tracking_inference.py`

当前写死：

- 默认 `lafan_29dof.pkl`。
- config 内 `lafan_tail_path` 回退值。
- `use_29dof=True`。
- 调用 `_get_g1env_observation`。

改造方案：

- `data_path` 无默认 G1 路径；从 checkpoint `config.json` 读取，CLI 可覆盖。
- 从 checkpoint/robot config 获取 `state_dim/action_dim`。
- 调用通用 `_get_robot_observation`。
- renderer 使用 checkpoint 对应的 robot MJCF。
- 输出 metadata：robot、DOF names、obs schema、control dt、default pose、action scale。

### 5.5 P2：goal/reward inference 修改

#### K. `humanoidverse/goal_inference.py`

当前写死：G1 LaFan path、`use_29dof=True`、`goal_frames_lafan29dof.json` 及 G1 目录。

修改：

- goal frame 文件作为 CLI/config 参数。
- goal JSON 中记录 motion dataset hash/version，避免 frame index 对错数据。
- ONNX layout 从 robot config 获取。
- Mini3 单独生成 `goal_frames_mini3.json`。

#### L. `humanoidverse/reward_inference.py`

当前依赖 `g1_env_helper`，并写死 G1 scene XML。完整支持 Mini3 有两条路线：

1. **推荐**：把 `g1_env_helper` 抽象为通用 MuJoCo humanoid env，名称全部配置化。
2. 短期：复制成 `mini3_env_helper`，快速跑通但会产生两套重复 reward 代码。

推荐路线的接口至少包括：

```python
RobotSemanticConfig(
    root_body,
    torso_body,
    head_body,
    left_knee_body,
    right_knee_body,
    left_foot_body,
    right_foot_body,
    imu_site,
    tracking_bodies,
    stand_height,
)
```

`g1_env_helper/rewards.py` 中所有 `pelvis`、`torso_link`、`left_knee_link`、G1 站高常量均通过该结构访问。站高和阈值必须按 Mini3 尺寸重新标定，不能机械替换名称。

#### M. `humanoidverse/utils/g1_env_config.py` 与 `envs/g1_env_helper/*`

这些模块是 G1 专用实现，包含：

- G1 XML root。
- 29-D velocity/effort limits。
- G1 env class。
- 36-D qpos validation。
- torso/pelvis/IMU site 名称。
- tracking body list。

最小训练闭环不需要先改完它们；只有启用 reward inference 或独立 MuJoCo benchmark 时才进入 P2。抽象时建议新增通用模块，而不是让 `G1EnvConfig` 接受 Mini3 后仍保留误导命名。

### 5.6 低优先级/条件修改

#### N. `humanoidverse/envs/gymnasium_wrapper.py`

`_get_g1env_observation` 只是在命名上写死，数据拼接可能是动态的。改名并复用统一 observation builder，避免两个 wrapper 逻辑漂移。

#### O. Isaac Gym 文件

`humanoidverse/simulator/isaacgym/isaacgym.py` 和 `isaacgym_hoi.py` 中 pelvis/torso 查找写死。若项目只使用 Isaac Sim，可暂不修改；如果承诺 Isaac Gym 支持，则使用 `robot.root_body_name/torso_name`。

#### P. `humanoidverse/simulator/isaacsim/isaacsim.py` 中疑似旧 body reorder

文件尾部存在固定整数数组和 H1/G1 body 名列表。应确认是否仅为注释/debug；若是运行路径，必须改为按名称生成。任何固定 body index 都应视作高风险硬编码。

---

## 6. 建议的通用化设计

为了避免每加一个机器人就重复修改，建议引入以下配置驱动结构。

### 6.1 Robot kinematics schema

```python
@dataclass
class RobotKinematicsSchema:
    name: str
    dof_names: list[str]
    body_names: list[str]
    root_body_name: str
    torso_body_name: str
    foot_body_names: list[str]
    knee_body_names: list[str]
    tracking_body_names: list[str]
    ignored_body_names: list[str]
    qpos_root_dim: int = 7
    qvel_root_dim: int = 6
```

### 6.2 Observation schema

训练保存 checkpoint 时，同时保存：

```json
{
  "robot": "mini3",
  "dof_names": ["..."],
  "observation_layout": {
    "state": 48,
    "last_action": 21,
    "history_actor": "resolved at runtime",
    "privileged_state": "resolved at runtime"
  },
  "action_dim": 21,
  "control_dt": 0.02,
  "quaternion_order": "xyzw"
}
```

ONNX 和部署程序读取 schema，不再通过 `use_29dof` 判断。

### 6.3 Name mapping helper

统一提供：

```python
def indices_by_name(source_names, target_names):
    ...
```

用于 simulator body reorder、motion body reorder、evaluation tracking bodies 和 actuator reorder。函数必须对缺失、重复名称报错。

---

## 7. 实施阶段与验收标准

### Phase 0：资产审计

任务：

- 列出 URDF/MJCF 的 21 joints、body、actuator 顺序。
- 比较 URDF 与 MJCF 的 axis/range/mass/inertia。
- 明确 fixed joints collapse 后的 body 列表。
- 生成或导入 Isaac USD。

验收：MuJoCo/Isaac 中同一 21-D pose 外观和关节方向一致。

### Phase 1：Robot config 和静态检查

任务：新增 `mini3_21dof.yaml` 和 config validator。

验收：

- DOF 数、名称、顺序与 simulator 一致。
- 每个 limit/PD/default 列表长度都是 21。
- 左右对称 index 不越界且不重复。
- body 名称全部能在 simulator 中解析。

### Phase 2：Reference motion

任务：retarget、保存、校验、可视化。

验收：

- FK 与 rigid-body pose 一致。
- 无超限/NaN/四元数异常。
- 脚底高度和 contact 合理。
- 随机抽样至少 10 段逐帧可视化通过。

### Phase 3：最小训练 smoke test

配置：1～16 env、短 motion、关闭复杂 DR、运行数千步。

验收：

- reset/step/backprop/checkpoint 全部通过。
- actor action shape `[N, 21]`。
- loss、reward、observation 无 NaN。
- motion sampling 和 episode termination 正常。

### Phase 4：Tracking evaluation

验收：

- body metric 按名称对齐。
- reference 和 robot 可视化一致。
- metric 随策略改进呈合理趋势。

### Phase 5：规模训练

逐步增加 env 数和 DR。先学习稳定站立/简单动作，再扩展全数据集。Mini3 尺寸、力矩和上肢 DoF 都不同，不应沿用 G1 reward scale 与 termination threshold 而不做消融。

### Phase 6：ONNX 与 sim-to-sim

验收：

- PyTorch/ONNX 输出一致。
- MuJoCo rollout 不发生关节顺序错位。
- observation、default pose、action scale、control dt 与训练一致。

### Phase 7：实机

先吊装/低力矩测试，再逐步开放动作范围。部署端必须检查 checkpoint metadata 中的 DOF names 与实机控制器顺序完全一致。

---

## 8. 测试建议

建议新增：

```text
tests/test_robot_config.py
tests/test_mini3_asset_names.py
tests/test_motion_schema.py
tests/test_body_name_mapping.py
tests/test_observation_shapes.py
tests/test_onnx_parity.py
```

关键测试：

1. `len(dof_names) == actions_dim == dof_obs_size == 21`。
2. simulator joint names 与 config 完全相等。
3. motion `dof_pos.shape[-1] == 21`。
4. `qpos = 7 + 21`、`qvel = 6 + 21`。
5. actor 输出最后一维为 21。
6. body mapping 对缺失/重复名称必定抛错。
7. ONNX parity。
8. G1 原有配置仍可完成 smoke test，防止通用化造成回归。

---

## 9. 风险与易错点

- **关节名称近似但不相同**：`knee_joint` 与 `knee_pitch_joint` 会影响 PD regex 和 reward body 查找。
- **关节顺序错误不一定报错**：策略会输出合法 shape，但控制到错误电机。
- **body 数量受 fixed-joint collapse 影响**：不要直接用 URDF link 数填写 `num_bodies`。
- **四元数顺序混用**：当前代码同时接触 MuJoCo `wxyz` 和内部 `xyzw`。
- **extend body 配置拼写不一致**：`nums_extend_bodies` 与 `num_extend_bodies` 需要统一。
- **只改训练不改导出**：训练可以成功，但 `use_29dof` 会导致 ONNX 输入切片错误。
- **沿用 G1 站高/reward 阈值**：Mini3 更小，reward inference 会得到系统性错误奖励。
- **复用 G1 checkpoint**：21-D 与 29-D 输入输出层不兼容。默认从头训练；迁移需要显式的 weight mapping/蒸馏方案。
- **MuJoCo scene 与 robot XML 混淆**：scene 需要 world、floor、light 等；robot XML 可能仅定义机器人。

---

## 10. 最小改动文件集合

若目标仅为“Isaac Sim 上完成 BFM-Zero Mini3 训练 + tracking eval”，预计至少新增/修改：

```text
# 新增
humanoidverse/config/robot/mini3/mini3_21dof.yaml
humanoidverse/config/exp/bfm_zero/bfm_zero_mini3.yaml
humanoidverse/data/mini3_reference_train.pkl
humanoidverse/data/mini3_reference_eval.pkl

# 修改
humanoidverse/train.py
humanoidverse/simulator/isaacsim/isaacsim.py
humanoidverse/simulator/isaacsim/isaaclab_cfg.py        # 若实际加载路径使用
humanoidverse/agents/envs/humanoidverse_isaac.py
humanoidverse/agents/evaluations/humanoidverse_isaac.py
```

若还要完整推理/导出/sim-to-sim，再增加：

```text
humanoidverse/simulator/mujoco/mujoco.py
humanoidverse/utils/helpers.py
humanoidverse/tracking_inference.py
humanoidverse/goal_inference.py
humanoidverse/reward_inference.py
humanoidverse/utils/g1_env_config.py                    # 抽象/替代
humanoidverse/envs/g1_env_helper/**                     # 抽象/替代
```

---

## 11. 推荐执行顺序

1. 冻结 Mini3 21-D DOF 顺序和 body semantic schema。
2. 校准 URDF/MJCF，并生成/验证 Isaac USD。
3. 编写 Mini3 robot config 和静态 validator。
4. 完成 motion retarget 与数据校验。
5. 修改 Isaac Sim 中 root body、COM、height scanner 和 body reorder 硬编码。
6. 通用化 HumanoidVerse wrapper 与 tracking evaluation。
7. 完成小规模训练 smoke test。
8. 跑 tracking evaluation，确认 metric/body mapping 正确。
9. 扩大训练规模并重新标定 reward/DR。
10. 修改 ONNX layout 和 MuJoCo backend。
11. 最后抽象 goal/reward inference 和 G1 helper。

该顺序可以最早获得可训练闭环，同时避免 reward inference 的大规模通用化阻塞核心训练。
