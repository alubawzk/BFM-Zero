# Mini3 训练数据对齐与实现问题审查

## 1. 审查结论

当前 Mini3 训练路径不存在判别器 expert/policy 输入的硬维度错位：

- policy `state`：48 维；
- expert `state`：48 维；
- policy `privileged_state`：358 维；
- expert `privileged_state`：358 维；
- policy/expert `z`：256 维；
- 判别器实际使用 `state + privileged_state`，即 406 维 observation；
- 判别器拼接 `z` 后的总输入维度为 662。

已经使用 Mini3 字段结构完成以下轻量验证：

1. expert 和 policy 分别通过判别器，输出均为 `[batch_size, 1]`；
2. expert 缺少 `history_actor`、policy 包含 `history_actor` 时，WGAN gradient penalty 可以正常计算；
3. 使用缩小后的网络配置成功执行了一次完整 `FBcprAuxAgent.update()`。

因此，当前主要问题不是 tensor shape 无法拼接，而是 expert 和 policy observation 的物理语义没有完全对齐。其中最重要的是根角速度的坐标系、缩放和噪声处理不一致，这可能为判别器提供错误的分类捷径。

本次修改已经解决其中的 observation noise 不一致：判别器现在使用无噪声的 policy `state`，actor、F/B 和 critic 仍使用原有带噪 observation。根角速度的坐标系与缩放不一致仍需单独修复。

## 2. 判别器输入维度核对

Mini3 是 21-DoF 机器人。`state` 由以下字段拼接：

```text
dof_pos             21
dof_vel             21
projected_gravity    3
base_ang_vel         3
----------------------
state               48
```

Policy observation 在 `HumanoidVerseVectorEnv._get_robot_observation()` 中构造：

```python
state = torch.cat(
    [
        raw_obs["dof_pos"],
        raw_obs["dof_vel"],
        raw_obs["projected_gravity"],
        raw_obs["base_ang_vel"],
    ],
    dim=-1,
)
```

Expert observation 在 `load_expert_trajectories_from_motion_lib()` 中使用相同字段顺序：

```python
state = torch.cat(
    [
        ref_dof_pos,
        ref_dof_vel,
        projected_gravity,
        ref_ang_vel,
    ],
    dim=-1,
)
```

判别器的 input filter 配置为：

```python
DictInputFilterConfig(
    key=["state", "privileged_state"],
)
```

各字段情况如下：

| 字段 | Policy | Expert | 判别器使用 |
|---|---:|---:|---|
| `state` | 48 | 48 | 是 |
| `privileged_state` | 358 | 358 | 是 |
| `last_action` | 21 | 21（全零） | 否 |
| `history_actor` | 276 | 不存在 | 否 |
| `z` | 256 | 256 | 是 |

由此得到：

```text
discriminator observation = 48 + 358 = 406
discriminator total input = 406 + 256 = 662
```

Expert 不含 `history_actor` 不会造成判别器报错，原因有两个：

1. 判别器 input filter 不读取 `history_actor`；
2. observation normalizer 设置了 `allow_mismatching_keys=True`，允许 expert 缺少 policy-only 字段。

WGAN gradient penalty 遍历 expert observation 中已有的字段。Policy 包含 expert 的 `state`、`privileged_state` 和 `last_action`，因此插值时不存在缺少字段的问题。`last_action` 不被判别器使用，对它的梯度为 `None`，随后会被过滤。

## 3. 高风险问题：expert/policy 根角速度语义不一致

### 3.1 Policy 使用局部坐标系

Policy 的 `base_ang_vel` 使用 base quaternion 将世界坐标角速度旋转到机体局部坐标系：

```python
self.base_ang_vel[:] = quat_rotate_inverse(
    self.base_quat,
    self.simulator.robot_root_states[:, 10:13],
    w_last=True,
)
```

位置：

```text
humanoidverse/envs/legged_base_task/legged_robot_base.py
```

### 3.2 Policy 还会应用 observation scale 和 noise

环境 observation 的统一处理为：

```python
buf_dict[obs_key] = (
    actor_obs
    + (torch.rand_like(actor_obs) * 2.0 - 1.0) * obs_noise
) * obs_scale
```

Mini3 使用的 observation 配置包含：

```yaml
obs_scales:
  base_ang_vel: 0.25

noise_scales:
  base_ang_vel: 0.2
  projected_gravity: 0.05
  dof_pos: 0.01
  dof_vel: 0.5
```

所以 actor 及其他策略网络看到的 policy `base_ang_vel` 是：

```text
机体局部坐标角速度 + noise，然后乘以 0.25
```

修改后，判别器使用单独生成的 `clean_policy_obs`。该 observation 仍应用相同的 observation scale，但不添加随机噪声。因此判别器看到的是：

```text
机体局部坐标角速度 × 0.25，无 observation noise
```

### 3.3 Expert 使用全局角速度，且未缩放

Expert observation 当前直接使用 motion library 给出的全局角速度：

```python
ref_ang_vel = ref_body_angular_vels[:, 0]
```

motion library 中的 `body_ang_vel_t` 来源于 `global_angular_velocity`。这里没有执行 `quat_rotate_inverse()`，也没有应用 `obs_scales.base_ang_vel=0.25`。

所以判别器看到的 expert `base_ang_vel` 是：

```text
世界坐标角速度，无 observation noise，缩放系数为 1.0
```

### 3.4 影响

当前两侧实际为：

```text
Policy discriminator input: local angular velocity × 0.25，不带噪声
Expert: global angular velocity × 1.0，不带噪声
```

这不会引发维度报错，但可能产生以下问题：

- 判别器通过角速度尺度直接识别 expert/policy；
- 判别器通过坐标系差异直接识别 expert/policy；
- discriminator reward 不再只反映动作是否接近专家分布；
- expert trajectory 经 backward encoder 得到的 latent 可能与 policy observation 分布不一致；
- tracking evaluation 中构造目标 latent 时也会继承相同问题。

BatchNorm 只能基于 policy running statistics 做数值归一化，不能修复全局/局部坐标系不同的问题。

### 3.5 建议修复

至少应将 expert 根角速度转换为与 policy 相同的局部坐标：

```python
ref_ang_vel = quat_rotate_inverse(
    base_quat,
    ref_body_angular_vels[:, 0],
    w_last=True,
)
```

随后按与 policy 相同的 observation scale 处理：

```python
ref_ang_vel = ref_ang_vel * env.config.obs.obs_scales.base_ang_vel
```

更稳妥的实现方式是抽出共享 observation 构造函数，让 policy、expert loader 和 tracking evaluation 使用同一套：

- 坐标系变换；
- default pose subtraction；
- observation scale；
- 字段顺序；
- dtype；
- feature dimension 检查。

对于 observation noise，需要明确训练意图：

- 当前已经采用“判别器比较无噪声物理状态”的方案，同时向判别器提供无噪声 policy/expert observation；
- 如果希望判别器在噪声下训练，应向 expert 应用同分布噪声；
- 不建议只给 policy 添加噪声，否则判别器可能学习数据来源而不是行为质量。

### 3.6 已实现的 clean policy observation 数据流

环境配置新增 `clean_policy_obs`，并通过 `no_noise_obs_keys` 禁用以下字段的随机 observation noise：

```text
base_ang_vel
projected_gravity
dof_pos
dof_vel
max_local_self
```

为了控制 replay buffer 显存，只额外保存判别器需要替换的 48 维 `clean_state`。`privileged_state` 原本的 noise scale 就是 0，因此不重复存储。

agent update 中执行以下逻辑：

```text
带噪 train_obs ───────────────→ actor / F / B / critic
clean_state + privileged_state → discriminator 与 discriminator reward
```

`clean_state` 使用与普通 `state` 相同的 BatchNorm running statistics 进行归一化。加载没有 `clean_state` 的旧 replay buffer 时，训练会保留 agent checkpoint，但在内存中创建一个新的空 replay buffer，确保判别器不会回退到带噪 policy observation。

## 4. 中风险问题：DOF position 的 reference 不一致

Policy 的 `dof_pos` 为：

```python
self.simulator.dof_pos - (
    self.default_dof_pos
    + self.default_dof_pos_offset
)
```

其中 `default_dof_pos_offset` 会在 domain randomization 时按环境采样：

```yaml
randomize_default_dof_pos: true
default_dof_pos_noise_range: [-0.02, 0.02]
```

Expert 则只减去 nominal default pose：

```python
ref_dof_pos = motion_res["dof_pos"] - env.default_dof_pos[0]
```

因此 expert/policy 的 `dof_pos` 最多可能存在约 `±0.02 rad` 的系统偏移。该差异不大，但仍可能成为判别器的数据来源特征。

建议为判别器单独定义统一 reference convention，或者在构造 expert batch 时采样与 policy 同分布的 default pose offset。

## 5. 中风险问题：`max_local_self` 静态维度少算一维

当前配置为：

```yaml
max_local_self: ${eval:'(3 + 6 + 3 + 3) * (${robot.num_bodies} + ${robot.motion.nums_extend_bodies}) + 1 - 3 - 1'}
```

Mini3 有 24 个 body、0 个 extend body，并启用了 root height。真实维度为：

```text
root height                1
local body position       (24 - 1) × 3 = 69
local body rotation        24 × 6     = 144
local body velocity        24 × 3     = 72
local angular velocity     24 × 3     = 72
------------------------------------------
total                                  358
```

当前公式计算结果是 357。正确形式应去掉最后的 `- 1`：

```yaml
max_local_self: ${eval:'(3 + 6 + 3 + 3) * (${robot.num_bodies} + ${robot.motion.nums_extend_bodies}) + 1 - 3'}
```

当前训练没有立刻因此失败，是因为 `HumanoidVerseVectorEnv` 根据实际 tensor 动态构造了 observation space，模型最终得到的是 358 维。

但是底层 `BaseTask` 根据静态配置声明的 Gym observation space 仍然少一维，可能影响：

- 直接使用底层环境的调用者；
- 依赖 `single_observation_space` 的其他算法；
- 后续测试或导出工具；
- 新增严格 shape validation 后的兼容性。

## 6. 中风险问题：`aux_critic` 配置被忽略

配置模型定义了独立字段：

```python
class FBcprAuxModelArchiConfig(FBcprModelArchiConfig):
    aux_critic: ForwardArchiConfig = ForwardArchiConfig()
```

但实际创建 auxiliary critic 时使用的是普通 critic 配置：

```python
self._aux_critic = cfg.archi.critic.build(
    obs_space,
    cfg.archi.z_dim,
    action_dim,
    output_dim=1,
)
```

这里应当使用：

```python
cfg.archi.aux_critic.build(...)
```

`update_aux_critic()` 中的：

```python
num_parallel = self.cfg.model.archi.critic.num_parallel
```

也应从 `archi.aux_critic` 获取。

当前 Mini3 的 `critic` 和 `aux_critic` 配置完全相同，因此暂时不会改变实际网络结构。但一旦未来单独修改 auxiliary critic，该配置会被静默忽略。

## 7. 显存风险

完整训练配置目前使用：

```text
buffer_size   = 5,120,000
buffer_device = cuda
```

Mini3 每条 transition 会保存：

- 703 维 observation；
- 21 维 action；
- 256 维 latent `z`；
- auxiliary rewards；
- reward/termination/step count；
- qpos/qvel 等信息。

仅 replay buffer 粗略估计就可能超过 20 GB，此外还需要：

- 2048 hidden dimension 的多个网络及 target networks；
- optimizer states；
- expert buffer；
- 1024 个 Isaac Sim environments；
- simulation tensors 和临时 activation。

在普通 24 GB RTX 4090 上，完整默认配置有较高 OOM 风险。建议根据实际显存考虑：

- 将 `buffer_device` 改为 `cpu`；
- 减小 `buffer_size`；
- 减小 `online_parallel_envs`；
- 减小 batch size 或网络宽度；
- 在正式训练前运行包含至少一次 agent update 的 smoke test。

## 8. 当前已确认正常的部分

以下 Mini3 路径目前未发现硬维度问题：

- motion 文件的 `dof_pos` 是 21 维；
- environment action dimension 是 21；
- policy/expert `state` 均为 48 维；
- policy/expert `privileged_state` 均为 358 维；
- discriminator input filter 只读取 `state` 和 `privileged_state`；
- expert 缺少 `history_actor` 不影响 discriminator；
- batch size 1024 可以被 sequence length 8 整除；
- expert motion 长度满足 sequence sampling；
- auxiliary reward 名称均存在于当前 reward 配置；
- WGAN gradient penalty 可以处理当前 expert/policy 字段差异；
- 缩小配置下完整 agent update 可以执行。

## 9. 建议增加的运行时检查

目前 expert loader 只检查时间维长度，没有明确检查 feature dimension。建议在 expert buffer 构造后增加：

```python
expected_state_dim = 2 * env.dim_actions + 6
assert expert_state.shape[-1] == expected_state_dim
assert expert_state.shape[-1] == policy_obs_space["state"].shape[-1]
assert expert_privileged.shape[-1] == policy_obs_space["privileged_state"].shape[-1]
```

在 `update_discriminator()` 前增加 discriminator 实际字段检查：

```python
for key in ("state", "privileged_state"):
    assert expert_obs[key].shape[1:] == train_obs[key].shape[1:]
    assert expert_obs[key].dtype == train_obs[key].dtype
```

还建议定期记录以下统计量：

```text
expert/state/base_ang_vel mean/std
policy/state/base_ang_vel mean/std
expert/state/dof_pos mean/std
policy/state/dof_pos mean/std
discriminator expert logits mean/std
discriminator policy logits mean/std
```

如果训练开始后判别器在 policy 尚未明显改善时就迅速将 expert/policy 完全分开，应优先检查 observation 语义、scale、noise 和 domain-randomization 差异。

## 10. 推荐修复顺序

1. 对齐 expert/policy `base_ang_vel` 的坐标系和 scale；
2. 已完成：判别器使用无噪声 policy/expert observation；
3. 对齐 `default_dof_pos_offset` 的 reference convention；
4. 修复 `max_local_self` 的 357/358 静态维度；
5. 修复 `aux_critic` 配置未生效的问题；
6. 增加 feature dimension、dtype 和统计分布检查；
7. 使用小规模配置跑过初始化、首次 evaluation、buffer sampling 和至少一次完整 agent update；
8. 最后再启动 384M steps 的完整训练。

## 11. 涉及的主要文件

```text
humanoidverse/train.py
humanoidverse/agents/envs/humanoidverse_isaac.py
humanoidverse/agents/fb_cpr/agent.py
humanoidverse/agents/fb_cpr_aux/agent.py
humanoidverse/agents/fb_cpr_aux/model.py
humanoidverse/agents/nn_models.py
humanoidverse/agents/normalizers.py
humanoidverse/envs/legged_base_task/legged_robot_base.py
humanoidverse/envs/legged_robot_motions/legged_robot_motions.py
humanoidverse/config/obs/bfm_zero_obs.yaml
humanoidverse/config/domain_rand/domain_rand.yaml
humanoidverse/config/robot/mini3/mini3_21dof.yaml
```

## 12. 审查范围说明

本次审查包括：

- 静态数据流和配置检查；
- Mini3 motion 文件字段检查；
- discriminator forward 合成测试；
- WGAN gradient penalty 合成测试；
- 缩小网络下的完整 `FBcprAuxAgent.update()` 合成测试。

由于当前审查环境无法访问 CUDA/Isaac Sim，没有在本地完成完整 Isaac Sim end-to-end 训练。已有服务器日志能够证明 Mini3 环境、77 条 motion、1024 个并行环境和首次 evaluation 可以启动，但这不能替代长时间训练稳定性与学习效果验证。
