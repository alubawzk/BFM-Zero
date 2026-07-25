from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Mapping, Sequence


def _get(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def _require_unique(names: Sequence[str], label: str) -> None:
    duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
    if duplicates:
        raise ValueError(f"{label} contains duplicate names: {duplicates}")


def indices_by_name(source_names: Iterable[str], target_names: Iterable[str]) -> list[int]:
    """Return target indices in source order, rejecting ambiguous name mappings."""
    source = list(source_names)
    target = list(target_names)
    _require_unique(source, "source_names")
    _require_unique(target, "target_names")

    source_index = {name: index for index, name in enumerate(source)}
    missing = [name for name in target if name not in source_index]
    if missing:
        raise ValueError(f"Names are missing from source_names: {missing}")
    return [source_index[name] for name in target]


def _validate_index_group(config: Any, fields: Sequence[str], size: int, label: str) -> None:
    indices = [int(index) for field in fields for index in _get(config, field, [])]
    duplicates = sorted(index for index, count in Counter(indices).items() if count > 1)
    out_of_range = sorted(index for index in indices if index < 0 or index >= size)
    if duplicates or out_of_range:
        raise ValueError(f"{label} invalid; duplicates={duplicates}, out_of_range={out_of_range}, size={size}")


def validate_robot_config(robot_config: Any) -> dict[str, int | str]:
    """Validate the robot/action/body schema before simulator construction."""
    robot_name = str(_get(robot_config, "name", _get(_get(robot_config, "asset"), "robot_type", "unknown")))
    dof_names = list(_get(robot_config, "dof_names", []))
    body_names = list(_get(robot_config, "body_names", []))
    _require_unique(dof_names, f"{robot_name}.dof_names")
    _require_unique(body_names, f"{robot_name}.body_names")

    if not dof_names or not body_names:
        raise ValueError(f"{robot_name}: dof_names and body_names must not be empty")

    expected_dofs = len(dof_names)
    for field in ("num_dofs", "dof_obs_size", "actions_dim"):
        value = _get(robot_config, field)
        if value is not None and int(value) != expected_dofs:
            raise ValueError(f"{robot_name}: {field}={value}, expected {expected_dofs}")

    if int(_get(robot_config, "num_bodies")) != len(body_names):
        raise ValueError(
            f"{robot_name}: num_bodies={_get(robot_config, 'num_bodies')}, expected {len(body_names)} from body_names"
        )

    for field in (
        "dof_pos_lower_limit_list",
        "dof_pos_upper_limit_list",
        "dof_vel_limit_list",
        "dof_effort_limit_list",
        "dof_armature_list",
        "dof_joint_friction_list",
    ):
        values = list(_get(robot_config, field, []))
        if len(values) != expected_dofs:
            raise ValueError(f"{robot_name}: {field} has {len(values)} values, expected {expected_dofs}")

    init_state = _get(robot_config, "init_state")
    default_angles = dict(_get(init_state, "default_joint_angles", {}))
    if set(default_angles) != set(dof_names):
        missing = sorted(set(dof_names) - set(default_angles))
        extra = sorted(set(default_angles) - set(dof_names))
        raise ValueError(f"{robot_name}: default_joint_angles mismatch; missing={missing}, extra={extra}")

    semantic_body_fields = (
        "root_body_name",
        "pelvis_name",
        "torso_name",
        "left_foot_name",
        "right_foot_name",
    )
    for field in semantic_body_fields:
        value = _get(robot_config, field)
        if value is not None and value not in body_names:
            raise ValueError(f"{robot_name}: {field}={value!r} is not in body_names")

    strict_schema = bool(_get(robot_config, "strict_dof_groups", False))
    if strict_schema:
        lower = list(_get(robot_config, "lower_dof_names", []))
        upper = list(_get(robot_config, "upper_dof_names", []))
        lower_dim = int(_get(robot_config, "lower_body_actions_dim"))
        upper_dim = int(_get(robot_config, "upper_body_actions_dim"))
        if len(lower) != lower_dim:
            raise ValueError(f"{robot_name}: lower_dof_names must match lower_body_actions_dim")
        if len(upper) != upper_dim:
            raise ValueError(f"{robot_name}: upper_dof_names must match upper_body_actions_dim")
        _require_unique(lower + upper, f"{robot_name}.lower/upper_dof_names")
        if set(lower + upper) != set(dof_names):
            raise ValueError(f"{robot_name}: lower/upper DOF groups must cover every configured DOF exactly once")

        symmetry = _get(robot_config, "symmetric_dofs_idx")
        _validate_index_group(
            symmetry,
            (
                "left_dofs_idx_no",
                "left_dofs_idx_op",
                "right_dofs_idx_no",
                "right_dofs_idx_op",
                "waist_dofs_idx_no",
                "waist_dofs_idx_op",
            ),
            expected_dofs,
            f"{robot_name}.symmetric_dofs_idx",
        )
        _validate_index_group(
            symmetry,
            (
                "lower_left_dofs_idx_no",
                "lower_left_dofs_idx_op",
                "lower_right_dofs_idx_no",
                "lower_right_dofs_idx_op",
            ),
            lower_dim,
            f"{robot_name}.lower symmetric indices",
        )
        _validate_index_group(
            symmetry,
            (
                "upper_left_dofs_idx_no",
                "upper_left_dofs_idx_op",
                "upper_right_dofs_idx_no",
                "upper_right_dofs_idx_op",
            ),
            upper_dim,
            f"{robot_name}.upper symmetric indices",
        )

        control = _get(robot_config, "control")
        stiffness = dict(_get(control, "stiffness", {}))
        damping = dict(_get(control, "damping", {}))
        if set(stiffness) != set(damping):
            raise ValueError(f"{robot_name}: stiffness and damping patterns must have identical keys")
        for dof_name in dof_names:
            matching_patterns = [pattern for pattern in stiffness if pattern in dof_name]
            if len(matching_patterns) != 1:
                raise ValueError(
                    f"{robot_name}: {dof_name} must match exactly one stiffness/damping pattern; "
                    f"matches={matching_patterns}"
                )

    motion = _get(robot_config, "motion")
    if motion is not None:
        extend_config = list(_get(motion, "extend_config", []))
        configured_extensions = int(_get(motion, "nums_extend_bodies", len(extend_config)))
        if configured_extensions != len(extend_config):
            raise ValueError(
                f"{robot_name}: motion.nums_extend_bodies={configured_extensions}, expected {len(extend_config)}"
            )
        available_tracking_bodies = body_names + [str(_get(item, "joint_name")) for item in extend_config]
        tracking_body_names = list(_get(motion, "tracking_body_names", available_tracking_bodies))
        indices_by_name(available_tracking_bodies, tracking_body_names)
        for field in ("motion_tracking_link", "lower_body_link", "upper_body_link"):
            indices_by_name(available_tracking_bodies, list(_get(motion, field, [])))

        visualization = _get(motion, "visualization")
        if strict_schema and bool(_get(visualization, "customize_color", False)):
            marker_colors = list(_get(visualization, "marker_joint_colors", []))
            if len(marker_colors) != len(available_tracking_bodies):
                raise ValueError(
                    f"{robot_name}: marker_joint_colors has {len(marker_colors)} entries, "
                    f"expected {len(available_tracking_bodies)}"
                )

    return {
        "robot": robot_name,
        "num_dofs": expected_dofs,
        "num_bodies": len(body_names),
        "state_dim": 2 * expected_dofs + 6,
        "action_dim": expected_dofs,
    }
