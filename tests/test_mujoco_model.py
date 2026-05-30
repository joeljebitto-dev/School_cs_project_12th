import numpy as np
import pytest

from robot_math import forward_kinematics
from mujoco_model import load_model


def test_mujoco_model_loads_and_steps():
    mujoco = pytest.importorskip("mujoco")

    model = load_model()
    data = mujoco.MjData(model)

    assert model.nq == 3
    assert model.nu == 3

    data.ctrl[:] = np.array([0.1, -0.1, 0.05])
    for _ in range(5):
        mujoco.mj_step(model, data)

    assert np.all(np.isfinite(data.qpos[:3]))


def test_mujoco_end_effector_matches_forward_kinematics():
    mujoco = pytest.importorskip("mujoco")

    model = load_model()
    data = mujoco.MjData(model)
    joint_angles = np.array([0.35, 0.45, -0.60])

    data.qpos[:3] = joint_angles
    mujoco.mj_forward(model, data)

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")
    assert site_id >= 0

    expected_position = forward_kinematics(joint_angles)
    simulated_position = data.site_xpos[site_id].copy()

    assert np.allclose(simulated_position, expected_position, atol=1e-6)
