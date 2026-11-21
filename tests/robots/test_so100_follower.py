#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from lerobot.robots.so_follower import (
    SO100Follower,
    SO100FollowerConfig,
)

pytestmark = pytest.mark.hardware


def _make_bus_mock() -> MagicMock:
    """Return a bus mock with just the attributes used by the robot."""
    bus = MagicMock(name="FeetechBusMock")
    bus.is_connected = False

    def _connect():
        bus.is_connected = True

    def _disconnect(_disable=True):
        bus.is_connected = False

    bus.connect.side_effect = _connect
    bus.disconnect.side_effect = _disconnect

    @contextmanager
    def _dummy_cm():
        yield

    bus.torque_disabled.side_effect = _dummy_cm

    return bus


@pytest.fixture(params=[True, False], ids=["gripper", "rigid_tool"])
def follower(tmp_path, request):
    bus_mock = _make_bus_mock()

    def _bus_side_effect(*_args, **kwargs):
        bus_mock.motors = kwargs["motors"]
        motors_order: list[str] = list(bus_mock.motors)

        bus_mock.sync_read.return_value = {motor: idx for idx, motor in enumerate(motors_order, 1)}
        bus_mock.sync_write.return_value = None
        bus_mock.write.return_value = None
        bus_mock.disable_torque.return_value = None
        bus_mock.enable_torque.return_value = None
        bus_mock.is_calibrated = True
        return bus_mock

    with patch(
        "lerobot.robots.so_follower.so_follower.SerialMotorsBus",
        side_effect=_bus_side_effect,
    ):
        cfg = SO100FollowerConfig(port="/dev/null", calibration_dir=tmp_path, use_gripper=request.param)
        robot = SO100Follower(cfg)
        yield robot
        if robot.is_connected:
            robot.disconnect()


def test_connect_disconnect(follower):
    assert not follower.is_connected

    follower.connect()
    assert follower.is_connected

    follower.disconnect()
    assert not follower.is_connected


def test_get_observation(follower):
    follower.connect()
    obs = follower.get_observation()

    expected_keys = {f"{m}.pos" for m in follower.bus.motors}
    assert set(obs.keys()) == expected_keys

    for idx, motor in enumerate(follower.bus.motors, 1):
        assert obs[f"{motor}.pos"] == idx

    assert ("gripper.pos" in obs) == follower.config.use_gripper
    assert set(obs) == set(follower.action_features) == set(follower.observation_features)
    assert [m.id for m in follower.bus.motors.values()] == list(
        range(1, 7 if follower.config.use_gripper else 6)
    )


def test_observation_keys_match_features_with_mock_camera(follower):
    # `observation_features` and `get_observation()` must use the same depth predicate:
    # a bare mock camera is not a `DepthCamera`, so neither side may advertise `<cam>_depth`.
    cam = MagicMock(name="CameraMock")
    cam.height, cam.width = 480, 640
    follower.cameras = {"wrist": cam}
    follower.connect()

    obs = follower.get_observation()
    assert set(obs) == set(follower.observation_features)
    assert "wrist" in obs and "wrist_depth" not in obs


def test_get_observation_uses_read_retries(follower):
    # Feetech buses can intermittently fail a sync_read; the follower should forward the configured
    # retry count so transient failures don't abort the control loop (see #3131).
    follower.config.num_read_retries = 7
    follower.connect()
    follower.get_observation()

    follower.bus.sync_read.assert_called_once_with("Present_Position", num_retry=7)


def test_send_action_uses_read_retries(follower):
    follower.config.max_relative_target = 10.0
    follower.config.num_read_retries = 7
    follower.connect()

    action = {f"{motor}.pos": value * 10 for value, motor in enumerate(follower.bus.motors, 1)}
    follower.send_action(action)

    follower.bus.sync_read.assert_called_once_with("Present_Position", num_retry=7)


def test_send_action(follower):
    follower.connect()

    action = {f"{m}.pos": i * 10 for i, m in enumerate(follower.bus.motors, 1)}
    returned = follower.send_action(action)

    assert returned == action

    goal_pos = {m: (i + 1) * 10 for i, m in enumerate(follower.bus.motors)}
    follower.bus.sync_write.assert_called_once_with("Goal_Position", goal_pos)


def test_absent_motor_rejected_before_bus_access(follower):
    follower.connect()
    follower.config.max_relative_target = 5.0
    motor = "absent" if follower.config.use_gripper else "gripper"
    with pytest.raises(ValueError, match="absent from this robot"):
        follower.send_action({f"{motor}.pos": 30.0})
    follower.bus.sync_read.assert_not_called()
    follower.bus.sync_write.assert_not_called()


def test_rigid_tool_reuses_arm_calibration_without_modifying_file(tmp_path):
    names = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
    saved = {
        name: {"id": i, "drive_mode": 0, "homing_offset": 0, "range_min": 0, "range_max": 4095}
        for i, name in enumerate(names, 1)
    }
    path = tmp_path / "pen.json"
    path.write_text(json.dumps(saved))
    bus = _make_bus_mock()
    with patch("lerobot.robots.so_follower.so_follower.SerialMotorsBus", return_value=bus) as factory:
        robot = SO100Follower(
            SO100FollowerConfig(port="NEVER_OPEN", id="pen", calibration_dir=tmp_path, use_gripper=False)
        )
        assert tuple(robot.calibration) == names[:-1]
        assert factory.call_args.kwargs["calibration"] == robot.calibration
        assert tuple(factory.call_args.kwargs["motors"]) == names[:-1]
        with patch("builtins.input", return_value=""):
            assert robot._keep_calibration_file(bus)
        bus.write_calibration.assert_called_once_with(robot.calibration)
    assert json.loads(path.read_text()) == saved


def test_rigid_tool_configures_only_arm_motors(follower):
    SO100Follower.configure(follower)
    configured = [call.args[1] for call in follower.bus.write.call_args_list]
    assert set(configured) == set(follower.bus.motors)


def test_configure_writes_position_pid_coefficients():
    bus_mock = _make_bus_mock()
    bus_mock.motors = ["shoulder_pan"]
    robot = MagicMock()
    robot.bus = bus_mock
    robot.config = SO100FollowerConfig(
        port="/dev/null",
        position_p_coefficient=32,
        position_i_coefficient=1,
        position_d_coefficient=16,
    )

    SO100Follower.configure(robot)

    bus_mock.write.assert_any_call("P_Coefficient", "shoulder_pan", 32)
    bus_mock.write.assert_any_call("I_Coefficient", "shoulder_pan", 1)
    bus_mock.write.assert_any_call("D_Coefficient", "shoulder_pan", 16)
