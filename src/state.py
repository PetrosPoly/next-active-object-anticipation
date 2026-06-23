"""Cross-frame state objects for the per-frame anticipation loop.

Encapsulates the mutable state that persists between frames so the main loop
reads as orchestration. Logic extracted verbatim from ``main.run_experiment``.
"""
from collections import deque

from utils.tools import exponential_filter, user_movement_calculation


class UserMotionState:
    """Tracks the user's EMA-filtered position/velocity across frames."""

    def __init__(self):
        self.position_before = None
        self.velocity_before = None
        self.positions = None   # deque[(t, ema_position)]
        self.velocities = None  # deque[(t, ema_velocity)]

    def update(self, aria_3d_pose_with_dt, T_Device_Cam, current_time_s):
        """Advance state with the current frame's device pose.

        Returns (T_Scene_Device, T_Scene_Cam, user_position_scene,
        user_velocity_device, user_ema_position, movement_timestep). The first
        valid frame yields movement_timestep == 0.
        """
        p_dev_scene = aria_3d_pose_with_dt.data()
        T_Scene_Device = p_dev_scene.transform_scene_device
        T_Scene_Cam = T_Scene_Device @ T_Device_Cam

        user_position_scene = aria_3d_pose_with_dt.data().transform_scene_device.translation()[0]
        user_velocity_device = aria_3d_pose_with_dt.data().device_linear_velocity
        # velocity device -> scene (rotation only)
        user_velocity_scene = T_Scene_Device.rotation().to_matrix() @ user_velocity_device

        if self.position_before is None and self.velocity_before is None:
            user_ema_position = self.position_before = user_position_scene
            user_ema_velocity = self.velocity_before = user_velocity_scene
            movement_timestep = 0
            self.positions = deque([(current_time_s, 0)])
            self.velocities = deque([(current_time_s, 0)])
        else:
            user_ema_position = exponential_filter(user_position_scene, self.position_before, alpha=0.9)
            user_ema_velocity = exponential_filter(user_velocity_scene, self.velocity_before, alpha=0.9)
            movement_timestep = user_movement_calculation(self.position_before, user_ema_position)
            self.positions.append((current_time_s, user_ema_position))
            self.velocities.append((current_time_s, user_ema_velocity))
            self.position_before = user_ema_position
            self.velocity_before = user_ema_velocity

        return (T_Scene_Device, T_Scene_Cam, user_position_scene,
                user_velocity_device, user_ema_position, movement_timestep)


class ActivationState:
    """Holds LLM activation flags and the per-timestamp output archives."""

    def __init__(self):
        self.llm_activated = False
        self.last_activation_time = 0

        # keyed by timestamp (serialized at the end of the run)
        self.possibility_dict = {}
        self.rationale_dict = {}
        self.prediction_dict = {}
        self.goal_dict = {}

        # chronological lists (kept for parity with the original)
        self.possibilities = []
        self.rationales = []
        self.predictions = []
        self.goals = []
        self.llm_times = []

    def record(self, t, possibility, rationale, predicted, goal):
        self.possibility_dict[t] = possibility
        self.rationale_dict[t] = rationale
        self.prediction_dict[t] = predicted
        self.goal_dict[t] = goal
        self.possibilities.append(possibility)
        self.rationales.append(rationale)
        self.predictions.append(predicted)
        self.goals.append(goal)
        self.llm_times.append(t)
