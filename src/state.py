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
