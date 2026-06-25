"""LLM activation logic: per-frame criteria evaluation and re-activation gating.

Extracted verbatim from ``main.run_experiment`` — logic unchanged.
"""
from dataclasses import dataclass, field
from typing import Dict, List

from utils.tools import object_within_radius


@dataclass
class ActivationCriteria:
    high_dot_counts_but_also_distance: Dict = field(default_factory=dict)
    low_distance_counts_but_also_high_dot: Dict = field(default_factory=dict)
    less_than_2_seconds_dict: Dict = field(default_factory=dict)
    names_high_dot_counts_and_distance_counts: Dict = field(default_factory=dict)
    names_low_distance_counts_and_high_dot_counts: Dict = field(default_factory=dict)
    names_high_dot_counts_and_distance_values: Dict = field(default_factory=dict)
    names_low_distance_counts_and_high_dot_values: Dict = field(default_factory=dict)
    time_to_approach_dict: Dict = field(default_factory=dict)
    high_dot_history: List = field(default_factory=list)


def evaluate_activation_criteria(filtered, parameters, statistics, gt_provider,
                                 user_velocity_device, user_ema_position, T_Scene_Device,
                                 gt_object_names, gt_start_times, gt_end_times,
                                 current_time_s, seen_high_dot):
    """Build the per-frame criterion dicts. `filtered` is a perception.FilteredObjects.

    Returns (ActivationCriteria, updated seen_high_dot set).
    """
    c = ActivationCriteria()
    high_dot_counts = {}
    low_distance_counts = {}

    objects_in_motion = gt_object_names[(gt_start_times <= current_time_s) & (current_time_s <= gt_end_times)].tolist()

    for index, object_id in enumerate(filtered.filtered_obj_ids):
        object_name = gt_provider.get_instance_info_by_id(object_id).name
        if object_name in objects_in_motion:
            continue

        object_time_xyz, object_time_xz = statistics.interaction_time_user_object(
            user_velocity_device, user_ema_position, filtered.filtered_obj_positions_scene[index][0], T_Scene_Device)

        if object_id in filtered.filtered_high_dot_counts and len(filtered.filtered_high_dot_counts[object_id]) >= parameters["high_dot_counters_threshold"]:
            high_dot_counts[object_name] = filtered.filtered_names_high_dot_counts[object_name]
            if object_id in filtered.filtered_low_distance_counts:
                c.high_dot_counts_but_also_distance[object_name] = high_dot_counts[object_name]

        if object_id in filtered.filtered_low_distance_counts and len(filtered.filtered_low_distance_counts[object_id]) >= parameters["distance_counters_threshold"]:
            low_distance_counts[object_name] = filtered.filtered_names_low_distance_counts[object_name]
            if object_id in filtered.filtered_high_dot_counts:
                c.low_distance_counts_but_also_high_dot[object_name] = low_distance_counts[object_name]

        if (object_name in filtered.filtered_names_high_dot_counts and
                object_name in filtered.filtered_names_low_distance_counts and
                filtered.filtered_names_duration[object_name] > 1):
            c.names_high_dot_counts_and_distance_counts[object_name] = filtered.filtered_names_high_dot_counts[object_name]
            c.names_high_dot_counts_and_distance_values[object_name] = f"{float(filtered.filtered_dot_products[index]):.3f}"
            c.names_low_distance_counts_and_high_dot_counts[object_name] = filtered.filtered_names_low_distance_counts[object_name]
            c.names_low_distance_counts_and_high_dot_values[object_name] = f"{float(filtered.filtered_distances[index]):.3f}"
            c.time_to_approach_dict[object_name] = object_time_xz
            if object_time_xz < parameters["time_threshold"]:
                c.less_than_2_seconds_dict[object_name] = object_time_xz

    seen_high_dot |= set(high_dot_counts.keys())
    c.high_dot_history = list(seen_high_dot)
    return c, seen_high_dot


def soft_activation(criteria, parameters, prev_scores):
    """Weighted, temporally-smoothed activation gate (#4 + #6).

    For each candidate object (one with both consistent focus and proximity
    counts), combine normalized focus, proximity and time-to-approach into a
    [0,1] score, EMA-smooth it across frames, and open the gate when the best
    candidate that also has focus history exceeds the threshold.

    Returns (gate_open: bool, updated_scores: dict).
    """
    focus = criteria.names_high_dot_counts_and_distance_counts        # {name: focus count}
    prox = criteria.names_low_distance_counts_and_high_dot_counts     # {name: proximity count}
    times = criteria.time_to_approach_dict                           # {name: seconds}
    history = set(criteria.high_dot_history)

    focus_cap = max(1, parameters["high_dot_counters_threshold"])
    prox_cap = max(1, parameters["distance_counters_threshold"])
    time_thr = parameters["time_threshold"]
    alpha = parameters.get("soft_score_ema_alpha", 1.0)
    threshold = parameters.get("soft_activation_threshold", 0.6)

    new_scores, best = {}, 0.0
    for name in focus:
        f = min(1.0, focus.get(name, 0) / focus_cap)
        p = min(1.0, prox.get(name, 0) / prox_cap)
        t = times.get(name, float("inf"))
        t_norm = max(0.0, min(1.0, time_thr / t)) if t and t > 0 else 0.0
        raw = (f + p + t_norm) / 3.0
        ema = alpha * raw + (1 - alpha) * prev_scores.get(name, raw)
        new_scores[name] = ema
        if name in history and ema > best:
            best = ema
    return best >= threshold, new_scores


def check_reactivation(llm_activated, last_activation_time, user_relative_total_movement,
                       group_analyzer, visible_distance_device_objects_scene, visible_obj_names,
                       current_time_s, parameters):
    """Decide whether to re-arm the LLM. Returns the (possibly updated) triple
    (llm_activated, last_activation_time, user_relative_total_movement)."""
    if llm_activated is not True:
        return llm_activated, last_activation_time, user_relative_total_movement

    current_objects_within_radius = object_within_radius(
        visible_distance_device_objects_scene, visible_obj_names, radius=1.5)

    group_analyzer.add_objects(current_time_s, current_objects_within_radius)
    user_objects = group_analyzer.compare_objects()
    users_move = group_analyzer.user_move(user_relative_total_movement)
    time_since_last_activation = current_time_s - last_activation_time

    if time_since_last_activation > parameters["minimum_time_deactivated"]:
        if users_move or user_objects or time_since_last_activation > parameters["maximum_time_deactivated"]:
            user_relative_total_movement = 0
            last_activation_time = current_time_s
            llm_activated = False

    return llm_activated, last_activation_time, user_relative_total_movement
