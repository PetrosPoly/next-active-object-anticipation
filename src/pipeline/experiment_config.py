"""Experiment parameter grid and output naming.

Each list below may hold one or more values; ``build_param_combinations()``
expands the full Cartesian grid, i.e. one experiment per combination. Add
values to any list to run a parameter sweep.
"""
from itertools import product

# --- Object-filtering / LLM-activation thresholds ---
TIME_THRESHOLDS = [2]               # seconds before interaction to activate the LLM
AVG_DOT_HIGH = [0.9]                # keep objects with average gaze dot product above this
AVG_DOT_LOW = [0.2]                 # ...or above this if they are very close
AVG_DISTANCE_HIGH = [3]             # metres: relaxed proximity for high-focus objects
AVG_DISTANCE_LOW = [1]              # metres: strict proximity for low-focus objects
HIGH_DOT_THRESHOLDS = [0.9]         # dot value counted as "looking at"
DISTANCE_THRESHOLDS = [2]           # metres counted as "near"
HIGH_DOT_COUNTERS_THRESHOLD = [60]  # frames of high gaze required to activate
DISTANCE_COUNTERS_THRESHOLD = [30]  # frames of proximity required to activate
WINDOW_TIMES = [3.0]                # seconds: sliding-window length

# --- LLM re-activation control ---
MIN_TIME_DEACTIVATED = [2.0]        # do not re-query before this (s)
MAX_TIME_DEACTIVATED = [5.0]        # force a re-query after this (s)
USER_RELATIVE_MOVEMENT = [2.0]      # user movement (m) that triggers a re-query
OBJECT_PERCENTAGE_OVERLAP = [0.7]   # overlap of surrounding objects = same area


def build_param_combinations():
    """Expand the grid into a list of parameter dicts (one per experiment)."""
    return [
        {
            "time_threshold": t,
            "avg_dot_high": adh,
            "avg_dot_low": adl,
            "avg_distance_high": adhg,
            "avg_distance_low": adlg,
            "high_dot_threshold": hdt,
            "distance_threshold": dt,
            "high_dot_counters_threshold": hdct,
            "distance_counters_threshold": dct,
            "window_time": w,
            "minimum_time_deactivated": mintd,
            "maximum_time_deactivated": maxtd,
            "user_relative_movement": urm,
            "object_percentage_overlap": obo,
        }
        for t, adh, adl, adhg, adlg, hdt, dt, hdct, dct, w, mintd, maxtd, urm, obo in product(
            TIME_THRESHOLDS, AVG_DOT_HIGH, AVG_DOT_LOW,
            AVG_DISTANCE_HIGH, AVG_DISTANCE_LOW,
            HIGH_DOT_THRESHOLDS, DISTANCE_THRESHOLDS,
            HIGH_DOT_COUNTERS_THRESHOLD, DISTANCE_COUNTERS_THRESHOLD, WINDOW_TIMES,
            MIN_TIME_DEACTIVATED, MAX_TIME_DEACTIVATED, USER_RELATIVE_MOVEMENT,
            OBJECT_PERCENTAGE_OVERLAP,
        )
    ]


def make_parameter_folder_name(p):
    """Folder name encoding the parameter combination (used under results/)."""
    return (
        f"time_{p['time_threshold']}_"
        f"highdot_{p['high_dot_threshold']}_"
        f"highdotcount_{p['high_dot_counters_threshold']}_"
        f"dist_{p['distance_threshold']}_"
        f"distcount_{p['distance_counters_threshold']}"
    )
