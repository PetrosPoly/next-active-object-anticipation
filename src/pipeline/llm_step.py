"""The LLM activation step: gate -> query -> parse -> record -> log.

Extracted verbatim from ``main.run_experiment`` section "LLM Query and
Activation". The only behavioural change is that debug ``print`` calls became
``logging`` calls.
"""
import logging
import os

from utils.openai_models_work import activate_llm, append_to_history_string, process_llm_response, setup_logger
from utils.llama import activate_llama
from pipeline.activation import soft_activation

_log = logging.getLogger(__name__)


def query_and_log_llm(state, criteria, args, parameters, llama, current_time_s, project_path, scene="Unknown"):
    """Run one LLM activation if the criteria/flags allow it.

    Mutates ``state`` (an ActivationState): sets llm_activated/last_activation_time
    and records the prediction. Returns the list of predicted objects when the
    LLM was activated this frame, otherwise ``None``.
    """
    # Activation gate. "soft" (default): weighted, temporally-smoothed score
    # (#4 + #6). "strict": the original AND-of-3-criteria + focus-history check.
    if parameters.get("activation_mode", "strict") == "soft":
        # update the EMA scores every frame for temporal consistency
        gate_open, state.candidate_scores = soft_activation(criteria, parameters, state.candidate_scores)
    else:
        gate_open = bool(
            criteria.high_dot_counts_but_also_distance
            and criteria.low_distance_counts_but_also_high_dot
            and criteria.less_than_2_seconds_dict
            and any(obj in criteria.high_dot_history for obj in criteria.less_than_2_seconds_dict.keys())
        )

    if not gate_open:
        return None

    # only when the LLM is enabled and currently disarmed
    if not (args.use_llm and not state.llm_activated):
        return None

    history_log = append_to_history_string(
        current_time_s,
        scene,
        criteria.names_high_dot_counts_and_distance_counts,
        criteria.names_low_distance_counts_and_high_dot_counts,
        criteria.names_high_dot_counts_and_distance_values,
        criteria.names_low_distance_counts_and_high_dot_values,
        criteria.time_to_approach_dict,
        state.prediction_dict,
    )
    history_log_string = str(history_log)

    _log.debug("Llama %s", "activated" if llama else "not activated")
    if llama is True:
        llm_response = activate_llama(history_log_string, parameters)
    else:
        llm_response = activate_llm(history_log, parameters)

    objects_possibility, rationale, predicted_objects, goal = process_llm_response(llm_response)
    _log.info("[%.3fs] predicted: %s | goal: %s", current_time_s, predicted_objects, goal)
    _log.debug("possibilities: %s | rationale: %s", objects_possibility, rationale)

    state.last_activation_time = current_time_s
    state.llm_activated = True
    state.record(current_time_s, objects_possibility, rationale, predicted_objects, goal)

    # Per-activation log files
    log_folder = os.path.join(project_path, f'logs/time_{current_time_s}.log')
    os.makedirs(os.path.dirname(log_folder), exist_ok=True)
    setup_logger(log_folder).info(f"LLM Response: {llm_response}")

    history_log_folder = os.path.join(project_path, f'logs/history_{current_time_s}.log')
    os.makedirs(os.path.dirname(history_log_folder), exist_ok=True)
    setup_logger(history_log_folder).info(f"History Log: {history_log}")

    return predicted_objects
