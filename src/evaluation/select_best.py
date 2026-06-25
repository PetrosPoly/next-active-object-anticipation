"""Rank parameter combinations by an evaluation metric (auto-tuning helper).

Reads ``results/eval/final_results.json`` (produced by ``results_parallel.py``)
and prints the parameter combinations sorted by the chosen metric, best first —
so the grid search can be turned into a concrete "best params" decision instead
of eyeballing plots.

Run:  python -m evaluation.select_best [--metric model_overall_accuracy] [--top 5]
"""
import argparse
import json
import os

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO = os.path.dirname(_SRC)


def _folder_name(p):
    return (f"time_{p.get('time_threshold')}_highdot_{p.get('high_dot_threshold')}_"
            f"highdotcount_{p.get('high_dot_counters_threshold')}_"
            f"dist_{p.get('distance_threshold')}_distcount_{p.get('distance_counters_threshold')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="model_overall_accuracy",
                    help="metric key to rank by (e.g. model_overall_accuracy, precision, recall, llm_interaction_accuracy)")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--results", default=os.path.join(_REPO, "results", "eval", "final_results.json"))
    args = ap.parse_args()

    if not os.path.exists(args.results):
        raise SystemExit(f"No eval results at {args.results}. Run: python -m evaluation.results_parallel")

    rows = json.load(open(args.results))
    if not rows:
        raise SystemExit("final_results.json is empty.")
    if args.metric not in rows[0]:
        raise SystemExit(f"Metric '{args.metric}' not found. Available: {sorted(rows[0].keys())}")

    rows.sort(key=lambda r: (r.get(args.metric) or 0.0), reverse=True)
    print(f"Top {args.top} parameter combinations by '{args.metric}':\n")
    for r in rows[:args.top]:
        val = r.get(args.metric) or 0.0
        print(f"  {val:.4f}  {r.get('sequence', '')}  {_folder_name(r.get('parameters', {}))}")


if __name__ == "__main__":
    main()
