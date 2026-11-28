"""Canonical RCA evaluation on an explicit forecast cohort, without model imports.

All event windows contribute to each service's mean; missing evidence is zero.
Ties receive their worst rank over the full service universe. No evidence is
abstention (zero hit and reciprocal rank), rather than an arbitrary ordering.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path


def _rank(scores, root, services):
    values = [scores.get(service, 0.) for service in services]
    if max(values, default=0.) <= 0:
        return None
    return sum(value >= scores.get(root, 0.) for value in values)


def _metrics(ranks):
    n = len(ranks)
    if not n:
        return {"n": 0, "hit_at_1": None, "hit_at_3": None, "hit_at_5": None,
                "mrr": None, "abstentions": 0}
    return {"n": n, "hit_at_1": sum(r == 1 for r in ranks) / n,
            "hit_at_3": sum(r is not None and r <= 3 for r in ranks) / n,
            "hit_at_5": sum(r is not None and r <= 5 for r in ranks) / n,
            "mrr": sum(1 / r for r in ranks if r is not None) / n,
            "abstentions": sum(r is None for r in ranks)}


def evaluate_attribution(audit, cohort, method="cpca"):
    folds = cohort["folds"]
    services = list(folds[0]["services"])
    if any(list(fold["services"]) != services for fold in folds):
        raise ValueError("Fold service order differs")
    windows = {}
    for case in audit["cases"].values():
        for window in case["windows"]:
            key = (case["case_id"], int(window["minute"]))
            if key in windows:
                raise ValueError(f"Duplicate audit window {key}")
            windows[key] = window
    events = defaultdict(list)
    window_ranks = []
    covered = 0
    seen = set()
    for fold in folds:
        for prediction in fold["test_predictions"]:
            if not prediction["failure_target"] or not prediction["root"]:
                continue
            key = (prediction["case"], int(prediction["minute"]))
            if key in seen:
                raise ValueError(f"Duplicate forecast window {key}")
            seen.add(key)
            root = prediction["root"]
            if root not in services:
                raise ValueError(f"Unknown root service {root}")
            window = windows.get(key, {})
            scores = window.get("scores_by_method", {}).get(method,
                     window.get("cpca_scores", {}) if method == "cpca" else {})
            covered += int(bool(window))
            window_ranks.append(_rank(scores, root, services))
            events[prediction["event"]].append((root, scores))
    event_ranks = {}
    for event, rows in events.items():
        if not event or len({root for root, _ in rows}) != 1:
            raise ValueError(f"Missing or inconsistent root for event {event!r}")
        means = {service: sum(scores.get(service, 0.) for _, scores in rows) / len(rows)
                 for service in services}
        event_ranks[event] = _rank(means, rows[0][0], services)
    return {"protocol": "explicit forecast cohort; zero-filled means; full services; worst ties; abstain=0RR",
            "method": method, "attribution_version": audit.get("implementation_version"),
            "positive_windows": len(window_ranks), "covered_windows": covered,
            "event": _metrics(list(event_ranks.values())), "window": _metrics(window_ranks),
            "event_ranks": event_ranks}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--forecast-cohort", type=Path, required=True)
    parser.add_argument("--method", default="cpca")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Output exists; use a new versioned path")
    report = evaluate_attribution(json.loads(args.audit.read_text()),
                                  json.loads(args.forecast_cohort.read_text()), args.method)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["event"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
