"""Unified inference: learned forecasting and operation-conditioned CPCA.

No labels, injection time or event identifiers are needed by this interface.
Normal references must have been collected before the decision window.
"""
from collections import defaultdict
import math

from .cpca import score_request
from .aggregation import aggregate_operations, MODES
from .config import ATTRIBUTION_DEFAULTS as DEFAULTS


def _p95(values):
    values = sorted(values)
    return values[max(math.ceil(.95 * len(values)) - 1, 0)]


def attribute_requests(requests, normal_local, normal_edge, normal_entry_p95,
                       dataset, *, min_op_requests=DEFAULTS["min_op_requests"],
                       max_requests=DEFAULTS["max_requests"], temperature=DEFAULTS["temperature"],
                       aggregation=DEFAULTS["aggregation"], restoration=DEFAULTS["restoration"],
                       gap_policy=DEFAULTS["gap_policy"], clock_tolerance_us=DEFAULTS["clock_tolerance_us"],
                       tree_policy=DEFAULTS["tree_policy"]):
    """Attribute one completed minute of requests, without ground-truth labels."""
    if temperature <= 0 or min_op_requests < 1 or max_requests < 1:
        raise ValueError("temperature and request limits must be positive")
    if aggregation not in MODES:
        raise ValueError(f"aggregation must be one of {MODES}")
    states = {}
    rejected = 0
    omitted_spans = 0
    partial_requests = 0
    for spans in requests:
        scored = score_request(spans, normal_local, normal_edge, dataset,
                               restoration, gap_policy, clock_tolerance_us, tree_policy)
        if scored is None:
            rejected += 1
            continue
        root, deltas, local = scored
        omitted = len(spans) - len(local)
        omitted_spans += omitted
        partial_requests += int(omitted > 0)
        key = (root.service, root.operation)
        state = states.setdefault(key, {"durations": [], "deltas": defaultdict(float)})
        if len(state["durations"]) >= max_requests:
            continue
        state["durations"].append(root.duration)
        for service, delta in deltas.items():
            state["deltas"][service] += delta
    if not states:
        return {"scores": {}, "ranking": [], "operation_weights": {},
                "abstained": True, "rejected_requests": rejected,
                "operation_request_counts": {}, "omitted_spans": omitted_spans,
                "partial_requests": partial_requests}
    eligible = {key: state for key, state in states.items()
                if len(state["durations"]) >= min_op_requests}
    if not eligible:
        key = max(states, key=lambda k: len(states[k]["durations"]))
        eligible = {key: states[key]}
    ratios = {key: (_p95(state["durations"]) / max(normal_entry_p95[key], 1e-6)
                    if normal_entry_p95.get(key, 0.) > 0 else 0.)
              for key, state in eligible.items()}
    operation_scores = {key: {service: delta / sum(state["durations"])
                              for service, delta in state["deltas"].items()}
                        for key, state in eligible.items()}
    counts = {key: len(state["durations"]) for key, state in eligible.items()}
    scores, weights = aggregate_operations(operation_scores, counts, ratios,
                                           mode=aggregation, temperature=temperature)
    abstained = max(scores.values(), default=0.) <= 0.
    return {"scores": dict(scores),
            "ranking": [] if abstained else sorted(scores, key=lambda s: (-scores[s], s)),
            "operation_weights": weights, "abstained": abstained,
            "rejected_requests": rejected,
            "operation_request_counts": {key: len(state["durations"]) for key, state in eligible.items()},
            "omitted_spans": omitted_spans, "partial_requests": partial_requests}


def predict_and_attribute(model, node_features, adjacency, operation_features,
                          operation_mask, requests, normal_local, normal_edge,
                          normal_entry_p95, dataset, *, threshold=.5,
                          calibration_temperature=1., **attribution_options):
    """Joint output for one window; CPCA runs only when the risk gate opens.

    ``threshold`` is fitted on raw forecast probabilities, as in the trainer.
    Calibration changes the displayed probability, not this gate decision.
    The model's original training/eval state is restored on return.
    """
    import torch
    if node_features.shape[0] != 1:
        raise ValueError("This request-tree interface accepts one window (batch=1)")
    if not 0 <= threshold <= 1 or calibration_temperature <= 0:
        raise ValueError("threshold must be in [0,1] and temperature positive")
    training = model.training
    try:
        model.eval()
        with torch.inference_mode():
            out = model(node_features, adjacency, operation_features, operation_mask)
            raw = out["probability"][0].clamp(1e-6, 1 - 1e-6)
            probability = torch.sigmoid(torch.logit(raw) / calibration_temperature).item()
            alert = raw.item() >= threshold
        attribution = None
        if alert:
            attribution = attribute_requests(requests, normal_local, normal_edge,
                                             normal_entry_p95, dataset, **attribution_options)
        return {"failure_probability": probability, "raw_failure_probability": raw.item(),
                "alert": alert,
                "operation_risks": out["operation_scores"][0].cpu().tolist(),
                "attribution": attribution}
    finally:
        model.train(training)
