"""Prediction-only and RCA-only author baselines with shared time boundaries."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import redirect_stdout, redirect_stderr
import io
import json
import os
from pathlib import Path
import pickle
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("DGLBACKEND", "pytorch")
os.environ.setdefault("DGLDEFAULTDIR", str(ROOT / "artifacts/baselines/.dgl"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "artifacts/baselines/.matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / "artifacts/baselines/.cache"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

RCA_METHODS = ("rcd", "circa", "tracerca", "microrank", "torai", "diagfusion", "tvdiag")
PREDICTION_METHODS = ("anofusion", "hades", "maat")
METHODS = RCA_METHODS + PREDICTION_METHODS


def dump(path, payload):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)+"\n")


def root_metrics(data, ends, predictions):
    import numpy as np
    if len(ends) != len(predictions):
        raise ValueError("Every evaluation window must have a result or explicit unavailable marker")
    d = data.dataset
    reciprocal, hit1, hit3, hit5, tie_expected = [], [], [], [], []
    available = ties = 0
    for end, scores in zip(ends, predictions):
        target = d.node_targets[end]
        valid = d._root_label_valid.get(d.timestamps[end], bool(target.any()))
        if not d.failure_targets[end] or not valid or not target.any():
            continue
        if scores is None:
            reciprocal.append(0.0); hit1.append(0); hit3.append(0); hit5.append(0); tie_expected.append(0.0)
            continue
        scores = np.asarray(scores)
        if scores.shape != target.shape or not np.isfinite(scores).all():
            raise ValueError("Invalid root scores; refusing to drop this window")
        available += 1
        order = np.argsort(-scores, kind="stable")
        rank = min(int(np.flatnonzero(order == i)[0])+1 for i in np.flatnonzero(target))
        reciprocal.append(1/rank); hit1.append(rank <= 1); hit3.append(rank <= 3); hit5.append(rank <= 5)
        true_score = max(scores[target > 0])
        greater, equal = int((scores > true_score).sum()), int((scores == true_score).sum())
        ties += equal > 1
        tie_expected.append(float(np.mean(1/np.arange(greater+1, greater+equal+1))))
    count = len(reciprocal)
    mean = lambda values: float(np.mean(values)) if count else None
    return {"attributed_positive_windows": count, "available_windows": available,
            "coverage": available/count if count else None,
            "mrr": mean(reciprocal), "hit_at_1": mean(hit1), "hit_at_3": mean(hit3), "hit_at_5": mean(hit5),
            "tie_expected_mrr": mean(tie_expected), "root_tied_windows": ties,
            "unavailable_policy": "zero reciprocal rank and hit, retained in denominator",
            "tie_break": "fixed panel service order; tie-expected MRR additionally reported"}


def metric_tensor(data, end):
    import torch
    d = data.dataset
    indices = [i for i, name in enumerate(d.feature_columns) if name.startswith("metric_")]
    return torch.from_numpy(d.features[end-4:end+1, :, indices].copy().reshape(5, -1))


def graph_input(data, events, end, *, root=None):
    import torch
    from .adapters.frameworks import dgl_batch
    graph = dgl_batch(data.windows([end])[0]["graph"][None])
    features = torch.from_numpy(events(end))
    for j, modality in enumerate(("metric", "log", "trace")):
        graph.ndata[modality] = features[:, j]
    if root is not None:
        target = torch.zeros(len(data.dataset.services), dtype=torch.long)
        target[root] = 1
        graph.ndata["root"] = target
    return graph


def infer_learned(method, model, data, ends, events=None):
    import dgl
    import numpy as np
    import torch
    model.eval()
    result = []
    with torch.no_grad():
        for end in ends:
            if method in ("diagfusion", "tvdiag"):
                roots, _ = model(graph_input(data, events, end))
                result.append(roots.reshape(-1).numpy())
            elif method == "hades":
                logits = model(metric_tensor(data, end)[None], torch.from_numpy(events(end))[None])
                result.append(float(logits.softmax(dim=-1)[0, 1]))
            elif method == "anofusion":
                x = data.windows([end])[0]["x"]
                result.append(float(model(x[None])["risk"][0]))
    return result


def train_neural(args, data, selected, raw, reference):
    import numpy as np
    import torch
    import dgl
    from torch.nn import functional as F
    from .adapters.events import EventFeatures
    from .adapters.learned_specialists import AnoFusionAuthor, HadesAuthor, DiagFusionAuthor, TVDiagAuthor
    from training.metrics import average_precision
    d, method = data.dataset, args.method
    events = None
    if method in ("diagfusion", "tvdiag", "hades"):
        events = EventFeatures(raw, reference, selected["train"], args.output_dir, method,
                               dimension=32, seed=args.seed)
        for end in sorted(set(sum(selected.values(), []))):
            events(end)
    valid_root = lambda end: bool(d.failure_targets[end] and d.node_targets[end].any()
                                 and d._root_label_valid.get(d.timestamps[end], False))
    if method in ("diagfusion", "tvdiag"):
        fit = [end for end in selected["train"] if valid_root(end)]
        model = (DiagFusionAuthor(32, len(d.services), len(d.fault_type_names), args.hidden)
                 if method == "diagfusion" else TVDiagAuthor(32, len(d.fault_type_names), args.hidden))
    elif method == "hades":
        fit = selected["train"]
        model = HadesAuthor(metric_tensor(data, fit[0]).shape[-1], hidden=args.hidden)
    else:
        fit = [end for end in data.normal_train if end in set(selected["train"])]
        if not fit and args.max_windows:
            fit = data.normal_train[:args.max_windows]
        model = AnoFusionAuthor(data)
    if not fit:
        raise ValueError("No eligible training windows for this method")
    labelled = set(fit)
    if method == "hades":
        # Fixed, class-stratified labelled subset of TRAINING windows only.
        labelled = set()
        rng = np.random.default_rng(args.seed)
        for target in (0, 1):
            pool = [end for end in fit if d.failure_targets[end] == target]
            if pool:
                labelled.update(rng.choice(pool, max(1, int(len(pool)*args.label_fraction)), replace=False).tolist())
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    best, best_epoch, history = -float("inf"), None, []
    for epoch in range(args.epochs):
        model.train()
        shuffled = list(fit); random.shuffle(shuffled)
        losses, pseudo_count = [], 0
        for offset in range(0, len(shuffled), args.batch_size):
            batch = shuffled[offset:offset+args.batch_size]
            optimizer.zero_grad()
            if method in ("diagfusion", "tvdiag"):
                roots = torch.tensor([int(d.node_targets[e].argmax()) for e in batch])
                types = torch.tensor([int(d.fault_type_targets[e]) for e in batch])
                graphs = [graph_input(data, events, e, root=int(r)) for e, r in zip(batch, roots)]
                if method == "tvdiag":
                    loss = model.objective(graphs, roots, types)
                else:
                    root_logits, type_logits = model(dgl.batch(graphs))
                    loss = F.cross_entropy(root_logits, roots)+F.cross_entropy(type_logits, types)
            elif method == "hades":
                metrics = torch.stack([metric_tensor(data, e) for e in batch])
                logs = torch.from_numpy(np.stack([events(e) for e in batch]))
                logits = model(metrics, logs)
                known = torch.tensor([e in labelled for e in batch])
                labels = torch.tensor([int(d.failure_targets[e]) for e in batch])
                probabilities = logits.detach().softmax(dim=-1)
                confidence, pseudo = probabilities.max(dim=-1)
                usable = known | ((confidence >= args.pseudo_confidence) & (epoch >= 1))
                if not usable.any():
                    continue
                targets = torch.where(known, labels, pseudo)
                pseudo_count += int((usable & ~known).sum())
                loss = F.cross_entropy(logits[usable], targets[usable])
            else:
                x = torch.stack([data.windows([e])[0]["x"] for e in batch])
                loss = model(x)["loss"]
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite {method} training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            losses.append(float(loss.detach()))
        if not losses:
            raise ValueError("No usable labelled/pseudo-labelled training batch")
        mean_loss = float(np.mean(losses))
        if method in RCA_METHODS:
            predictions = infer_learned(method, model, data, selected["validation"], events)
            criterion = root_metrics(data, selected["validation"], predictions)["mrr"]
            selection = "maximum validation window MRR; earliest epoch on tie"
        elif method == "anofusion":
            criterion = -mean_loss
            selection = "minimum normal-training reconstruction loss"
        else:
            predictions = infer_learned(method, model, data, selected["validation"], events)
            criterion = average_precision(d.failure_targets[selected["validation"]], np.asarray(predictions))
            selection = "maximum validation AP; earliest epoch on tie"
        if criterion is None:
            raise ValueError("Validation partition lacks the labels required for model selection")
        history.append({"epoch": epoch+1, "loss": mean_loss, "selection_value": criterion,
                        "pseudo_labels_used": pseudo_count})
        print(json.dumps({"method": method, **history[-1]}), flush=True)
        if criterion > best:
            best, best_epoch = criterion, epoch+1
            torch.save({"model_state_dict": model.state_dict(), "method": method,
                        "epoch": epoch+1, "fit_ends": fit, "labelled_ends": sorted(labelled),
                        "hyperparameters": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}},
                       args.output_dir / "best_model.pt")
    checkpoint = torch.load(args.output_dir / "best_model.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    validation = infer_learned(method, model, data, selected["validation"], events)
    test = infer_learned(method, model, data, selected["test"], events)
    return validation, test, {"training_history": history, "selected_epoch": best_epoch,
                               "selection_rule": selection, "fit_windows": len(fit),
                               "labelled_windows": len(labelled)}


def run_maat(args, data, selected):
    import numpy as np
    import torch
    from .adapters.learned_specialists import MaatAuthor
    from .author_imports import author_package
    d = data.dataset
    names = {name: i for i, name in enumerate(d.feature_columns)}
    pairs = [(s, i, names["observed_"+name.removeprefix("metric_")])
             for s in range(len(d.services)) for i, name in enumerate(d.feature_columns)
             if name.startswith("metric_")]
    pairs = [(s, i, mask) for s, i, mask in pairs
             if (d.features[data.input_fit_indices, s, mask] > 0).mean() >= 0.95]
    if not pairs:
        raise ValueError("No sufficiently observed metric channels for Maat")
    def values(end, *, future=False):
        if future:
            index = d._time_indices[d.label_target_times[d.timestamps[end]]]
            return np.asarray([d.features[index, s, i] for s, i, _ in pairs], dtype=np.float32)
        return np.stack([d.features[end-4:end+1, s, i] for s, i, _ in pairs], axis=-1)
    def observed(end):
        index = d._time_indices[d.label_target_times[d.timestamps[end]]]
        return np.asarray([d.features[index, s, mask] for s, _, mask in pairs], dtype=np.float32)
    def past_observed(end):
        return np.stack([d.features[end-4:end+1, s, mask] for s, _, mask in pairs], axis=-1)
    model = MaatAuthor(len(pairs), args.diff_steps, args.forecast_samples, args.hidden)
    optimizer = torch.optim.Adam(model.train_net.parameters(), lr=args.learning_rate)
    history, best, best_epoch = [], float("inf"), None
    for epoch in range(args.epochs):
        model.train_net.train()
        ends = list(selected["train"]); random.shuffle(ends)
        losses = []
        for offset in range(0, len(ends), args.batch_size):
            batch = ends[offset:offset+args.batch_size]
            x = torch.from_numpy(np.stack([values(e) for e in batch]))
            target = torch.from_numpy(np.stack([values(e, future=True) for e in batch]))
            mask = torch.from_numpy(np.stack([observed(e) for e in batch]))
            optimizer.zero_grad()
            past_mask = torch.from_numpy(np.stack([past_observed(e) for e in batch]))
            loss = model.loss(x, target, mask, past_mask)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite Maat diffusion loss")
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.train_net.parameters(), 5); optimizer.step()
            losses.append(float(loss.detach()))
        mean_loss = float(np.mean(losses)); history.append({"epoch": epoch+1, "loss": mean_loss})
        print(json.dumps({"method": "maat", **history[-1]}), flush=True)
        if mean_loss < best:
            best, best_epoch = mean_loss, epoch+1
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch+1,
                        "channels": pairs, "model_config": model.config}, args.output_dir / "best_model.pt")
    model.load_state_dict(torch.load(args.output_dir / "best_model.pt", map_location="cpu", weights_only=False)["model_state_dict"])
    def features(ends):
        result = []
        for end in ends:
            x = values(end)
            samples = model.forecast(torch.from_numpy(x[None]), args.seed+end,
                                     torch.from_numpy(past_observed(end)[None])).numpy()
            result.append(model.forecast_features(samples, x[None, -1])[0])
        return np.asarray(result)
    train_features = features(selected["train"])
    api = author_package("maat", "src", "model.detection")
    detector = api.Detector(n_jobs=1)
    # Fit original isolation-forest detector on TRAINING forecasts whose future
    # business labels are normal; no test-time observed future enters features.
    normal = d.failure_targets[selected["train"]] == 0
    if normal.sum() < 2:
        raise ValueError("Maat detector needs normal training forecasts")
    validation_features = features(selected["validation"])
    selected_features = detector.feat_filter(validation_features, d.failure_targets[selected["validation"]])[:args.detector_features]
    detector.fit(train_features[normal][:, selected_features])
    with (args.output_dir / "forecast_detector.pkl").open("wb") as handle:
        pickle.dump(detector, handle)
    np.save(args.output_dir / "detector_feature_indices.npy", selected_features)
    validation = -detector.clf.decision_function(detector.scaler.transform(validation_features[:, selected_features]))
    test = -detector.clf.decision_function(detector.scaler.transform(features(selected["test"])[:, selected_features]))
    return validation.tolist(), test.tolist(), {"training_history": history, "selected_epoch": best_epoch,
        "selection_rule": "minimum training diffusion loss; detector normal training forecasts only",
        "metric_channels": [[d.services[s], d.feature_columns[i]] for s, i, _ in pairs],
        "forecast_features": "sample mean, change from latest observation, and sample std",
        "detector_feature_selection": "author random forest using validation labels only",
        "future_telemetry_access": "training loss only; forbidden during forecast/scoring"}


def run(args):
    import numpy as np
    import torch
    from .adapters.data import PreparedData, file_hash
    from .adapters.raw import RawCaseStore
    from .adapters.statistical_rca import RCDAuthor, CIRCAAuthor, TraceRCAAuthor, MicroRankAuthor, TORAIAuthor
    from .evaluation import validation_threshold, binary_metrics
    from .source import source_lock, verify_sources
    from training.metrics import average_precision
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite run directory: {args.output_dir}")
    if min(args.epochs, args.batch_size, args.threads, args.diff_steps,
           args.forecast_samples, args.detector_features) < 1:
        raise ValueError("Training and sampling counts must be positive")
    if args.max_windows is not None and args.max_windows < 2:
        raise ValueError("Smoke prefix must contain at least two windows")
    if not 0 < args.label_fraction <= 1 or not 0 <= args.pseudo_confidence <= 1:
        raise ValueError("Require 0<label_fraction<=1 and 0<=pseudo_confidence<=1")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.set_num_threads(args.threads)
    panel = args.panel_dir or ROOT / f"artifacts/re2_{args.dataset}_panel_fair_v1"
    data = PreparedData(panel, args.fold)
    d = data.dataset
    selected = {k: v[:args.max_windows] if args.max_windows else list(v) for k, v in data.splits.items()}
    args.output_dir.mkdir(parents=True)
    identity = data.identity(); identity["run_ends"] = selected
    dump(args.output_dir / "data_manifest.json", identity)
    np.savez(args.output_dir / "scaler.npz", mean=data.mean, scale=data.scale)
    archive = args.archive or ROOT / f"data/RCAEval/RE2-{args.dataset.upper()}.zip"
    needs_raw = args.method not in ("anofusion", "maat")
    raw, reference, information = None, None, {}
    started = time.monotonic()
    unavailable = Counter()
    try:
        if needs_raw:
            # Skip only modalities the selected algorithm never consumes.
            # This leaves every used observation and all scoring unchanged.
            raw = RawCaseStore(archive, data,
                include_traces=args.method in ("circa", "tracerca", "microrank", "diagfusion", "tvdiag"),
                include_logs=args.method in ("torai", "diagfusion", "tvdiag", "hades"))
            reference, median, edges, operation_stats = raw.training_reference(
                args.reference_rows, selected["train"] if args.max_windows else None)
            reference.to_csv(args.output_dir / "normal_reference.csv", index=False)
            dump(args.output_dir / "reference_metadata.json", {
                "source": "observed healthy TRAINING input minutes only",
                "rows": len(reference), "columns": list(reference.columns),
                "dependencies": sorted(edges), "raw_archive_sha256": file_hash(archive)})
        if args.method in ("diagfusion", "tvdiag", "hades", "anofusion"):
            validation, test, information = train_neural(args, data, selected, raw, reference)
        elif args.method == "maat":
            validation, test, information = run_maat(args, data, selected)
        else:
            entry_index = d.feature_columns.index("entry_span_count")
            entry_values = d.features[data.input_fit_indices, :, entry_index]*data.scale[entry_index]+data.mean[entry_index]
            entry = d.services[int(entry_values.mean(axis=0).argmax())]
            factories = {"rcd": lambda: RCDAuthor(list(d.services), args.seed),
                         "circa": lambda: CIRCAAuthor(list(d.services), edges, reference, entry),
                         "tracerca": lambda: TraceRCAAuthor(list(d.services)),
                         "microrank": lambda: MicroRankAuthor(list(d.services)),
                         "torai": lambda: TORAIAuthor(list(d.services))}
            model = factories[args.method]()
            test = []
            for end in selected["test"]:
                # RCA is evaluated conditionally on the common annotated failure
                # targets. No future label, injection identity or onset is passed
                # into the algorithm itself. These are not alarm-gated metrics.
                if not (d.failure_targets[end] and d.node_targets[end].any()
                        and d._root_label_valid.get(d.timestamps[end], False)):
                    test.append(None)
                    continue
                window_started = time.monotonic()
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    if args.method in ("rcd", "circa"):
                        normal, current = raw.metric_pair(end, reference, median)
                        prediction = model(normal, current)
                    elif args.method in ("tracerca", "microrank"):
                        invocations = raw.invocation_frame(end, operation_stats)
                        prediction = (model(invocations) if args.method == "tracerca"
                                      else model(raw.window(end)[1], invocations))
                    else:
                        normal, current = raw.metric_pair(end, reference, median)
                        frame = raw.window(end)[0]
                        current_logs = raw.log_samples(d.case_ids[end], frame.time.to_numpy())
                        prediction = model(normal, current, raw.reference_log_samples, current_logs)
                if prediction is None:
                    unavailable["no_discriminative_root_evidence"] += 1
                test.append(prediction)
                print(json.dumps({"method": args.method, "evaluated_input": d.timestamps[end],
                                  "seconds": time.monotonic()-window_started}), flush=True)
            validation = []
            information = {"fit": "author statistical algorithm; no neural checkpoint",
                           "root_reference": "normal training data vs permitted current history",
                           "evaluation_condition": "common true attributed positive targets; labels are not algorithm inputs; no detector gating",
                           "entry_sli": entry if args.method == "circa" else None,
                           "trace_detector": "training-operation latency 3-sigma or observed span error" if args.method in ("tracerca", "microrank") else None}
        prediction_metrics, root_result = None, None
        if args.method in PREDICTION_METHODS:
            scores = np.asarray(test, dtype=float)
            if not np.isfinite(scores).all():
                raise ValueError("Nonfinite prediction risk; no test windows may be silently removed")
            threshold = validation_threshold(d.failure_targets[selected["validation"]], validation)
            prediction_metrics = {"average_precision": average_precision(d.failure_targets[selected["test"]], scores),
                                  "at_validation_threshold": binary_metrics(d.failure_targets[selected["test"]], scores, threshold)}
        else:
            root_result = root_metrics(data, selected["test"], test)
        repo = "rcaeval" if args.method == "torai" else args.method
        provenance = source_lock()["sources"][repo]
        verify_sources(repo)
        from .author_imports import FILE_PATCHES
        report = {"method": args.method, "implementation_id": args.method+"-upstream-history-adapted-v1",
                  "dataset": args.dataset, "fold": args.fold, "seed": args.seed,
                  "smoke_only": bool(args.max_windows), "samples": len(selected["test"]),
                  "failure_prediction": prediction_metrics, "root_cause_window": root_result,
                  "source_repository": provenance["repository_url"], "source_commit": provenance["commit"],
                  "runtime_patch_files": [path for (repository, path) in FILE_PATCHES if repository == repo],
                  "unavailable_windows": dict(unavailable), "elapsed_seconds": time.monotonic()-started,
                  "split_manifest_sha256": identity["split_manifest_sha256"],
                  "hyperparameters": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  "information": information}
        dump(args.output_dir / "metrics.json", report)
        with (args.output_dir / "predictions.jsonl").open("w") as handle:
            for end, prediction in zip(selected["test"], test):
                root = args.method in RCA_METHODS
                scores = np.asarray(prediction) if root and prediction is not None else None
                eligible = bool(d.failure_targets[end] and d.node_targets[end].any()
                                and d._root_label_valid.get(d.timestamps[end], False))
                record = {"case_id": d.case_ids[end], "input_minute": d.timestamps[end],
                          "input_available_before": d.timestamp_epochs[end]+60,
                          "target_minute": d.label_target_times[d.timestamps[end]],
                          "failure_target": int(d.failure_targets[end]),
                          "root_targets": d.node_targets[end].tolist(),
                          "root_label_valid": bool(d._root_label_valid.get(d.timestamps[end], False)),
                          "risk_score": float(prediction) if not root else None,
                          "root_scores": scores.tolist() if scores is not None else None,
                          "available": prediction is not None,
                          "status": ("not_in_conditional_rca_evaluation" if root and not eligible
                                     else "no_discriminative_root_evidence" if prediction is None else "scored"),
                          "ranked_services": [d.services[i] for i in np.argsort(-scores, kind="stable")] if scores is not None else None}
                handle.write(json.dumps(record, allow_nan=False)+"\n")
        print(json.dumps({"saved": str(args.output_dir), "method": args.method,
                          "AP": prediction_metrics["average_precision"] if prediction_metrics else None,
                          "MRR": root_result["mrr"] if root_result else None}), flush=True)
    finally:
        if raw:
            raw.close()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", choices=METHODS, required=True)
    p.add_argument("--dataset", choices=("ob", "tt"), required=True)
    p.add_argument("--fold", type=int, choices=range(5), default=0)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=0.001)
    p.add_argument("--reference-rows", type=int, default=2048)
    p.add_argument("--max-windows", type=int)
    p.add_argument("--diff-steps", type=int, default=100)
    p.add_argument("--forecast-samples", type=int, default=16)
    p.add_argument("--detector-features", type=int, default=64)
    p.add_argument("--label-fraction", type=float, default=0.2)
    p.add_argument("--pseudo-confidence", type=float, default=0.95)
    p.add_argument("--panel-dir", type=Path)
    p.add_argument("--archive", type=Path)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args(argv)
    if not 0 < args.label_fraction <= 1 or not 0 < args.pseudo_confidence <= 1:
        p.error("label fraction and pseudo confidence must lie in (0,1]")
    run(args)


if __name__ == "__main__":
    main()
