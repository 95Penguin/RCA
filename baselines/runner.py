"""Standalone author-baseline runner; legacy models/results remain separate."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("DGLBACKEND", "pytorch")
os.environ.setdefault("DGLDEFAULTDIR", str(ROOT / "artifacts/baselines/.dgl"))


def dump(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def evaluate(model, data, ends, method, batch_size, repeats, seed, *, include_roots=True):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from .adapters.frameworks import permutation_scores
    scores, root_scores = [], []
    model.eval()
    with torch.no_grad():
        for batch in DataLoader(data.windows(ends), batch_size=batch_size, shuffle=False):
            out = model(batch["x"], batch["graph"])
            score = out["risk_score"] if method == "art" else out["failure_logit"].sigmoid()
            scores.append(score.numpy())
            if method == "instantops":
                root = permutation_scores(model, batch["x"], batch["graph"], repeats=repeats, seed=seed) if include_roots else torch.zeros_like(batch["root"])
            else:
                root = out["root_logits"]
            root_scores.append(root.numpy())
    return np.concatenate(scores), np.concatenate(root_scores)


def run(args):
    import numpy as np
    import torch
    from torch.nn import functional as F
    from torch.utils.data import DataLoader
    from .adapters.data import PreparedData
    from .adapters.frameworks import IMPLEMENTATIONS, build_framework
    from .evaluation import validation_threshold, summarize_predictions
    from .source import PATCHES, source_lock, verify_sources
    from training.metrics import average_precision

    if args.epochs < 1 or args.batch_size < 1 or args.history < 2 or args.repeats < 1 or args.threads < 1:
        raise ValueError("epochs, batch size and repeats must be positive; history>=2")
    if args.max_windows is not None and args.max_windows < 2:
        raise ValueError("max-windows must be at least two")
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite run: {args.output_dir}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    panel = args.panel_dir or ROOT / f"artifacts/re2_{args.dataset}_panel_fair_v1"
    data = PreparedData(panel, args.fold, args.history)
    model = build_framework(args.method, data, hidden=args.hidden, dropout=args.dropout, top_k=args.top_k)
    source_repo = None if args.method == "instantops" else args.method
    upstream = source_lock()["sources"].get(source_repo)
    if source_repo:
        verify_sources(source_repo)
    selected = {k: v[:args.max_windows] if args.max_windows else list(v) for k, v in data.splits.items()}
    train_ends = data.normal_train if args.method == "art" else selected["train"]
    if args.max_windows:
        train_ends = train_ends[:args.max_windows]
    if not train_ends:
        raise ValueError("No eligible training windows; ART needs five observed normal training minutes")
    labels = data.dataset.failure_targets[train_ends]
    pos_weight = torch.tensor(float((len(labels)-labels.sum()) / max(1, labels.sum())))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(data.windows(train_ends), batch_size=args.batch_size,
                        shuffle=True, generator=generator)
    args.output_dir.mkdir(parents=True)
    identity = data.identity()
    identity["run_ends"] = {**selected, "fit": train_ends}
    dump(args.output_dir / "data_manifest.json", identity)
    np.savez(args.output_dir / "scaler.npz", mean=data.mean, scale=data.scale)
    history, best, best_epoch = [], -float("inf"), None
    start = time.monotonic()
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for batch in loader:
            optimizer.zero_grad()
            out = model(batch["x"], batch["graph"])
            if args.method == "art":
                loss = out["reconstruction_loss"]
            else:
                prediction_loss = F.binary_cross_entropy_with_logits(
                    out["failure_logit"], batch["y"].float(), pos_weight=pos_weight)
                valid = batch["root_valid"] & (batch["y"] == 1) & (batch["root"].sum(dim=1) > 0)
                if args.method == "eadro":
                    # Equal weighting of author detector/localizer objectives;
                    # preserve independent business labels rather than infer them from root labels.
                    loss = 0.5 * prediction_loss
                    if valid.any():
                        loss = loss + 0.5*F.cross_entropy(out["root_logits"][valid], batch["root"][valid].argmax(dim=1))
                else:
                    loss = prediction_loss
                    if args.method == "ltfp" and valid.any():
                        loss = loss + 0.5*F.cross_entropy(out["root_logits"][valid], batch["root"][valid].argmax(dim=1))
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.extend([float(loss.detach())] * len(batch["y"]))
        val_risk, _ = evaluate(model, data, selected["validation"], args.method,
                               args.batch_size, args.repeats, args.seed, include_roots=False)
        val_y = data.dataset.failure_targets[selected["validation"]]
        ap = average_precision(val_y, val_risk)
        if args.method == "art":
            # Select unsupervised model by normal training reconstruction loss,
            # not by validation/test root labels or incident classification labels.
            criterion = -float(np.mean(losses))
            selection = "minimum normal-train reconstruction epoch loss"
        else:
            # All-negative validation cannot define AP; fail rather than consult test.
            if ap is None:
                raise ValueError("Validation split has no business positives; AP model selection undefined")
            criterion, selection = ap, "maximum validation AP; earliest epoch on tie"
        history.append({"epoch": epoch+1, "training_loss": float(np.mean(losses)), "validation_ap": ap})
        if criterion > best:
            best, best_epoch = criterion, epoch+1
            torch.save({"model_state_dict": model.state_dict(), "epoch": best_epoch,
                        "method": args.method, "implementation_id": IMPLEMENTATIONS[args.method],
                        "source_commit": upstream["commit"] if upstream else None,
                        "scaler": {"mean": data.mean, "scale": data.scale},
                        "model_config": {"feature_columns": list(data.dataset.feature_columns),
                                         "services": list(data.dataset.services),
                                         "history": args.history, "hidden": args.hidden,
                                         "dropout": args.dropout, "top_k": args.top_k}},
                       args.output_dir / "best_model.pt")
        print(json.dumps({"method": args.method, **history[-1]}), flush=True)
    checkpoint = torch.load(args.output_dir / "best_model.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    val_risk, _ = evaluate(model, data, selected["validation"], args.method,
                           args.batch_size, args.repeats, args.seed, include_roots=False)
    threshold = validation_threshold(data.dataset.failure_targets[selected["validation"]], val_risk)
    risk, roots = evaluate(model, data, selected["test"], args.method,
                           args.batch_size, args.repeats, args.seed)
    metrics = summarize_predictions(data, selected["test"], risk, roots, threshold, args.method != "art")
    metrics.update({"method": args.method, "implementation_id": IMPLEMENTATIONS[args.method],
                    "dataset": args.dataset, "fold": args.fold, "seed": args.seed,
                    "source_repository": upstream["repository_url"] if upstream else None,
                    "source_commit": upstream["commit"] if upstream else None,
                    "compatibility_patches": PATCHES.get(args.method, []),
                    "selection_rule": selection, "selected_epoch": best_epoch,
                    "training_history": history, "elapsed_seconds": time.monotonic()-start,
                    "smoke_only": bool(args.max_windows),
                    "protocol": "5 completed minute history -> exact next-minute business state and root ranking",
                    "task_adaptation": "historical observed-t residual for future state" if args.method == "art" else "future business labels with panel input; see RUNNING.md",
                    "framework_input": "common RE2 minute panel; Eadro uses log counts instead of author Hawkes inputs",
                    "hyperparameters": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    "environment": {"python": sys.version, "torch": torch.__version__, "platform": platform.platform()},
                    "split_manifest_sha256": identity["split_manifest_sha256"]})
    dump(args.output_dir / "metrics.json", metrics)
    d = data.dataset
    with (args.output_dir / "predictions.jsonl").open("w") as handle:
        for row, end in enumerate(selected["test"]):
            order = np.argsort(-roots[row], kind="stable")
            record = {"case_id": d.case_ids[end], "input_minute": d.timestamps[end],
                      "target_minute": d.label_target_times[d.timestamps[end]],
                      "event_id": d.event_ids[end], "failure_target": int(d.failure_targets[end]),
                      "root_targets": d.node_targets[end].tolist(),
                      "root_label_valid": bool(d._root_label_valid.get(d.timestamps[end], bool(d.node_targets[end].any()))),
                      "score_kind": "historical_residual" if args.method == "art" else "failure_probability",
                      "risk_score": float(risk[row]), "root_scores": roots[row].tolist(),
                      "ranked_services": [d.services[i] for i in order],
                      "method": args.method, "implementation_id": IMPLEMENTATIONS[args.method]}
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+"\n")
    print(json.dumps({"saved": str(args.output_dir), "AP": metrics["average_precision"],
                      "MRR": metrics["root_cause_window"]["mrr"]}), flush=True)


def doctor():
    from .source import verify_sources
    pinned = verify_sources()
    dependencies = {name: bool(importlib.util.find_spec(name))
                    for name in ("torch", "numpy", "pandas", "dgl", "scipy", "sklearn", "packaging")}
    print(json.dumps({"sources_verified": pinned, "dependencies_present": dependencies,
                      "implemented_frameworks": ["ltfp", "instantops", "eadro", "art"],
                      "other_methods": "source snapshots pinned; adapters pending"}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor")
    p = sub.add_parser("run")
    p.add_argument("--method", choices=("ltfp", "instantops", "eadro", "art"), required=True)
    p.add_argument("--dataset", choices=("ob", "tt"), required=True)
    p.add_argument("--panel-dir", type=Path)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--fold", type=int, choices=range(5), default=0)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--history", type=int, choices=[5], default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--learning-rate", type=float, default=0.001)
    p.add_argument("--top-k", type=int, default=2)
    p.add_argument("--repeats", type=int, default=16)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--max-windows", type=int, help="Prefix subset per split for smoke testing only")
    args = parser.parse_args()
    if args.command == "doctor":
        doctor()
    else:
        run(args)


if __name__ == "__main__":
    main()
