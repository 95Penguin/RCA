"""Canonical raw RE2 observations, cut strictly by availability time."""
from __future__ import annotations

import csv
import hashlib
import io
import math
import re
import zipfile
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd


def service_alias(name):
    return "frontendservice" if name == "frontend" else name


def log_template(message):
    """Fixed parser, not a template model fitted to evaluation data."""
    text = str(message).lower()
    text = re.sub(r"\b\d{4}-\d\d-\d\d\b|\b\d\d:\d\d:\d\d(?:\.\d+)?", " <time> ", text)
    text = re.sub(r"\b(?:[0-9a-f]{8,}|\d+(?:\.\d+)*)\b", " <var> ", text)
    return " ".join(text.split())[:512]


class RawCaseStore:
    def __init__(self, archive, data, *, include_traces=True, include_logs=True):
        self.archive = Path(archive)
        self.data, self.dataset = data, data.dataset
        self.include_traces, self.include_logs = include_traces, include_logs
        self.zip = zipfile.ZipFile(self.archive)
        self.members = {}
        for name in self.zip.namelist():
            parts = name.split("/")
            if len(parts) == 4 and parts[2] in {"1", "2", "3"}:
                self.members.setdefault(f"{parts[0]}_{parts[1]}_{parts[2]}", {})[parts[-1]] = name
        missing = set(self.dataset.case_ids) - self.members.keys()
        if missing:
            raise ValueError(f"Raw archive misses panel cases: {sorted(missing)[:3]}")
        self.service_index = {s: i for i, s in enumerate(self.dataset.services)}

    def close(self):
        self.zip.close()

    def rows(self, case_id, filename):
        with self.zip.open(self.members[case_id][filename]) as raw:
            yield from csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8-sig", newline=""))

    @lru_cache(maxsize=2)
    def case(self, case_id):
        frame = pd.DataFrame(self.rows(case_id, "simple_metrics.csv"))
        frame = frame.apply(pd.to_numeric, errors="coerce")
        frame = frame.loc[np.isfinite(frame["time"])].sort_values("time")
        frame.columns = ["time" if c == "time" else service_alias(c.rsplit("_", 1)[0])+"_"+c.rsplit("_", 1)[1]
                         for c in frame.columns]
        spans = []
        for row in (self.rows(case_id, "traces.csv") if self.include_traces else ()):
            try:
                start = float(row["startTime"])/1e6
                duration = float(row["duration"])/1e6
            except (ValueError, TypeError):
                continue
            if not math.isfinite(start+duration) or duration < 0:
                continue
            spans.append({"trace": row["traceID"], "span": row["spanID"], "parent": row["parentSpanID"],
                          "service": service_alias(row["serviceName"]), "operation": row["operationName"],
                          "start": start, "available": start+duration, "latency": duration*1000,
                          "error": str(row.get("statusCode", "")).startswith("5")})
        spans.sort(key=lambda s: s["available"])
        logs = []
        for row in (self.rows(case_id, "logs.csv") if self.include_logs else ()):
            try:
                available = float(row["timestamp"])/1e9
            except (TypeError, ValueError):
                continue
            if math.isfinite(available):
                logs.append({"available": available, "service": service_alias(row["container_name"]),
                             "text": log_template(row.get("message", "")), "level": str(row.get("level", "")).lower()})
        logs.sort(key=lambda s: s["available"])
        return frame, spans, logs

    def bounds(self, end):
        d = self.dataset
        return d.timestamp_epochs[end] - (d.history-1)*60, d.timestamp_epochs[end]+60

    def window(self, end):
        d = self.dataset
        lo, hi = self.bounds(end)
        metrics, spans, logs = self.case(d.case_ids[end])
        return (metrics.loc[(metrics.time >= lo) & (metrics.time < hi)].copy(),
                [s for s in spans if lo <= s["available"] < hi],
                [s for s in logs if lo <= s["available"] < hi])

    def training_reference(self, max_rows=2048, fit_ends=None):
        """Reference rows from observed healthy training input minutes only."""
        if max_rows < 10:
            raise ValueError("Reference requires at least ten rows")
        if fit_ends is not None and not set(fit_ends).issubset(self.data.train):
            raise ValueError("Reference may use training windows only")
        d = self.dataset
        current = {target: int(d.failure_targets[d._time_indices[stamp]])
                   for stamp, target in d.label_target_times.items() if d._label_valid.get(stamp, False)}
        minutes = defaultdict(set)
        fit_indices = (self.data.input_fit_indices if fit_ends is None else
                       sorted({i for end in fit_ends for i in range(end-d.history+1, end+1)}))
        for index in fit_indices:
            if current.get(d.timestamps[index]) == 0:
                minutes[d.case_ids[index]].add(d.timestamp_epochs[index])
        frames = []
        edges, operation_values = set(), defaultdict(list)
        for case_id, allowed in minutes.items():
            frame, spans, _ = self.case(case_id)
            subset = frame.loc[(np.floor(frame.time/60)*60).isin(allowed)].copy()
            subset["_case_key"] = case_id
            frames.append(subset)
            known = {(s["trace"], s["span"]): s for s in spans if int(s["available"]//60)*60 in allowed}
            for s in known.values():
                operation_values[(s["service"], s["operation"])].append(s["latency"])
                parent = known.get((s["trace"], s["parent"]))
                if parent and parent["service"] != s["service"]:
                    edges.add((parent["service"], s["service"]))
        reference = pd.concat(frames, ignore_index=True)
        columns = [c for c in reference if c not in {"time", "_case_key"} and c.rsplit("_", 1)[0] in self.service_index
                   and c.rsplit("_", 1)[1] in {"cpu", "mem", "diskio", "socket", "latency-90"}]
        columns = [c for c in columns if np.isfinite(reference[c]).sum() >= 10]
        if len(reference) > max_rows:
            reference = reference.iloc[np.linspace(0, len(reference)-1, max_rows, dtype=int)]
        self.reference_log_samples = pd.DataFrame(0.0, index=range(len(reference)),
                                                  columns=[s+"_log" for s in d.services])
        reference = reference.reset_index(drop=True)
        for case_id, rows in reference.groupby("_case_key", sort=False):
            self.reference_log_samples.loc[rows.index] = self.log_samples(case_id, rows.time.to_numpy()).to_numpy()
        reference = reference[columns]
        median = reference.median().fillna(0)
        reference = reference.fillna(median).reset_index(drop=True)
        if len(reference) < 10 or not columns:
            raise ValueError("Insufficient observed normal training reference")
        statistics = {key: (float(np.mean(values)), max(float(np.std(values)), 0.1))
                      for key, values in operation_values.items() if len(values) >= 2}
        return reference, median, edges, statistics

    def log_samples(self, case_id, times):
        _, _, logs = self.case(case_id)
        counts = defaultdict(int)
        for row in logs:
            if row["service"] in self.service_index:
                counts[(int(row["available"]), row["service"])] += 1
        return pd.DataFrame([[counts[(int(t), s)] for s in self.dataset.services] for t in times],
                             columns=[s+"_log" for s in self.dataset.services], dtype=float)

    def metric_pair(self, end, reference, median):
        frame, _, _ = self.window(end)
        current = frame.reindex(columns=reference.columns).fillna(median)
        if len(current) < 5:
            raise ValueError("Insufficient high-frequency observations in historical window")
        return reference.copy(), current.reset_index(drop=True)

    def invocation_frame(self, end, operation_stats):
        _, spans, _ = self.window(end)
        by_id = {(s["trace"], s["span"]): s for s in spans}
        rows = []
        for s in spans:
            parent = by_id.get((s["trace"], s["parent"]))
            # Parent outside the permitted history is not looked up from the full case.
            source = parent["service"] if parent else s["service"]
            stat = operation_stats.get((s["service"], s["operation"]))
            anomalous = bool(stat and abs(s["latency"]-stat[0]) > 3*stat[1]) or s["error"]
            rows.append({"trace_id": s["trace"], "source": source, "target": s["service"],
                         "predict": anomalous, "latency": s["latency"]})
        return pd.DataFrame(rows, columns=["trace_id", "source", "target", "predict", "latency"])

    def event_tokens(self, end, reference, max_events=32):
        d = self.dataset
        _, spans, logs = self.window(end)
        result = [{"metric": [], "log": [], "trace": []} for _ in d.services]
        values = d.features[end-d.history+1:end+1]
        # Scaler already fitted using training inputs only. These are explicit
        # deterministic alert tokens, not ground-truth root/fault labels.
        for i in range(len(d.services)):
            for k, name in enumerate(d.feature_columns):
                if name.startswith("metric_"):
                    for sign, mask in (("high", values[:, i, k] > 3), ("low", values[:, i, k] < -3)):
                        if mask.any():
                            result[i]["metric"].append(name+":"+sign)
        for s in spans:
            i = self.service_index.get(s["service"])
            if i is not None and len(result[i]["trace"]) < max_events:
                op = hashlib.sha256(s["operation"].encode()).hexdigest()[:16]
                result[i]["trace"].append("trace:"+op+(":error" if s["error"] else ":observed"))
        for log in logs:
            i = self.service_index.get(log["service"])
            if i is not None and len(result[i]["log"]) < max_events:
                template = hashlib.sha256(log["text"].encode()).hexdigest()[:16]
                result[i]["log"].append("log:"+template+":"+log["level"])
        return result
