"""Label-free causal CPCA features for every minute, not only positives.

Uses the repository's fixed median/replace/post/uniform ledger. Conservative
availability is the last raw span completion of the entire trace. Warm-up
features stay unavailable; no injection time or outcome labels are read.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import statistics
import sys
import time
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path('/Users/95penguin/project/NextMinuteRCA')
sys.path.insert(0, str(ROOT / 'src'))
from ocsid import cpca
from ocsid.aggregation import aggregate_operations
from ocsid.audit import _collect_normal, _robust_stats
from ocsid.config import ATTRIBUTION_DEFAULTS as DEFAULTS


def minute_of(spans):
    return int(max(s.end for s in spans) // 60_000_000) * 60


def build_case(archive, dataset, member, allowed_minutes):
    began = time.monotonic()
    traces = defaultdict(list)
    diag = defaultdict(int)
    with archive.open(member + '/traces.csv') as raw:
        for row in csv.DictReader(io.TextIOWrapper(raw, encoding='utf-8-sig', newline='')):
            try:
                start, duration = float(row['startTime']), float(row['duration'])
            except (ValueError, TypeError):
                diag['malformed_rows'] += 1
                continue
            if not math.isfinite(start) or not math.isfinite(duration) or duration < 0:
                diag['malformed_rows'] += 1
                continue
            traces[row['traceID']].append(cpca.Span(
                row['spanID'], row['parentSpanID'], cpca.normalize_service(row['serviceName']),
                row['operationName'], start, duration))
    if not traces:
        raise ValueError(f'No traces: {member}')
    cutoff = min(minute_of(spans) for spans in traces.values()) + 600
    buckets = {k: defaultdict(list) for k in ('local', 'pre', 'post', 'entry')}
    normal_minutes = {}
    kwargs = dict(gap_policy='post', clock_tolerance_us=1000., tree_policy='entry-component')
    for spans in traces.values():
        completed = minute_of(spans)
        if completed < cutoff:
            _collect_normal(spans, buckets, dataset, completed, normal_minutes, **kwargs)
    normal_local = {k: statistics.median(v) for k, v in buckets['local'].items()}
    normal_edge = {k: (statistics.median(buckets['pre'][k]),
                       statistics.median(buckets['post'][k])) for k in buckets['pre']}
    healthy_service = defaultdict(list)
    for per_service in normal_minutes.values():
        for service, values in per_service.items():
            healthy_service[service].append(statistics.mean(values['dur']))
    normal_z = {s: _robust_stats(v) for s, v in healthy_service.items()}
    states = defaultdict(dict)
    for spans in traces.values():
        minute = minute_of(spans)
        if minute < cutoff or minute not in allowed_minutes:
            continue
        entries = [s for s in spans if cpca.is_entry(s, dataset)]
        if not entries:
            diag['no_entry'] += 1
            continue
        # Match the official first-100-valid-requests budget and source order.
        entry = entries[0]
        op = (entry.service, entry.operation)
        if op in states[minute] and states[minute][op]['count'] >= 100:
            continue
        diag['examined_requests'] += 1
        parsed = cpca.decompose_tree(spans, dataset, **kwargs)
        scored = cpca.score_decomposition(parsed, normal_local, normal_edge,
                                          dataset, restoration='replace')
        if scored is None:
            diag['invalid_requests'] += 1
            continue
        root, deltas, _ = scored
        if int(root.end // 60_000_000) * 60 != minute:
            diag['availability_differs_from_entry_minute'] += 1
        state = states[minute].setdefault(op, {
            'count': 0, 'duration': 0., 'delta': defaultdict(float),
            'durations': defaultdict(list)})
        state['count'] += 1
        state['duration'] += root.duration
        for service, delta in deltas.items():
            state['delta'][service] += delta
        for span in parsed[0].values():
            state['durations'][span.service].append(span.duration)
        diag['scored_requests'] += 1
        diag['omitted_spans'] += len(spans) - len(parsed[0])
    output = {}
    operations = set()
    for minute, ops in states.items():
        if not ops:
            continue
        op_scores = {op: {s: d / st['duration'] for s, d in st['delta'].items()}
                     for op, st in ops.items()}
        scores, _ = aggregate_operations(op_scores, {op: st['count'] for op, st in ops.items()},
                                         mode='uniform')
        durations = defaultdict(list)
        for op, state in ops.items():
            operations.add(op)
            for service, values in state['durations'].items():
                durations[service].extend(values)
        simple = {}
        for service, values in durations.items():
            if service in normal_z:
                center, scale = normal_z[service]
                simple[service] = max(-10., min(10., (statistics.mean(values) - center) / scale))
        output[minute] = {'cpca': scores, 'simple': simple, 'observed': sorted(scores),
                          'operations': [list(op) for op in ops]}
    return output, {'cutoff': cutoff, 'trace_count': len(traces),
                    'normal_local_keys': len(normal_local), 'normal_edge_keys': len(normal_edge),
                    'operations': [list(op) for op in sorted(operations)],
                    'seconds': time.monotonic() - began, **dict(diag)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=('ob', 'tt'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists')
    panel = ROOT / f'artifacts/re2_{args.dataset}_panel_fair_v1'
    case_minutes = defaultdict(set)
    for row in csv.DictReader((panel / 'case_minutes.csv').open()):
        case_minutes[row['case_id']].add(int(datetime.fromisoformat(row['timestamp_utc']).timestamp()))
    prefix = f'RE2-{args.dataset.upper()}'
    report = {'dataset': args.dataset, 'source': str(ROOT),
              'protocol': 'no labels/injection time; first10 median; replace/post/uniform; '
                          'last-raw-span completion availability; pre-cutoff unavailable',
              'cases': {}, 'diagnostics': {}}
    with zipfile.ZipFile(ROOT / f'data/RCAEval/{prefix}.zip') as archive:
        members = sorted(n[:-len('/traces.csv')] for n in archive.namelist()
                         if n.startswith(prefix + '/') and n.endswith('/traces.csv'))
        # Archives may contain an auxiliary directory that is not a panel case.
        report['excluded_archive_members'] = [m for m in members if
            prefix + '_' + m[len(prefix) + 1:].replace('/', '_') not in case_minutes]
        members = [m for m in members if
            prefix + '_' + m[len(prefix) + 1:].replace('/', '_') in case_minutes]
        if args.limit:
            members = members[:args.limit]
        for index, member in enumerate(members, 1):
            case = prefix + '_' + member[len(prefix) + 1:].replace('/', '_')
            if case not in case_minutes:
                raise ValueError(f'Unknown case {case}')
            output, diag = build_case(archive, args.dataset, member, case_minutes[case])
            report['cases'][case] = output
            report['diagnostics'][case] = diag
            print(f'{args.dataset} {index}/{len(members)} {case}: {len(output)} minutes, '
                  f'{diag["seconds"]:.2f}s', flush=True)
    report['source_hashes'] = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in (ROOT / 'src/ocsid/cpca.py', ROOT / 'src/ocsid/clocks.py')}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False) + '\n')
    print(f'Saved {args.output}', flush=True)


if __name__ == '__main__':
    main()
