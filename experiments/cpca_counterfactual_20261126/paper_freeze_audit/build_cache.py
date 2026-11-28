"""Full raw-trace coverage audit and compact reference-resampling cache.

Does not change repository parsers, attribution defaults or forecast cohorts.
All raw trace IDs are audited, not only the capped scoring sample. Request
selection and baseline normal references reproduce the frozen audit protocol.
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
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from audit_paths import ROOT, EXPERIMENT
sys.path.insert(0, str(ROOT / 'src'))
from ocsid import cpca

HERE = Path(__file__).resolve().parent


def epoch_minute(value):
    return int(value // 60_000_000) * 60


def tree_flags(component):
    ids = {s.sid: s for s in component}
    children = defaultdict(list)
    start_violation = end_violation = overlap = 0.
    for s in component:
        parent = ids.get(s.parent)
        if parent is not None:
            children[s.parent].append(s)
            start_violation = max(start_violation, parent.start - s.start)
            end_violation = max(end_violation, s.end - parent.end)
    for values in children.values():
        values.sort(key=lambda s: s.start)
        latest = values[0].end
        for s in values[1:]:
            overlap = max(overlap, latest - s.start)
            latest = max(latest, s.end)
    return {'sibling_overlap_us': overlap, 'before_parent_us': start_violation,
            'after_parent_us': end_violation}


def parse_with_reason(spans, dataset):
    if not spans:
        return None, None, 'empty', {}
    if any(not math.isfinite(s.start) or not math.isfinite(s.duration) or s.duration < 0 for s in spans):
        return None, None, 'invalid_numeric_span', {}
    ids = {s.sid: s for s in spans}
    if len(ids) != len(spans):
        return None, None, 'duplicate_span_id', {}
    entries = [s for s in spans if s.parent not in ids and cpca.is_entry(s, dataset)]
    if not entries:
        name = 'entry_not_a_root' if any(cpca.is_entry(s, dataset) for s in spans) else 'no_recognized_business_entry'
        return None, None, name, {}
    if len(entries) > 1:
        return None, None, 'multiple_business_entries', {}
    component = cpca.entry_component(spans, dataset)
    if component is None:
        return None, None, 'entry_component_structure_failure', {}
    parsed = cpca.decompose_tree(component, dataset, gap_policy='post',
                                 clock_tolerance_us=1000., tree_policy='strict')
    if parsed is not None:
        return parsed, component, None, {}
    flags = tree_flags(component)
    # These are observable temporal patterns, not diagnoses of physical cause.
    if flags['sibling_overlap_us'] > 0:
        reason = 'rejected_with_sibling_overlap'
    elif flags['before_parent_us'] > 0 or flags['after_parent_us'] > 0:
        reason = 'rejected_with_parent_child_boundary_violation'
    else:
        reason = 'other_serial_or_alignment_failure'
    return None, component, reason, flags


def add_coverage(counter, spans, parsed, component, reason, flags, raw_row_count):
    counter['raw_trace_ids'] += 1
    counter['raw_rows'] += raw_row_count
    counter['parsed_numeric_rows'] += len(spans)
    if any(cpca.is_entry(s, counter['_dataset']) for s in spans):
        counter['traces_with_any_recognized_entry_span'] += 1
    if reason is not None:
        counter['rejected_' + reason] += 1
        for name, value in flags.items():
            counter['rejected_flag_' + name.replace('_us', '')] += int(value > 0)
        return
    by_id, children, _, local, _, _ = parsed
    counter['accepted_trace_ids'] += 1
    counter['accepted_raw_spans'] += len(spans)
    counter['retained_spans'] += len(by_id)
    counter['omitted_spans'] += len(spans) - len(by_id)
    counter['accepted_complete' if len(spans) == len(by_id) else 'accepted_partial'] += 1
    changed = shifted = clipped = 0
    raw = {s.sid: s for s in component}
    for sid, s in by_id.items():
        shift = abs(s.start - raw[sid].start)
        clip = abs(s.duration - raw[sid].duration)
        shifted += shift > 0
        clipped += clip > 0
        changed += shift > 0 or clip > 0
        counter['max_total_start_displacement_us'] = max(counter['max_total_start_displacement_us'], shift)
        counter['max_duration_change_us'] = max(counter['max_duration_change_us'], clip)
        counter['span_total_displacement_over_1ms'] += shift > 1000.
        if children.get(sid):
            counter['internal_spans'] += 1
            counter['internal_spans_with_nonzero_local_over_1us'] += local[sid] > 1.
        else:
            counter['leaf_spans'] += 1
    counter['accepted_with_clock_changes'] += bool(changed)
    counter['shifted_spans'] += shifted
    counter['duration_changed_spans'] += clipped


def score_cache(cache, references, mode):
    """Reproduce per-request service clipping, then operation normalization."""
    keep = cache['selection'][:, mode].astype(bool)
    denom = np.bincount(cache['request_group'], weights=cache['request_duration'] * keep,
                        minlength=int(cache['n_groups'][0]))
    count_by_window = np.bincount(cache['group_window'][denom > 0],
                                  minlength=int(cache['n_windows'][0]))
    n_services = int(cache['n_services'][0])
    values = references[cache['component_key']]
    gain = cache['component_observed'] - np.where(np.isfinite(values), values, cache['component_observed'])
    pair_delta = np.maximum(np.bincount(cache['component_pair'], weights=gain,
                                        minlength=len(cache['pair_request'])), 0.)
    pair_delta *= keep[cache['pair_request']]
    group_scores = np.bincount(cache['pair_group_service'], weights=pair_delta,
                               minlength=int(cache['n_groups'][0]) * n_services).astype(float).reshape(-1, n_services)
    group_scores /= np.maximum(denom[:, None], 1e-30)
    window_scores = np.zeros((int(cache['n_windows'][0]), n_services))
    np.add.at(window_scores, cache['group_window'], group_scores)
    window_scores /= np.maximum(count_by_window[:, None], 1)
    return window_scores


def reference_draws(cache, repetitions, seed):
    """Resample whole healthy minute blocks jointly across all reference keys."""
    blocks = int(cache['n_blocks'][0])
    rng = np.random.default_rng(seed)
    if blocks:
        draws = rng.multinomial(blocks, np.full(blocks, 1. / blocks), size=repetitions)
    else:
        draws = np.zeros((repetitions, 0), dtype=int)
    result = np.full((repetitions, len(cache['key_offsets']) - 1), np.nan)
    missing = 0
    for key, (start, end) in enumerate(zip(cache['key_offsets'][:-1], cache['key_offsets'][1:])):
        if start == end:
            continue
        values = cache['normal_values'][start:end]
        weights = draws[:, cache['normal_blocks'][start:end]]
        cumulative = np.cumsum(weights, axis=1)
        total = cumulative[:, -1]
        ok = total > 0
        low = ((total - 1) // 2)[:, None]
        high = (total // 2)[:, None]
        lo_index = np.argmax(cumulative > low, axis=1)
        hi_index = np.argmax(cumulative > high, axis=1)
        result[ok, key] = (values[lo_index[ok]] + values[hi_index[ok]]) / 2
        missing += int((~ok).sum())
    return result, {'warmup_minute_blocks': blocks,
                    'missing_resampled_key_references': missing,
                    'reference_keys_with_original_support': int(np.diff(cache['key_offsets']).astype(bool).sum())}


def build_case(archive, dataset, member, case_id, windows, services, output_dir, repetitions):
    began = time.monotonic()
    traces = defaultdict(list)
    raw_counts = Counter()
    malformed = Counter()
    with archive.open(member + '/traces.csv') as raw:
        for row in csv.DictReader(io.TextIOWrapper(raw, encoding='utf-8-sig', newline='')):
            trace = row['traceID']
            raw_counts[trace] += 1
            try:
                start, duration = float(row['startTime']), float(row['duration'])
            except (ValueError, TypeError):
                malformed[trace] += 1
                continue
            if not math.isfinite(start) or not math.isfinite(duration) or duration < 0:
                malformed[trace] += 1
                continue
            traces[trace].append(cpca.Span(row['spanID'], row['parentSpanID'],
                cpca.normalize_service(row['serviceName']), row['operationName'], start, duration))
    cutoff = min(epoch_minute(max(s.end for s in spans)) for spans in traces.values()) + 600
    counts = {'all': Counter(_dataset=dataset), 'positive_cohort': Counter(_dataset=dataset)}
    key_index = {}
    normal = defaultdict(list)
    chosen = []
    full_counts = Counter()
    complete_counts = Counter()
    budget = Counter()
    positive = {int(w['minute']) for w in windows}

    def key_id(key):
        if key not in key_index:
            key_index[key] = len(key_index)
        return key_index[key]

    def components(parsed):
        ids, _, _, local, pre, post = parsed
        for span in ids.values():
            yield span.service, key_id(('local', span.service, span.operation)), local[span.sid]
            parent = ids.get(span.parent)
            if parent is not None:
                edge = (parent.service, parent.operation, span.service, span.operation)
                yield span.service, key_id(('pre', *edge)), pre[span.sid]
                yield span.service, key_id(('post', *edge)), post[span.sid]

    for trace, raw_count in raw_counts.items():
        spans = traces.get(trace, [])
        entry = next((s for s in spans if cpca.is_entry(s, dataset)), None)
        minute = epoch_minute(entry.end if entry else max((s.end for s in spans), default=0))
        in_cohort = minute in positive and minute >= cutoff
        op = (entry.service, entry.operation) if entry else None
        budget_examined = in_cohort and entry is not None and full_counts[minute, op] < 100
        if budget_examined:
            budget['examined'] += 1
        parsed, component, reason, flags = parse_with_reason(spans, dataset)
        for scope in ('all', 'positive_cohort') if in_cohort else ('all',):
            add_coverage(counts[scope], spans, parsed, component, reason, flags, raw_count)
            counts[scope]['malformed_rows'] += malformed[trace]
            counts[scope]['traces_with_malformed_rows'] += bool(malformed[trace])
        if parsed is None:
            if budget_examined:
                budget['invalid'] += 1
            continue
        root = parsed[2]
        if minute < cutoff:
            for _, key, value in components(parsed):
                normal[key].append((value, minute))
            continue
        if root.duration <= 0:
            if budget_examined:
                budget['invalid'] += 1
            continue
        # Keep the frozen audit's original entry completion bin.
        if not in_cohort:
            continue
        op = (root.service, root.operation)
        group = (minute, op)
        full = full_counts[group] < 100
        complete = len(spans) == len(parsed[0])
        refill = complete and complete_counts[group] < 100
        if full:
            full_counts[group] += 1
            budget['accepted'] += 1
        if refill:
            complete_counts[group] += 1
        if not (full or refill):
            continue
        raw_ids = {s.sid: s for s in component}
        no_clock = all(s.start == raw_ids[sid].start and s.duration == raw_ids[sid].duration
                       for sid, s in parsed[0].items())
        chosen.append({'group': group, 'duration': root.duration,
                       'components': list(components(parsed)),
                       'selection': (full, refill, full and complete, full and no_clock)})

    service_index = {s: i for i, s in enumerate(services)}
    window_index = {int(w['minute']): i for i, w in enumerate(windows)}
    groups = list(dict.fromkeys(r['group'] for r in chosen))
    group_index = {g: i for i, g in enumerate(groups)}
    comp_obs, comp_key, comp_pair = [], [], []
    pair_req, pair_group_service = [], []
    for req_id, request in enumerate(chosen):
        pairs = {}
        for service, key, observed in request['components']:
            if service not in service_index:
                raise ValueError(f'Unknown service {service}')
            if service not in pairs:
                pairs[service] = len(pair_req)
                pair_req.append(req_id)
                pair_group_service.append(group_index[request['group']] * len(services) + service_index[service])
            comp_obs.append(observed)
            comp_key.append(key)
            comp_pair.append(pairs[service])
    blocks = sorted({minute for values in normal.values() for _, minute in values})
    block_index = {m: i for i, m in enumerate(blocks)}
    values, block_ids, offsets = [], [], [0]
    baseline_ref = []
    for key in range(len(key_index)):
        pool = sorted(normal[key])
        values.extend(v for v, m in pool)
        block_ids.extend(block_index[m] for v, m in pool)
        offsets.append(len(values))
        baseline_ref.append(statistics.median(v for v, m in pool) if pool else math.nan)
    cache = {'normal_values': np.asarray(values), 'normal_blocks': np.asarray(block_ids, dtype=np.int32),
             'key_offsets': np.asarray(offsets, dtype=np.int64), 'baseline_reference': np.asarray(baseline_ref),
             'component_observed': np.asarray(comp_obs), 'component_key': np.asarray(comp_key, dtype=np.int32),
             'component_pair': np.asarray(comp_pair, dtype=np.int32), 'pair_request': np.asarray(pair_req, dtype=np.int32),
             'pair_group_service': np.asarray(pair_group_service, dtype=np.int32),
             'request_group': np.asarray([group_index[r['group']] for r in chosen], dtype=np.int32),
             'request_duration': np.asarray([r['duration'] for r in chosen]),
             'selection': np.asarray([r['selection'] for r in chosen], dtype=np.int8).reshape(-1, 4),
             'group_window': np.asarray([window_index[g[0]] for g in groups], dtype=np.int32),
             'n_groups': np.asarray([len(groups)]), 'n_windows': np.asarray([len(windows)]),
             'n_services': np.asarray([len(services)]), 'n_blocks': np.asarray([len(blocks)])}
    scores = {name: score_cache(cache, cache['baseline_reference'], mode)
              for mode, name in enumerate(('full', 'complete_refill', 'complete_shared_cap', 'no_clock_shared_cap'))}
    seed = 20261126 + int(hashlib.sha256(case_id.encode()).hexdigest()[:8], 16)
    if windows:
        draws, draw_diag = reference_draws(cache, repetitions, seed)
        sampled = np.stack([score_cache(cache, reference, 0) for reference in draws])
    else:
        sampled = np.zeros((repetitions, 0, len(services)))
        draw_diag = {'warmup_minute_blocks': len(blocks)}
    for c in counts.values():
        del c['_dataset']
    metadata = {'case_id': case_id, 'cutoff': cutoff, 'reference_blocks': blocks,
                'windows': windows, 'coverage': counts, 'budget': budget,
                'sampling': draw_diag, 'seconds': time.monotonic() - began,
                'unique_reference_and_query_keys': len(key_index),
                'modes': list(scores), 'npz_format': 1}
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / (case_id + '.npz'), **cache,
                        **{'scores_' + k: v for k, v in scores.items()}, sampled_scores=sampled)
    (output_dir / (case_id + '.json')).write_text(json.dumps(metadata, ensure_ascii=False) + '\n')
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=('ob', 'tt'), required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--repetitions', type=int, default=100)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    cohort = json.loads((EXPERIMENT / f'inputs/prediction_{args.dataset}_corrected_v2_seed42.json').read_text())
    services = cohort['folds'][0]['services']
    windows = defaultdict(list)
    cases = set()
    for fold in cohort['folds']:
        for row in fold['test_predictions']:
            cases.add(row['case'])
            if row['failure_target'] and row['root']:
                windows[row['case']].append({k: row[k] for k in ('minute', 'root', 'event')})
    # Coverage includes all panel cases, including cases with no forecast rows.
    panel = ROOT / f'artifacts/re2_{args.dataset}_panel_fair_v1/case_minutes.csv'
    cases.update(row['case_id'] for row in csv.DictReader(panel.open()))
    prefix = 'RE2-' + args.dataset.upper()
    all_metadata = {}
    with zipfile.ZipFile(ROOT / f'data/RCAEval/{prefix}.zip') as archive:
        members = sorted(n[:-len('/traces.csv')] for n in archive.namelist()
                         if n.startswith(prefix + '/') and n.endswith('/traces.csv')
                         and prefix + '_' + n[len(prefix)+1:-len('/traces.csv')].replace('/', '_') in cases)
        if args.limit:
            members = members[:args.limit]
        for index, member in enumerate(members, 1):
            case = prefix + '_' + member[len(prefix)+1:].replace('/', '_')
            existing = args.output_dir / (case + '.json')
            if existing.exists():
                meta = json.loads(existing.read_text())
            else:
                meta = build_case(archive, args.dataset, member, case,
                                   sorted(windows[case], key=lambda r: r['minute']),
                                   services, args.output_dir, args.repetitions)
            all_metadata[case] = meta
            print(f'{args.dataset} {index}/{len(members)} {case}: {meta["seconds"]:.2f}s', flush=True)
    report = {'dataset': args.dataset, 'repetitions': args.repetitions, 'services': services,
              'protocol': 'frozen median/replace/post/uniform; first10; full-cohort evaluation; '
                          'healthy minute-block bootstrap; complete modes share full normal reference',
              'cases': all_metadata,
              'source_hashes': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in (ROOT/'src/ocsid/cpca.py', ROOT/'src/ocsid/clocks.py')}}
    (args.output_dir / 'index.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(f'Finished {args.dataset}', flush=True)


if __name__ == '__main__':
    main()
