"""Descriptive event-aligned ledger dynamics; labels used only after extraction."""
import json
import statistics
from collections import defaultdict
from pathlib import Path

from summarize_pilot import _load_case_labels

HERE = Path(__file__).resolve().parent


def main():
    output = {'note': 'Descriptive only, not evidence of intervention effects or predictor faithfulness.',
              'datasets': {}}
    for ds in ('ob', 'tt'):
        features = json.loads((HERE / f'features_{ds}.json').read_text())
        events = defaultdict(list)
        for case, minutes in _load_case_labels(ds).items():
            for minute, row in minutes.items():
                if row['failure'] and row['event']:
                    events[case, row['event']].append((minute, row['root']))
        aligned = defaultdict(list)
        paired = []
        onset_pairs = []
        for (case, event), rows in events.items():
            roots = {root for _, root in rows if root}
            if len(roots) != 1:
                continue
            root = next(iter(roots))
            onset = min(m for m, _ in rows)
            values = {}
            for offset in range(-3, 4):
                evidence = features['cases'].get(case, {}).get(str(onset + offset * 60))
                if evidence is None:
                    continue
                score = evidence['cpca'].get(root, 0.)
                values[offset] = score
                aligned[offset].append({'score': score, 'root_observed': root in evidence['observed']})
            before = [values[o] for o in (-3, -2, -1) if o in values]
            after = [values[o] for o in (1, 2, 3) if o in values]
            if 0 in values and 1 in values:
                onset_pairs.append((values[0], values[1]))
            if before and after:
                paired.append({'case': case, 'event': event, 'root': root,
                               'before': statistics.mean(before), 'after': statistics.mean(after)})
        output['datasets'][ds] = {
            'offset_unit': 'decision minutes relative to first next-minute-positive decision; target is t+1',
            'offsets': {offset: {'events_with_any_evidence': len(rows),
                                 'root_observed_events': sum(r['root_observed'] for r in rows),
                                 'mean_root_responsibility': statistics.mean(r['score'] for r in rows),
                                 'median_root_responsibility': statistics.median(r['score'] for r in rows)}
                        for offset, rows in sorted(aligned.items())},
            'paired_events': len(paired),
            'root_score_increased': sum(r['after'] > r['before'] for r in paired),
            'decision_to_next_minute_pairs': {
                'events': len(onset_pairs),
                'median_at_decision': statistics.median(a for a, b in onset_pairs),
                'median_at_next_minute': statistics.median(b for a, b in onset_pairs),
                'median_paired_change': statistics.median(b - a for a, b in onset_pairs),
                'increased_events': sum(b > a for a, b in onset_pairs)},
            'pairs': paired}
    (HERE / 'TIMING_DIAGNOSTIC.json').write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
    for ds, result in output['datasets'].items():
        print(ds, result['offsets'])


if __name__ == '__main__':
    main()
