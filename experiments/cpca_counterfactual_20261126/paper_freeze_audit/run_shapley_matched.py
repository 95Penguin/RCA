"""Author ShapleyIQ core with aligned input and explicit aggregation controls.

The original adapter is preserved beside this script. This wrapper aligns the
accepted component before handing it to the author core, and records both the
native per-trace average and a CPCA-matched operation-equal aggregation. No
author attribution formula is changed. Both no-cache and warm-up mean variants
are retained; comparisons must prioritize the warm-up-equipped variant.
"""
import argparse
import hashlib
import json
import time
import sys
from collections import defaultdict
from pathlib import Path

from audit_paths import ROOT, EXPERIMENT
sys.path.insert(0,str(ROOT/'src'))
import shapley_adapter_original as adapter
adapter.ROOT=ROOT

HERE = Path(__file__).resolve().parent
EXP = EXPERIMENT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=('ob', 'tt'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    adapter.DATASET = args.dataset
    cohort = json.loads((EXP / f'inputs/prediction_{args.dataset}_corrected_v2_seed42.json').read_text())
    positive = defaultdict(dict)
    for fold in cohort['folds']:
        for row in fold['test_predictions']:
            if row['failure_target'] and row['root']:
                positive[row['case']][int(row['minute'])] = row
    original_component = adapter.cpca.entry_component
    original_loader = adapter.load_upstream_class

    def aligned_component(spans, dataset):
        component = original_component(spans, dataset)
        if component is None:
            return None
        parsed = adapter.cpca.decompose_tree(component, dataset, tree_policy='strict', clock_tolerance_us=1000.)
        return [parsed[0][s.sid] for s in component] if parsed is not None else None

    adapter.cpca.entry_component = aligned_component
    active = {}

    def recording_loader():
        base = original_loader()
        counter = [0]
        class Recording(base):
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                self.variant = 'shapleyiq_default' if counter[0] == 0 else 'shapleyiq_warmup10'
                counter[0] += 1

            def analyze_trace(self, trace):
                result = super().analyze_trace(trace)
                ids = {s['spanID'] for s in trace}
                root = next(s for s in trace if s['parentSpanID'] not in ids)
                minute = int((root['TimeStamp'] + root['Duration']) // 60_000_000) * 60
                op = (root['serviceName'], root['rpc'])
                key = (minute, op)
                if self.variant == 'shapleyiq_default':
                    active['denominator'][key] += root['Duration']
                    active['count'][key] += 1
                for operation, value in result.items():
                    service = operation.split(':', 1)[0]
                    if service in adapter.SERVICES[args.dataset]:
                        active['sums'][self.variant][key][service] += value
                return result
        return Recording

    adapter.load_upstream_class = recording_loader
    report = {'dataset': args.dataset, 'implementation_version': 'author-core-aligned-operation-control-v1',
              'protocol': 'frozen cohort/full services/worst ties; original author-core attribution; '
                          '1ms aligned entry component input; max100 per operation/minute; '
                          'native trace-average and operation-uniform normalized contribution controls',
              'source_sha256': hashlib.sha256((HERE/'third_party/shapley_value_rca.py').read_bytes()).hexdigest(),
              'cases': {}}
    if args.output.exists():
        report = json.loads(args.output.read_text())
    began = time.monotonic()
    for index, case in enumerate(sorted(positive), 1):
        if case in report['cases']:
            continue
        active.clear()
        active.update(denominator=defaultdict(float), count=defaultdict(int),
                      sums={m: defaultdict(lambda: defaultdict(float)) for m in ('shapleyiq_default', 'shapleyiq_warmup10')})
        result = adapter.evaluate_case(args.dataset, case, set(positive[case]), positive[case], 100)
        for window in result['windows']:
            minute = window['minute']
            ops = [key for key in active['count'] if key[0] == minute]
            for method in ('shapleyiq_default', 'shapleyiq_warmup10'):
                combined = defaultdict(float)
                for key in ops:
                    for service, value in active['sums'][method][key].items():
                        combined[service] += value / active['denominator'][key] / len(ops)
                window['scores_by_method'][method + '_uniform'] = dict(combined)
            window['operation_request_counts'] = {str(k[1]): active['count'][k] for k in ops}
        report['cases'][case] = result
        report['elapsed_seconds'] = time.monotonic() - began
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False) + '\n')
        print(args.dataset, index, len(positive), case, result['diagnostics']['accepted_requests'],
              round(report['elapsed_seconds'], 1), flush=True)
    report['metrics'] = {m: adapter.evaluate_attribution(report, cohort, m)
                         for m in ('shapleyiq_default', 'shapleyiq_warmup10',
                                   'shapleyiq_default_uniform', 'shapleyiq_warmup10_uniform')}
    args.output.write_text(json.dumps(report, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
