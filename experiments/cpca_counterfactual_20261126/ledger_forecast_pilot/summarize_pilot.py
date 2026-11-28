"""Compare frozen pilot variants, check cohorts and evaluate alarm-gated RCA."""
from __future__ import annotations

import hashlib
import json
import statistics
import sys
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path('/Users/95penguin/project/NextMinuteRCA')
sys.path.insert(0, str(ROOT / 'src'))
from ocsid.audit import _load_case_labels

HERE = Path(__file__).resolve().parent


def row_key(row):
    return row['case'], int(row['minute'])


def reciprocal_rank(scores, root, services):
    if max((scores.get(s, 0.) for s in services), default=0.) <= 0:
        return 0.
    return 1. / sum(scores.get(s, 0.) >= scores.get(root, 0.) for s in services)


def evaluate(doc, features, labels, fpr_budget=None):
    entries, events = [], defaultdict(list)
    fold_results = []
    for fold in doc['folds']:
        threshold = fold['calibration']['threshold']
        if fpr_budget is not None:
            validation = fold['validation_predictions_for_fpr']
            negatives = sorted(p for p, y in zip(validation['probabilities'], validation['labels']) if not y)
            budget = int(len(negatives) * fpr_budget)
            # Lowest threshold allowing at most the budgeted negative alerts.
            threshold = math.nextafter(negatives[-budget - 1], math.inf)
        for pred in fold['test_predictions']:
            row = dict(pred, alert=pred['probability'] >= threshold)
            evidence = features['cases'].get(row['case'], {}).get(str(int(row['minute'])), {})
            row['rr'] = (reciprocal_rank(evidence.get('cpca', {}), row['root'], fold['services'])
                         if row.get('root') else 0.)
            entries.append(row)
            if row['failure_target'] and row.get('root') and row.get('event'):
                events[row['event']].append(row)
        fold_results.append({'fold': fold['fold'], 'ap': fold['average_precision'],
                             'f1': fold['calibration']['test_f1_at_validation_threshold']})
    tp = sum(r['alert'] and r['failure_target'] for r in entries)
    fp = sum(r['alert'] and not r['failure_target'] for r in entries)
    p = sum(r['failure_target'] for r in entries)
    n = len(entries) - p
    positives = [r for rows in events.values() for r in rows]
    event_rr, onset, any_alert = [], [], []
    onset_excluded = 0
    for rows in events.values():
        rows.sort(key=lambda r: r['minute'])
        case, event = rows[0]['case'], rows[0]['event']
        true_onset = min(m for m, label in labels[case].items()
                         if label['failure'] and label['event'] == event)
        onset_row = next((r for r in rows if r['minute'] == true_onset), None)
        if onset_row is not None:
            onset.append(int(onset_row['alert']))
        else:
            onset_excluded += 1
        alerted = [r for r in rows if r['alert']]
        any_alert.append(int(bool(alerted)))
        event_rr.append(alerted[0]['rr'] if alerted else 0.)
    return {'folds': fold_results, 'ap': statistics.mean(r['ap'] for r in fold_results),
            'f1': statistics.mean(r['f1'] for r in fold_results),
            'precision': tp / (tp + fp) if tp + fp else 0.,
            'recall': tp / p if p else 0., 'fpr': fp / n if n else 0.,
            'event_any_alert': statistics.mean(any_alert),
            'onset_recall': statistics.mean(onset) if onset else None,
            'onset_eligible_events': len(onset), 'onset_excluded_events': onset_excluded,
            'first_alert_event_mrr': statistics.mean(event_rr),
            'e2e_window_mrr': sum(r['rr'] * r['alert'] for r in positives) / len(positives),
            'offline_window_mrr': statistics.mean(r['rr'] for r in positives),
            'test_samples': len(entries), 'events': len(events)}


def main():
    output = {'protocol': 'single-seed development pilot, no hyperparameter search; '
                         'baseline trainer reused; validation-F1 gates, not matched FPR',
              'datasets': {}}
    lines = ['# 责任账本加入预测输入：首轮结果', '',
             '固定 seed42，deployment 五折、30 epochs；只改输入，复用原训练器。',
             '这是开发性试验，不是独立测试。告警阈值由每折验证 F1 选择，误报率未强行匹配。', '',
             '| 数据 | 输入 | AP | F1 | 误报率 | onset recall | 首次告警事件 MRR | 端到端窗口 MRR |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for dataset in ('ob', 'tt'):
        features = json.loads((HERE / f'features_{dataset}.json').read_text())
        labels = _load_case_labels(dataset)
        variants = {}
        baseline_cohort = None
        for variant in ('baseline', 'ledger', 'simple', 'mask'):
            path = HERE / f'{variant}_{dataset}_seed42.json'
            if not path.exists():
                continue
            doc = json.loads(path.read_text())
            if variant == 'baseline':
                val_path = HERE / f'validation_baseline_{dataset}.json'
                if val_path.exists():
                    saved_val = json.loads(val_path.read_text())['folds']
                    for f, v in zip(doc['folds'], saved_val):
                        assert f['fold'] == v['fold']
                        f['validation_predictions_for_fpr'] = v
            cohort = [(f['fold'], [(row_key(r), r['failure_target'], r['root'], r['event'])
                                   for r in f['test_predictions']]) for f in doc['folds']]
            if baseline_cohort is None:
                baseline_cohort = cohort
            elif cohort != baseline_cohort:
                raise ValueError(f'Cohort changed: {dataset}/{variant}')
            result = evaluate(doc, features, labels)
            if all('validation_predictions_for_fpr' in f for f in doc['folds']):
                result['validation_fpr05_gate'] = evaluate(doc, features, labels, fpr_budget=.05)
            variants[variant] = result
            lines.append(f'| {dataset.upper()} | {variant} | {result["ap"]:.4f} | '
                         f'{result["f1"]:.4f} | {result["fpr"]:.4f} | '
                         f'{result["onset_recall"]:.4f} | {result["first_alert_event_mrr"]:.4f} | '
                         f'{result["e2e_window_mrr"]:.4f} |')
        if variants:
            for name, result in variants.items():
                result['delta_ap_vs_baseline'] = result['ap'] - variants['baseline']['ap']
                result['delta_f1_vs_baseline'] = result['f1'] - variants['baseline']['f1']
        frozen_path = ROOT / f'experiments/cpca_counterfactual_20261126/inputs/prediction_{dataset}_corrected_v2_seed42.json'
        frozen = json.loads(frozen_path.read_text())
        if 'baseline' in variants:
            current = json.loads((HERE / f'baseline_{dataset}_seed42.json').read_text())
            max_diff = 0.
            for a, b in zip(current['folds'], frozen['folds']):
                assert a['fold'] == b['fold']
                assert [row_key(r) for r in a['test_predictions']] == [row_key(r) for r in b['test_predictions']]
                max_diff = max(max_diff, max(abs(r['probability'] - s['probability'])
                                            for r, s in zip(a['test_predictions'], b['test_predictions'])))
            variants['baseline']['max_probability_diff_vs_frozen'] = max_diff
        diagnostics = features['diagnostics']
        output['datasets'][dataset] = {'variants': variants, 'feature_diagnostics': {
            'cases': len(diagnostics), 'minutes_with_evidence': sum(len(v) for v in features['cases'].values()),
            'scored_requests': sum(v.get('scored_requests', 0) for v in diagnostics.values()),
            'invalid_requests': sum(v.get('invalid_requests', 0) for v in diagnostics.values()),
            'availability_changed_requests': sum(v.get('availability_differs_from_entry_minute', 0) for v in diagnostics.values())}}
    lines += ['', '## 固定验证集误报预算（5%）', '',
              '预算只在验证集约束；测试误报率可能因分布变化不同，不宣称测试误报率完全匹配。', '',
              '| 数据 | 输入 | 测试误报率 | onset recall | 首次告警事件 MRR | 端到端窗口 MRR |',
              '|---|---|---:|---:|---:|---:|']
    for dataset, data in output['datasets'].items():
        for variant, result in data['variants'].items():
            r = result.get('validation_fpr05_gate')
            if r:
                lines.append(f'| {dataset.upper()} | {variant} | {r["fpr"]:.4f} | '
                             f'{r["onset_recall"]:.4f} | {r["first_alert_event_mrr"]:.4f} | '
                             f'{r["e2e_window_mrr"]:.4f} |')
    lines += ['', 'CPCA 对所有方法相同；端到端差异来自告警集合变化。',
              '特征使用所有原始 span 的完成时间分桶，比原入口完成时间策略更保守。',
              'onset 为原始标签中事件的第一个正例输入分钟；不在评价集合的 onset 单独计数。',
              'baseline 与历史冻结预测的最大概率差、逐折结果和样本核验见 RESULTS.json。']
    output['source_hashes'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in (ROOT / 'src/ocsid/training.py', ROOT / 'src/models/ocsid.py')}
    (HERE / 'RESULTS.json').write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
    (HERE / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
