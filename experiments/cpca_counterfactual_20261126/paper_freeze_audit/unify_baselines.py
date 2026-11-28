"""Re-evaluate saved baseline scores on the frozen cohort, respecting logits.

An available all-negative logit vector is valid ranking evidence. A common
per-window translation preserves both window and event score order, while
allowing the CPCA-oriented evaluator to use an explicit unavailable sentinel.
Missing/null/available=False scores remain abstentions. No models are retrained.
"""
import hashlib
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, f1_score

from build_cache import ROOT, EXPERIMENT
from ocsid.evaluation import evaluate_attribution
from ocsid.features import load_operation_table

HERE = Path(__file__).resolve().parent
BASE = ROOT / 'artifacts/baselines/comparison_2026-11-19_seed42'


def translate_available_scores(vector, services, available=True):
    if vector is None or available is False:
        return {}
    if len(vector) != len(services) or not all(math.isfinite(float(v)) for v in vector):
        raise ValueError('Invalid baseline score schema')
    shift = max(0., 1. - min(vector))
    return {s: float(v) + shift for s, v in zip(services, vector)}


def paired_delta(a, b, event_cases):
    assert set(a['event_ranks']) == set(b['event_ranks'])
    names = sorted(a['event_ranks'])
    rr = lambda r: 0. if r is None else 1. / r
    delta = np.array([rr(a['event_ranks'][e]) - rr(b['event_ranks'][e]) for e in names])
    rng = np.random.default_rng(42)
    draws = delta[rng.integers(0, len(delta), size=(10000, len(delta)))].mean(axis=1)
    cases=sorted({event_cases[e] for e in names})
    case_sums=np.array([sum(delta[i] for i,e in enumerate(names) if event_cases[e]==c) for c in cases])
    case_counts=np.array([sum(event_cases[e]==c for e in names) for c in cases])
    sampled_cases=rng.integers(0,len(cases),size=(10000,len(cases)))
    clustered=case_sums[sampled_cases].sum(axis=1)/case_counts[sampled_cases].sum(axis=1)
    return {'delta_event_mrr': float(delta.mean()), 'event_bootstrap_interval95': np.quantile(draws,[.025,.975]).tolist(),
            'case_cluster_bootstrap_interval95':np.quantile(clustered,[.025,.975]).tolist(),
            'case_clusters':len(cases),
            'improved_events': int((delta > 0).sum()), 'worse_events': int((delta < 0).sum()),
            'tied_events': int((delta == 0).sum()), 'note': 'event-block descriptive bootstrap on development data, not selection-free significance'}


def supervised_priors(cohort,ds):
    """Cheap root/operation priors, fit on training events of each fold only."""
    table,_=load_operation_table(ROOT/f'artifacts/re2_{ds}_ocsid_operation_features_v1.csv')
    services=cohort['folds'][0]['services']
    outputs={name:{'cases':{}} for name in ('train_event_root_prior','train_event_operation_root_prior')}
    for fold in cohort['folds']:
        f=fold['fold']
        training_rows=[r for g in cohort['folds'] if g['fold'] not in (f,(f+1)%5) for r in g['test_predictions']]
        assert len(training_rows)==fold['train_samples'], 'Training cohort mismatch for priors'
        events={}; operations=defaultdict(set)
        for r in training_rows:
            if not r['failure_target'] or not r['root']:
                continue
            events[r['event']]=r['root']
            operations[r['event']].update(table.get((r['case'],int(r['minute'])),{}))
        counts=defaultdict(float); by_op=defaultdict(lambda:defaultdict(float))
        for event,root in events.items():
            counts[root]+=1
            for op in operations[event]:
                by_op[op][root]+=1
        global_scores={s:counts[s]/len(events) for s in services}
        for r in fold['test_predictions']:
            if not r['failure_target'] or not r['root']:
                continue
            ops=list(table.get((r['case'],int(r['minute'])),{}))
            op_scores=defaultdict(float)
            if ops:
                for op in ops:
                    total=sum(by_op[op].values())
                    for s in services:
                        op_scores[s]+=(by_op[op][s]/total if total else global_scores[s])/len(ops)
            else:
                op_scores.update(global_scores)
            for name,values in zip(outputs,(global_scores,op_scores)):
                case=outputs[name]['cases'].setdefault(r['case'],{'case_id':r['case'],'windows':[]})
                case['windows'].append({'minute':r['minute'],'scores_by_method':{name:dict(values)}})
    return {name:evaluate_attribution(doc,cohort,name) for name,doc in outputs.items()}


def main():
    report = {'protocol': 'frozen test cohort; event zero-filled score means; full services/worst ties; '
                          'explicit unavailable state; available logits rank regardless of sign',
              'datasets': {}}
    lines = ['# 同 cohort 基线重算（修正负 logits 弃权错误）', '',
             '已有作者代码适配版本；seed42；五折测试裁到当前固定窗口。未重训基线。',
             'CPCA 的全零恢复分按自身定义弃权；基线的合法负 logits 不弃权。',
             '每个可用基线向量加同一个常数，保持窗口排名和事件均值排名不变；缺失输出仍为零证据。', '']
    for ds in ('ob','tt'):
        cohort = json.loads((EXPERIMENT/f'inputs/prediction_{ds}_corrected_v2_seed42.json').read_text())
        services = cohort['folds'][0]['services']
        event_cases={r['event']:r['case'] for f in cohort['folds'] for r in f['test_predictions'] if r['failure_target'] and r['root']}
        audit = json.loads((EXPERIMENT/'attribution'/ds/'audit.json').read_text())
        cpca = evaluate_attribution(audit,cohort,'cpca_uniform')
        methods = {'cpca_uniform': cpca}
        methods.update(supervised_priors(cohort,ds))
        # Use operation-level evidence to explicitly equalize every simple control.
        for name in ('global_z','inclusive_z','exclusive_z','exclusive_p95_z','cpca_local','cpca_edge','cpca_caller'):
            converted = {'cases': {}}
            for key, case in audit['cases'].items():
                rows=[]
                for w in case['windows']:
                    evidence=w.get('operation_evidence',[])
                    if not evidence:
                        raise ValueError('No operation-level evidence for canonical aggregation')
                    scores=defaultdict(float)
                    for op in evidence:
                        for s,v in op['scores'][name].items():
                            scores[s]+=v/len(evidence)
                    rows.append({'minute':w['minute'],'scores_by_method':{name:dict(scores)}})
                converted['cases'][key]={'case_id':case['case_id'],'windows':rows}
            methods[name]=evaluate_attribution(converted,cohort,name)
            if name.endswith('_z'):
                methods[name+'_stored_native']=evaluate_attribution(audit,cohort,name)
            if name.endswith('_z'):
                signed={'cases':{}}
                for case_key, case in converted['cases'].items():
                    signed['cases'][case_key]={'case_id':case['case_id'],'windows':[
                        {'minute':w['minute'],'scores_by_method':{name+'_rank_all_finite':
                            translate_available_scores([w['scores_by_method'][name].get(s,0.) for s in services],services)}}
                        for w in case['windows']]}
                methods[name+'_rank_all_finite']=evaluate_attribution(signed,cohort,name+'_rank_all_finite')
        details={}
        for section, names in (('frameworks',('ltfp','art','eadro','instantops')),
                               ('specialists',('circa','tracerca','microrank','tvdiag','torai','rcd','diagfusion'))):
            for name in names:
                converted={'cases':{}}
                detail={'folds':[], 'all_negative_valid_positive_windows':0, 'unavailable_positive_windows':0}
                for fold in cohort['folds']:
                    path=BASE/section/f'{ds}_{name}_fold{fold["fold"]}_seed42'
                    rows=[json.loads(line) for line in (path/'predictions.jsonl').open()]
                    lookup={(r['case_id'],int(datetime.fromisoformat(r['input_minute']).timestamp())):r for r in rows}
                    assert len(lookup)==len(rows)
                    metrics=json.loads((path/'metrics.json').read_text())
                    manifest=json.loads((path/'data_manifest.json').read_text())
                    schema=json.loads((ROOT/f'artifacts/re2_{ds}_panel_fair_v1/feature_schema.json').read_text())
                    assert schema['services']==services
                    mismatched_files=[]
                    for file, expected in manifest['files_sha256'].items():
                        current=ROOT/f'artifacts/re2_{ds}_panel_fair_v1'/file
                        if hashlib.sha256(current.read_bytes()).hexdigest()!=expected:
                            mismatched_files.append(file)
                    if mismatched_files:
                        raise ValueError(f'Panel changed: {ds}/{name}: {mismatched_files}')
                    probabilities=[]; labels=[]
                    for p in fold['test_predictions']:
                        key=(p['case'],int(p['minute']))
                        if key not in lookup:
                            raise ValueError(f'Baseline missing cohort window {name}/{key}')
                        r=lookup[key]
                        assert int(r['failure_target'])==p['failure_target']
                        assert list(map(float,r['root_targets']))==[float(s==p['root']) for s in services]
                        if section=='frameworks':
                            probabilities.append(float(r['risk_score'])); labels.append(p['failure_target'])
                        if not p['failure_target'] or not p['root']:
                            continue
                        raw=r.get('root_scores')
                        available=r.get('available',True)
                        detail['all_negative_valid_positive_windows']+=bool(raw and max(raw)<0 and available)
                        detail['unavailable_positive_windows']+=int(raw is None or available is False)
                        scores=translate_available_scores(raw,services,available)
                        case=converted['cases'].setdefault(p['case'],{'case_id':p['case'],'windows':[]})
                        case['windows'].append({'minute':p['minute'],'scores_by_method':{name:scores}})
                    f={'fold':fold['fold'],'cohort_windows':len(fold['test_predictions']),
                       'extra_baseline_windows_excluded':len(rows)-len(fold['test_predictions']),
                       'source_commit':metrics.get('source_commit'), 'smoke_only':metrics.get('smoke_only'),
                       'implementation_id':metrics.get('implementation_id')}
                    if section=='frameworks':
                        threshold=metrics['at_validation_threshold']['threshold']
                        f.update(ap=float(average_precision_score(labels,probabilities)),
                                 f1=float(f1_score(labels,np.array(probabilities)>=threshold)), threshold=threshold)
                    detail['folds'].append(f)
                evaluated=evaluate_attribution(converted,cohort,name)
                methods[name]=evaluated
                detail['cpca_paired_delta']=paired_delta(cpca,evaluated,event_cases)
                if section=='frameworks':
                    detail['prediction']={k:statistics.mean(f[k] for f in detail['folds']) for k in ('ap','f1')}
                details[name]=detail
        old_path=HERE/'inputs'/f'shapleyiq_old_{ds}.json'
        old_shapley=json.loads(old_path.read_text())
        for name in ('shapleyiq_default','shapleyiq_warmup10'):
            methods[name+'_old_adapter']=evaluate_attribution(old_shapley,cohort,name)
        new_path=HERE/f'shapley_matched_{ds}.json'
        if new_path.exists():
            new=json.loads(new_path.read_text())
            if len(new['cases'])==len(old_shapley['cases']):
                for name in ('shapleyiq_default','shapleyiq_warmup10','shapleyiq_default_uniform','shapleyiq_warmup10_uniform'):
                    methods[name+'_aligned']=evaluate_attribution(new,cohort,name)
        for name,evaluated in methods.items():
            if name!='cpca_uniform':
                details.setdefault(name,{})['cpca_paired_delta']=paired_delta(cpca,evaluated,event_cases)
        report['datasets'][ds]={'methods':methods,'baseline_checks':details}
        lines += ['## '+ds.upper(),'', '| 方法 | 事件 H@1 | 事件 MRR | 窗口 H@1 | 窗口 MRR |', '|---|---:|---:|---:|---:|']
        for name,m in methods.items():
            lines.append(f'| {name} | {m["event"]["hit_at_1"]:.4f} | {m["event"]["mrr"]:.4f} | '
                         f'{m["window"]["hit_at_1"]:.4f} | {m["window"]["mrr"]:.4f} |')
        lines.append('')
    lines += ['旧 ShapleyIQ 使用原有 per-trace 平均，不能称为 operation 等权；aligned 控制另补 operation 等权。',
              '简单控制按每个 operation 的原始分数显式等权重算；global_z 等仍按其定义将正异常作为证据。',
              '分数语义、训练输入和标签适配不同；本表不等价于各原论文完整系统的同环境复现。',
              '逐折核验、负 logits 计数、事件配对区间见 BASELINES_UNIFIED.json。']
    (HERE/'BASELINES_UNIFIED.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    (HERE/'BASELINES_UNIFIED.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__=='__main__':
    main()
