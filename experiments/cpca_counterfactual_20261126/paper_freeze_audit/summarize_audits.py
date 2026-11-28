"""Freeze evidence: exact baseline parity, coverage and reference sensitivity."""
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from build_cache import ROOT, EXPERIMENT

HERE = Path(__file__).resolve().parent


def ranks(scores, roots):
    # scores: [..., windows, services]; roots: [windows]
    root_values=np.take_along_axis(scores,np.broadcast_to(roots,scores.shape[:-1])[...,None],axis=-1)[...,0]
    result=(scores>=root_values[...,None]).sum(axis=-1)
    return np.where(scores.max(axis=-1)>0,result,0)


def metrics(rank):
    rank=np.asarray(rank)
    reciprocal=np.divide(1.,rank,out=np.zeros_like(rank,dtype=float),where=rank>0)
    return {'n':int(rank.shape[-1]),'hit_at_1':np.mean(rank==1,axis=-1).tolist(),
            'mrr':np.mean(reciprocal,axis=-1).tolist(),'abstentions':np.sum(rank==0,axis=-1).tolist()}


def describe(values):
    v=np.asarray(values)
    return {'mean':float(v.mean()),'min':float(v.min()),'max':float(v.max()),
            'reference_resampling_range95':np.quantile(v,[.025,.975]).tolist()}


def aggregate_coverage(cases,scope):
    total=Counter()
    for case in cases.values():
        for key,value in case['coverage'][scope].items():
            if key.startswith('max_'):
                total[key]=max(total[key],value)
            else:
                total[key]+=value
    denom=total['traces_with_any_recognized_entry_span']
    total['acceptance_among_traces_with_recognized_entry']=total['accepted_trace_ids']/denom if denom else None
    total['partial_fraction_among_accepted']=total['accepted_partial']/total['accepted_trace_ids'] if total['accepted_trace_ids'] else None
    total['retained_span_fraction_among_accepted']=total['retained_spans']/total['accepted_raw_spans'] if total['accepted_raw_spans'] else None
    total['retained_span_fraction_among_all_raw_rows']=total['retained_spans']/total['raw_rows'] if total['raw_rows'] else None
    return dict(total)


def main():
    report={'protocol':'fixed median CPCA; 100 whole-healthy-minute bootstrap references; no model retraining',
            'interval_note':'reference sensitivity ranges, NOT confidence intervals of causal repair effects',
            'datasets':{}}
    lines=['# 定型审计：参考稳定性、全量覆盖和完整请求分层','',
           '冻结 median/replace/post/uniform。100 次健康分钟块重采样，整块请求及其所有分量共同抽取。',
           '分位范围只描述参考变化，不能称为真实修复效果的置信区间。','']
    for ds in ('ob','tt'):
        index=json.loads((HERE/f'cache_{ds}/index.json').read_text())
        services=index['services']; windows=[]; modes=defaultdict(list); sampled=[]
        for case,meta in index['cases'].items():
            with np.load(HERE/f'cache_{ds}'/(case+'.npz')) as cache:
                for mode in meta['modes']:
                    modes[mode].append(cache['scores_'+mode])
                sampled.append(cache['sampled_scores'])
            windows.extend([dict(w,case=case) for w in meta['windows']])
        modes={k:np.concatenate(v,axis=0) for k,v in modes.items()}
        sampled=np.concatenate(sampled,axis=1)
        root_ids=np.array([services.index(w['root']) for w in windows])
        events=list(dict.fromkeys(w['event'] for w in windows))
        event_rows=[np.array([i for i,w in enumerate(windows) if w['event']==e]) for e in events]
        event_roots=np.array([root_ids[rows[0]] for rows in event_rows])
        for rows in event_rows:
            assert len({windows[i]['root'] for i in rows})==1
        mode_results={}
        event_scores_by_mode={}
        for mode,scores in modes.items():
            event_scores=np.stack([scores[rows].mean(axis=0) for rows in event_rows])
            event_scores_by_mode[mode]=event_scores
            er=ranks(event_scores,event_roots); wr=ranks(scores,root_ids)
            mode_results[mode]={'event':metrics(er),'window':metrics(wr),'event_ranks':dict(zip(events,er.tolist()))}
        frozen=json.loads((EXPERIMENT/'attribution'/ds/'audit.json').read_text())
        lookup={(c['case_id'],int(w['minute'])):w['scores_by_method']['cpca_uniform']
                for c in frozen['cases'].values() for w in c['windows']}
        expected=np.array([[lookup.get((w['case'],int(w['minute'])),{}).get(s,0.) for s in services] for w in windows])
        maxdiff=float(np.max(np.abs(expected-modes['full'])))
        assert maxdiff<1e-12, (ds,maxdiff)
        expected_event=np.stack([expected[rows].mean(axis=0) for rows in event_rows])
        np.testing.assert_array_equal(ranks(expected_event,event_roots),ranks(event_scores_by_mode['full'],event_roots))
        sample_window_ranks=ranks(sampled,root_ids)
        sample_event_scores=np.stack([sampled[:,rows,:].mean(axis=1) for rows in event_rows],axis=1)
        sample_event_ranks=ranks(sample_event_scores,event_roots)
        base_ranks=ranks(event_scores_by_mode['full'],event_roots)
        sample_metric={'event':metrics(sample_event_ranks),'window':metrics(sample_window_ranks)}
        sensitivity={scope:{name:describe(values[name]) for name in ('hit_at_1','mrr','abstentions')}
                     for scope,values in sample_metric.items()}
        per_event=[]
        for j,event in enumerate(events):
            r=sample_event_ranks[:,j]
            root_score=sample_event_scores[:,j,event_roots[j]]
            per_event.append({'event':event,'case':windows[event_rows[j][0]]['case'],
                              'baseline_rank':int(base_ranks[j]),'reference_rank_agreement':float(np.mean(r==base_ranks[j])),
                              'rank_min':int(r.min()),'rank_max':int(r.max()),
                              'hit_at_1_frequency':float(np.mean(r==1)),
                              'root_score_mean':float(root_score.mean()),'root_score_sd':float(root_score.std(ddof=1))})
        sensitivity.update(mean_event_rank_agreement=float(np.mean(sample_event_ranks==base_ranks)),
                           events_never_changing_rank=sum(e['reference_rank_agreement']==1 for e in per_event),
                           events_agreement_below_90pct=sum(e['reference_rank_agreement']<.9 for e in per_event),
                           per_event=per_event)
        strata={}
        for grouping in ('entry_acceptance','complete_fraction'):
            buckets=defaultdict(list)
            for j,event in enumerate(events):
                case=windows[event_rows[j][0]]['case']; c=index['cases'][case]['coverage']['positive_cohort']
                if grouping=='entry_acceptance':
                    value=c.get('accepted_trace_ids',0)/max(c.get('traces_with_any_recognized_entry_span',0),1)
                else:
                    value=c.get('accepted_complete',0)/max(c.get('accepted_trace_ids',0),1)
                bucket='>=90%' if value>=.9 else ('50%-90%' if value>=.5 else '<50%')
                buckets[bucket].append(base_ranks[j])
            strata[grouping]={k:metrics(v) for k,v in buckets.items()}
        coverage={scope:aggregate_coverage(index['cases'],scope) for scope in ('all','positive_cohort')}
        budget={k:sum(m['budget'].get(k,0) for m in index['cases'].values()) for k in ('examined','invalid','accepted')}
        old_budget={k:sum(c.get('tree_diagnostics',{}).get(k,0) for c in frozen['cases'].values())
                    for k in ('positive_requests_examined','invalid_positive_trees')}
        assert budget['examined']==old_budget['positive_requests_examined']
        assert budget['invalid']==old_budget['invalid_positive_trees']
        np.testing.assert_array_equal(ranks(expected,root_ids),ranks(modes['full'],root_ids))
        references={'cases_with_less_than_5_healthy_minute_blocks':sum(len(m['reference_blocks'])<5 for m in index['cases'].values()),
                    'min_healthy_minute_blocks':min(len(m['reference_blocks']) for m in index['cases'].values()),
                    'max_healthy_minute_blocks':max(len(m['reference_blocks']) for m in index['cases'].values()),
                    'cutoff_after_injection_cases':[]}
        evaluation_cases=[m for m in index['cases'].values() if m['windows']]
        references['evaluation_cases']=len(evaluation_cases)
        references['min_blocks_in_evaluation_cases']=min(len(m['reference_blocks']) for m in evaluation_cases)
        with zipfile.ZipFile(ROOT/f'data/RCAEval/RE2-{ds.upper()}.zip') as archive:
            for case,meta in index['cases'].items():
                rawcase=case[len('RE2-'+ds.upper()+'_'):]
                fault,repetition=rawcase.rsplit('_',1)
                inject=int(archive.read(f'RE2-{ds.upper()}/{fault}/{repetition}/inject_time.txt').decode().strip())
                if meta['cutoff']>inject:
                    references['cutoff_after_injection_cases'].append(case)
        report['datasets'][ds]={'baseline_max_score_difference':maxdiff,'modes':mode_results,
                               'reference_sensitivity':sensitivity,'reference_support':references,
                               'coverage':coverage,'budget':budget,'historical_capped_budget':old_budget,'strata':strata}
        lines += ['## '+ds.upper(),'', '| 请求集合（共享主健康参考） | 事件 H@1 | 事件 MRR | 窗口 MRR | 无证据窗口 |',
                  '|---|---:|---:|---:|---:|']
        for mode,m in mode_results.items():
            lines.append(f'| {mode} | {m["event"]["hit_at_1"]:.4f} | {m["event"]["mrr"]:.4f} | '
                         f'{m["window"]["mrr"]:.4f} | {m["window"]["abstentions"]} |')
        lines += ['',f'原始冻结分数最大差：{maxdiff:.3g}；事件排名完全复现。',
                  f'100 次参考重采样事件 MRR：{sensitivity["event"]["mrr"]}。',
                  f'事件名次平均一致率：{sensitivity["mean_event_rank_agreement"]:.4f}；'
                  f'名次始终不变事件：{sensitivity["events_never_changing_rank"]}/{len(events)}。','']
        c=coverage['all']; p=coverage['positive_cohort']
        lines += ['| 全量覆盖量 | 数值 |','|---|---:|',
                  f'| 原始 trace ID | {c["raw_trace_ids"]} |',f'| 原始 span 行 | {c["raw_rows"]} |',
                  f'| 有识别入口的 trace | {c["traces_with_any_recognized_entry_span"]} |',
                  f'| 接受 trace | {c["accepted_trace_ids"]} |',f'| 完整接受 / 部分接受 | {c["accepted_complete"]} / {c["accepted_partial"]} |',
                  f'| 接受请求中的 span 保留率 | {c["retained_span_fraction_among_accepted"]:.4%} |',
                  f'| 正例入口 trace 接受率（不截断到100） | {p["acceptance_among_traces_with_recognized_entry"]:.4%} |',
                  f'| 接受请求中发生时钟修改 | {c["accepted_with_clock_changes"]} |',
                  f'| 最大 span 总开始位移（μs） | {c["max_total_start_displacement_us"]} |',
                  f'| 总开始位移超过1ms的 span | {c["span_total_displacement_over_1ms"]} |','']
        lines += ['拒绝原因：'+str({k:v for k,v in c.items() if k.startswith('rejected_') and not k.startswith('rejected_flag_')}),
                  '参考支持：'+str(references),'']
    lines += ['## 解释限制','',
              '- no_clock_shared_cap 仅保留原 cap 中完全不需要时间修改的请求；仍共享主参考且不补样，不等价于重跑完整的零容差模型。',
              '- complete_shared_cap 是原样本内完整请求子集；complete_refill 允许按相同预算补取后续完整请求；两者均保留所有评价窗口，缺证据记零。',
              '- 无识别入口的原始 trace 可能是内部片段或入口埋点缺失，不能一概当业务请求拒绝，也无法仅凭 span CSV 判断物理缺失原因。',
              '- sibling overlap/boundary violation 是观测模式，不能直接区分真实并行、缺失记录和时钟差。',
              '- 参考重采样分位范围不是干预效果置信区间，未参与重新选择主结构。']
    (HERE/'AUDIT_RESULTS.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    (HERE/'AUDIT_RESULTS.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__=='__main__':
    main()
