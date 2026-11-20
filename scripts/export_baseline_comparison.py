#!/usr/bin/env python3
"""Export complete five-fold baseline tables without changing training or scoring."""
from __future__ import annotations
import argparse,csv,json,statistics,time
from pathlib import Path

KEYS=('Precision','Recall','F1','AP','Hit@1','Hit@3','Hit@5','MRR')
FRAMEWORKS=('ltfp','instantops','eadro','art')
PREDICTION=('anofusion','hades','maat')
RCA=('diagfusion','tvdiag','torai','rcd','circa','tracerca','microrank')

def write_csv(path,rows,fields):
    temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    temp.replace(path)

def dump(path,payload):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(payload,indent=2,ensure_ascii=False,allow_nan=False)+'\n');temp.replace(path)

def export(root):
    folds=[];errors=[]
    for group in ('frameworks','specialists'):
        for path in sorted((root/group).glob('*/metrics.json')):
            try:r=json.loads(path.read_text())
            except json.JSONDecodeError:
                errors.append(str(path));continue
            if r.get('smoke_only'):raise ValueError(f'Refusing smoke results: {path}')
            p=r if group=='frameworks' else r['failure_prediction'];c=p['at_validation_threshold'] if p else None
            rc=r.get('root_cause_window')
            row={k:r[k] for k in ('dataset','method','fold','seed','implementation_id')}
            row.update(group=group,metrics_file=str(path.relative_to(root)))
            for key in ('Precision','Recall','F1'):row[key]=c[key.lower()] if c else None
            row['AP']=p['average_precision'] if p else None
            for key,k in (('Hit@1','hit_at_1'),('Hit@3','hit_at_3'),('Hit@5','hit_at_5'),('MRR','mrr')):row[key]=rc[k] if rc else None
            row['root_coverage']=rc.get('coverage',1.0) if rc else None
            row['validation_threshold']=c['threshold'] if c else None
            fixed=r.get('at_probability_0_5')
            if r['method']=='hades':
                records=[json.loads(x) for x in (path.parent/'predictions.jsonl').read_text().splitlines()]
                tp=sum(x['failure_target']==1 and x['risk_score']>=.5 for x in records)
                fp=sum(x['failure_target']==0 and x['risk_score']>=.5 for x in records)
                fn=sum(x['failure_target']==1 and x['risk_score']<.5 for x in records)
                precision=tp/(tp+fp) if tp+fp else 0.;recall=tp/(tp+fn) if tp+fn else 0.
                fixed={'precision':precision,'recall':recall,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.}
            for key in ('precision','recall','f1'):row[key+'_at_0_5']=fixed[key] if fixed else None
            folds.append(row)
    summaries=[]
    for dataset,method in sorted({(r['dataset'],r['method']) for r in folds}):
        runs=[r for r in folds if (r['dataset'],r['method'])==(dataset,method)]
        if len({(r['fold'],r['seed']) for r in runs})!=len(runs):raise ValueError('Duplicate fold/seed')
        complete=len(runs)==5 and {r['fold'] for r in runs}==set(range(5)) and len({r['seed'] for r in runs})==1
        row={'dataset':dataset,'method':method,'group':runs[0]['group'],'completed_folds':len(runs),'complete':complete,'implementation_id':runs[0]['implementation_id'],'seed':runs[0]['seed']}
        for key in (*KEYS,'root_coverage','precision_at_0_5','recall_at_0_5','f1_at_0_5'):
            vals=[r[key] for r in runs if r[key] is not None]
            row[key]={'mean':statistics.fmean(vals) if vals else None,'sd':statistics.stdev(vals) if len(vals)>1 else None,'n':len(vals)}
        summaries.append(row)
    out=root/'full_metrics';out.mkdir(exist_ok=True)
    dump(out/'summary_all.json',summaries)
    dump(out/'progress.json',{'completed_runs':len(folds),'expected_runs':140,'complete_method_dataset_pairs':sum(r['complete'] for r in summaries),'expected_pairs':28,'temporarily_unreadable_metrics':errors})
    if folds:write_csv(out/'per_fold_metrics.csv',folds,list(folds[0]))
    def flat(row):
        x={k:row[k] for k in ('dataset','method','seed','completed_folds','implementation_id')}
        for key in KEYS:
            x[key+'_mean']=row[key]['mean'];x[key+'_sd']=row[key]['sd']
        return x
    complete=[flat(r) for r in summaries if r['complete']]
    fields=['dataset','method','seed','completed_folds','implementation_id']+[k+'_'+suffix for k in KEYS for suffix in ('mean','sd')]
    write_csv(out/'completed_five_fold_metrics.csv',complete,fields)
    for name,methods,columns in (('frameworks',FRAMEWORKS,KEYS),('prediction',PREDICTION,KEYS[:4]),('rca',RCA,KEYS[4:])):
        rows=[flat(r) for r in summaries if r['complete'] and r['method'] in methods]
        selected=['dataset','method','seed','completed_folds','implementation_id']+[k+'_'+s for k in columns for s in ('mean','sd')]
        write_csv(out/(name+'_five_fold.csv'),[{k:r[k] for k in selected} for r in rows],selected)
    lines=['# Baseline 完整指标','',f'已完成 {len(folds)}/140 次运行。下面仅列五折齐全的结果。数值为五折非加权均值 ± 样本标准差；seed=42。', '',
           '输入过去5个完整分钟，目标为精确下一分钟业务故障状态及可归因服务。Precision、Recall、F1采用各折验证集最大F1阈值；AP使用原始连续分数。框架两组指标来自同一checkpoint。根因指标为窗口级，不混入事后事件聚合。','']
    def cell(row,key):
        s=row[key]
        return '—' if s['mean'] is None else f"{s['mean']:.3f} ± {s['sd']:.3f}"
    for title,methods,columns in (('框架类：预测与根因全部指标',FRAMEWORKS,KEYS),('预测专用方法',PREDICTION,KEYS[:4]),('根因专用方法',RCA,KEYS[4:])):
        lines+=['## '+title,'','| 数据集 | 方法 | '+' | '.join(columns)+' |','|---|---|'+'---|'*len(columns)]
        for ds in ('ob','tt'):
            for method in methods:
                row=next((r for r in summaries if r['dataset']==ds and r['method']==method and r['complete']),None)
                if row:lines.append('| '+ds.upper()+' | '+method+' | '+' | '.join(cell(row,k) for k in columns)+' |')
        lines+=['']
    lines+=['## 尚未完成','','| 数据集 | 方法 | 已完成折数 |','|---|---|---:|']
    for ds in ('ob','tt'):
        for method in (*FRAMEWORKS,*PREDICTION,*RCA):
            row=next((r for r in summaries if r['dataset']==ds and r['method']==method),None)
            if row is None or not row['complete']:lines.append(f"| {ds.upper()} | {method} | {row['completed_folds'] if row else 0}/5 |")
    lines+=['','概率输出方法的固定0.5阈值指标作为补充保存在 per_fold_metrics.csv（*_at_0_5），不与验证阈值指标混用。原始每折完整精度、阈值及实现标识保留在CSV中。','',
            '这里是本项目统一任务下的适配版本结果，不能当作作者原论文原任务的精确复现。各方法的输入与监督差异见 baselines/RUNNING.md 和 baselines/SPECIALISTS.md。']
    temp=out/'RESULTS_FULL.md.tmp';temp.write_text('\n'.join(lines)+'\n');temp.replace(out/'RESULTS_FULL.md')
    return len(folds)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-root',type=Path,required=True);p.add_argument('--watch',action='store_true');args=p.parse_args()
    while True:
        count=export(args.run_root);print(json.dumps({'exported_runs':count}),flush=True)
        if not args.watch:break
        status_path=args.run_root/'status.json'
        status=json.loads(status_path.read_text()) if status_path.exists() else {}
        codes=status.get('process_exit_codes',{})
        if codes and all(x is not None for x in codes.values()):break
        time.sleep(30)
if __name__=='__main__':main()
