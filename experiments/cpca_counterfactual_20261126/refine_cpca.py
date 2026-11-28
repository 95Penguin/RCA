"""Request-aligned empirical restoration; same-pass counterfactual ablations."""
import contextlib
import hashlib
import io
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
import argparse

import run_distributional_cpca as old
from ocsid import audit, cpca
from ocsid.evaluation import evaluate_attribution

METHODS = ('fixed_median', 'global_mean', 'matched_mean', 'matched_median', 'request_mean')
BANK = defaultdict(list)
STATS = {}
ACC = {}
DIAG = defaultdict(int)
EXAMPLES = []
LAST = None
LAST_RESULT = None

def layout(parsed):
    by_id, children, root, local, pre, post = parsed
    order = []
    def visit(v, path):
        key = path + ((v.service, v.operation),)
        order.append((key, v))
        for j, c in enumerate(sorted(children.get(v.sid, ()), key=lambda c: (c.start, c.sid))):
            visit(c, key + ((j,),))
    visit(root, ())
    signature = tuple(key for key, _ in order)
    vectors = tuple((local[v.sid], pre.get(v.sid, 0.), post.get(v.sid, 0.)) for _, v in order)
    return signature, order, vectors


def collect(*args, **kwargs):
    # Reuse the same validated tree for the original normal statistics and bank.
    real = cpca.decompose_tree
    parsed = real(args[0], args[2], args[5] if len(args)>5 else 'post',
                  args[6] if len(args)>6 else 0., args[7] if len(args)>7 else 'strict')
    cpca.decompose_tree = lambda *a, **k: parsed
    try:
        old.collect_normal_with_joint_pools(*args, **kwargs)
    finally:
        cpca.decompose_tree = real
    if parsed is not None and cpca.is_entry(parsed[2], args[2]):
        sig, _, vectors = layout(parsed)
        BANK[sig].append(vectors)


def pooled(values, median=False):
    key = (id(values), median)
    if key not in STATS:
        sums = [sum(v) for v in values]
        STATS[key] = statistics.median(sums) if median else statistics.fmean(sums)
    return STATS[key]


def score(parsed, normal_local, normal_edge, dataset, restoration='replace', *, components='all', edge_owner='callee'):
    global LAST, LAST_RESULT
    if parsed is None or restoration != 'replace' or components != 'all' or edge_owner != 'callee':
        return old.ORIGINAL_SCORE(parsed, normal_local, normal_edge, dataset, restoration, components=components, edge_owner=edge_owner)
    if parsed is LAST:
        return LAST_RESULT
    by_id, children, root, local, pre, post = parsed
    if root.duration <= 0 or not cpca.is_entry(root, dataset):
        return None
    delta = {m:defaultdict(float) for m in METHODS}
    fixed = old.ORIGINAL_SCORE(parsed, normal_local, normal_edge, dataset, 'replace')[1]
    delta['fixed_median'].update(fixed)
    ek = (root.service, root.operation)
    for v in by_id.values():
        parent = by_id.get(v.parent)
        obs = local[v.sid]+pre.get(v.sid,0.)+post.get(v.sid,0.)
        ref = normal_local.get((v.service,v.operation),local[v.sid])
        if parent:
            ref += sum(normal_edge.get((parent.service,parent.operation,v.service,v.operation),(pre[v.sid],post[v.sid])))
        _, vals = old.choose_pool(ek,v,parent)
        glob = old.POOLS['global_operation'].get((v.service,v.operation,parent is not None),())
        for m, values, med in [('global_mean',glob,False),('matched_mean',vals,False),('matched_median',vals,True)]:
            delta[m][v.service] += obs - (pooled(values,med) if values else ref)
    sig, order, vectors = layout(parsed)
    bank = BANK.get(sig, ())
    enough = len(bank)>=old.MIN_POOL
    if enough:
        key = ('trace',sig)
        if key not in STATS:
            mean_vectors = tuple(tuple(statistics.fmean(row[j][k] for row in bank) for k in range(3)) for j in range(len(order)))
            service_samples = defaultdict(list)
            for row in bank:
                sums = defaultdict(float)
                for (_,v),vec in zip(order,row): sums[v.service]+=sum(vec)
                for s,cost in sums.items(): service_samples[s].append(cost)
            STATS[key]=(mean_vectors,dict(service_samples))
        mean_vectors,samples=STATS[key]
        obs_cost=defaultdict(float)
        for (_,v),vec,ref in zip(order,vectors,mean_vectors):
            obs_cost[v.service]+=sum(vec)
            delta['request_mean'][v.service]+=sum(vec)-sum(ref)
        # Explicit tree replay retains all unmodified components and topology.
        def replay_target(v, target, replacement):
            own = replacement[v.sid][0] if v.service==target else local[v.sid]
            for c in children.get(v.sid,()):
                gaps = replacement[c.sid][1:] if c.service==target else (pre[c.sid],post[c.sid])
                own += sum(gaps)+replay_target(c,target,replacement)
            return own
        ref_by_id={v.sid:vec for (_,v),vec in zip(order,mean_vectors)}
        factual=replay_target(root,None,ref_by_id)
        if not math.isclose(factual,root.duration,abs_tol=.01,rel_tol=1e-8):
            raise ValueError('Factual tree replay does not conserve entry latency')
        for s in obs_cost:
            cf=replay_target(root,s,ref_by_id)
            if not math.isclose(root.duration-cf,delta['request_mean'][s],abs_tol=.01,rel_tol=1e-8):
                raise ValueError('Replay/ledger disagreement')
            if len(EXAMPLES)<40:
                costs=sorted(samples[s]); n=len(costs)
                EXAMPLES.append({'trace_id':hashlib.sha256(str(root.sid).encode()).hexdigest()[:12],
                  'service':s,'normal_requests':n,'observed_entry_us':root.duration,
                  'restored_entry_mean_us':cf,'signed_gain_us':root.duration-cf,
                  'restored_entry_empirical_q05_q95_us':[root.duration-obs_cost[s]+costs[int(.05*(n-1))], root.duration-obs_cost[s]+costs[int(.95*(n-1))]],
                  'note':'Empirical healthy-reference spread, not a confidence interval or causal effect.'})
        DIAG['matched_requests']+=1
        DIAG['replay_checks']+=len(obs_cost)
    else:
        delta['request_mean'].update(fixed)
        DIAG['fallback_requests']+=1
    minute=int(root.end//60_000_000)*60
    state=ACC.setdefault((minute,ek), {'D':0.,'n':0,'scores':{m:defaultdict(float) for m in METHODS}})
    state['D']+=root.duration; state['n']+=1
    for m in METHODS:
        for s,v in delta[m].items(): state['scores'][m][s]+=max(v,0.)
    LAST=parsed
    LAST_RESULT=(root,{s:max(v,0.) for s,v in delta['request_mean'].items()},local)
    return LAST_RESULT


def main():
    global LAST,LAST_RESULT
    p=argparse.ArgumentParser(); p.add_argument('--dataset',required=True,choices=('ob','tt')); p.add_argument('--output', type=Path, default=None); a=p.parse_args()
    output = a.output or old.OUT/'counterfactual_v2'/a.dataset/'results.json'
    out = output.parent
    out.mkdir(parents=True, exist_ok=True)
    if output.exists(): raise FileExistsError(output)
    cohort=json.loads((old.VALIDATION/f'prediction_{a.dataset}_corrected_v2_seed42.json').read_text())
    cases=sorted({r['case'] for f in cohort['folds'] for r in f['test_predictions'] if r['failure_target'] and r['root']})
    report={'implementation_version':'request-aligned-empirical-cpca-v2','dataset':a.dataset,
      'protocol':'All ground-truth positive labeled forecast windows; not alert-conditioned. Uniform operations; same requests; first10 reference; seed42 cohort.',
      'config':{'min_requests':old.MIN_POOL,'normal_minutes':10,'max_requests':100,'aggregation':'uniform'},'cases':{},'diagnostics':{}}
    audit._collect_normal=collect; cpca.score_decomposition=score
    for i,name in enumerate(cases):
        old.reset_case_state(); BANK.clear(); STATS.clear(); ACC.clear(); DIAG.clear(); EXAMPLES.clear(); LAST=None; LAST_RESULT=None
        path=old.flatten_case_name(a.dataset,name)
        per_case = out / output.stem / 'per_case'
        per_case.mkdir(parents=True, exist_ok=True)
        one=per_case/(path.replace('/','_')+'.json')
        print(f'[{a.dataset} {i+1}/{len(cases)}] {path}',flush=True)
        with contextlib.redirect_stdout(io.StringIO()):
            status=audit.main(['--dataset',a.dataset,'--cases',path,'--max-requests','100','--min-op-requests','1','--aggregation','uniform','--normal-mode','first10','--normal-minutes','10','--tree-policy','entry-component','--clock-tolerance-us','1000','--restoration','replace','--gap-policy','post','--output',str(one)])
        if status: raise RuntimeError(status)
        doc=json.loads(one.read_text())
        for key,case in doc['cases'].items():
            for w in case['windows']:
                ops=[tuple(v) for v in w['operations']]
                for m in METHODS:
                    merged=defaultdict(float)
                    for op in ops:
                        state=ACC.get((int(w['minute']),op))
                        if state:
                            for s,v in state['scores'][m].items(): merged[s]+=v/state['D']/len(ops)
                    w['scores_by_method'][m]=dict(merged)
            case['counterfactual_examples']=list(EXAMPLES)
            report['cases'][key]=case
        report['diagnostics'][path]={**DIAG,'healthy_requests':sum(map(len,BANK.values())),'healthy_topologies':len(BANK)}
    report['metrics']={m:evaluate_attribution(report,cohort,m) for m in METHODS}
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({m:{k:v for k,v in metric.items() if k in ('event','window')} for m,metric in report['metrics'].items()},indent=2),flush=True)

if __name__=='__main__': main()
