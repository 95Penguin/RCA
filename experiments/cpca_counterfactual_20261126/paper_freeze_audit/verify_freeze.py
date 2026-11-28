"""Verify the archived scientific outputs and input/parser provenance."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from audit_paths import ROOT

HERE=Path(__file__).resolve().parent


def main():
    summary=json.loads((HERE/'AUDIT_RESULTS.json').read_text())
    baseline=json.loads((HERE/'BASELINES_UNIFIED.json').read_text())
    checked=0
    results={}
    for ds in ('ob','tt'):
        index=json.loads((HERE/f'cache_{ds}/index.json').read_text())
        assert len(index['cases'])==90
        assert index['repetitions']==100
        for name,digest in index['source_hashes'].items():
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
        total_windows=0
        for case,meta in index['cases'].items():
            with np.load(HERE/f'cache_{ds}'/(case+'.npz')) as cache:
                sample=cache['sampled_scores']
                assert sample.shape==(100,len(meta['windows']),len(index['services']))
                assert np.isfinite(sample).all() and (sample>=-1e-12).all()
                assert (sample.sum(axis=-1)<=1+1e-9).all()
                for mode in meta['modes']:
                    scores=cache['scores_'+mode]
                    assert np.isfinite(scores).all() and (scores>=-1e-12).all()
                    assert (scores.sum(axis=-1)<=1+1e-9).all()
            total_windows+=len(meta['windows']); checked+=1
        assert total_windows=={'ob':455,'tt':436}[ds]
        assert summary['datasets'][ds]['baseline_max_score_difference']<1e-12
        new=json.loads((HERE/f'shapley_matched_{ds}.json').read_text())
        assert new['source_sha256']==hashlib.sha256((HERE/'third_party/shapley_value_rca.py').read_bytes()).hexdigest()
        count_mismatches=[]
        for case,meta in new['cases'].items():
            if meta['diagnostics']['accepted_requests']!=index['cases'][case]['budget']['accepted']:
                count_mismatches.append(case)
        assert not count_mismatches
        assert len(new['metrics'])==4
        methods=baseline['datasets'][ds]['methods']
        for method in ('ltfp','shapleyiq_warmup10_uniform_aligned'):
            assert methods[method]['positive_windows']==total_windows
            assert methods[method]['event']['n']=={'ob':55,'tt':64}[ds]
        results[ds]={'cases_checked':90,'positive_windows':total_windows,
                     'baseline_max_score_difference':summary['datasets'][ds]['baseline_max_score_difference'],
                     'shapley_request_count_parity':True,'sampled_scores_finite_nonnegative_bounded':True,
                     'current_parser_hash_matches':True}
    payload={'status':'passed','cache_files_checked':checked,'datasets':results}
    (HERE/'VERIFY_RESULTS.json').write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps(payload,indent=2))
    files={}
    for path in sorted(HERE.rglob('*')):
        if path.is_file() and path.name!='MANIFEST.json' and '__pycache__' not in path.parts:
            files[str(path.relative_to(HERE))]={'bytes':path.stat().st_size,
                                               'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    manifest={'python':sys.version,'numpy':np.__version__,'root':str(ROOT),
              'files':files,'note':'Local audit archive; no trained model or raw dataset modified.'}
    (HERE/'MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':
    main()
