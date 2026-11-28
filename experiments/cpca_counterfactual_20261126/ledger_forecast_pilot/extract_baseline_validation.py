"""Recover validation outputs from frozen checkpoints for equal-budget gates.

Checks that the same checkpoint reproduces every saved test probability before
using its validation outputs; never fits thresholds on test labels.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path('/Users/95penguin/project/NextMinuteRCA')
sys.path.insert(0, str(ROOT / 'src'))
from ocsid import training


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=('ob', 'tt'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output exists')
    torch.set_num_threads(1)
    source = ROOT / f'experiments/cpca_counterfactual_20261126/inputs/prediction_{args.dataset}_corrected_v2_seed42.json'
    frozen = json.loads(source.read_text())
    _, _, ends, raw_x, a, raw_o, m, y, _, _ = training._build_arrays(args.dataset, training.PANELS[args.dataset])
    positions = {end: i for i, end in enumerate(ends)}
    output = {'dataset': args.dataset, 'folds': []}
    for fold in frozen['folds']:
        ds = training.Re2WindowDataset(training.PANELS[args.dataset], history=5, split_policy='deployment',
                                      split_fold=fold['fold'], graph_direction='symmetric')
        starts = {}
        for case, minute in zip(ds.case_ids, ds.timestamp_epochs):
            starts.setdefault(case, minute)
        ready = lambda e: ds.timestamp_epochs[e] >= starts[ds.case_ids[e]] + 600
        train, val, test = [[e for e in split if ready(e)] for split in ds.predefined_splits]
        x, o = training._standardize(ds, raw_x, raw_o, m, train, ends)
        checkpoint_path = Path(frozen['config']['checkpoint_dir']) / f'{args.dataset}_fold{fold["fold"]}_seed42.pt'
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        model = training.OCSIDModel(**checkpoint['model_kwargs'])
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        def predict(indices):
            idx = [positions[e] for e in indices]
            with torch.no_grad():
                out = model(*[torch.from_numpy(v[idx]) for v in (x, a, o, m)])
            return out['probability'].numpy(), y[idx]
        tp, _ = predict(test)
        saved = np.asarray([r['probability'] for r in fold['test_predictions']])
        assert len(tp) == len(saved)
        max_difference = float(np.max(np.abs(tp - saved)))
        if max_difference > 1e-7:
            raise ValueError(f'Checkpoint mismatch: {max_difference}')
        vp, vy = predict(val)
        output['folds'].append({'fold': fold['fold'], 'probabilities': vp.tolist(),
                                'labels': vy.tolist(), 'max_test_probability_difference': max_difference,
                                'checkpoint': str(checkpoint_path)})
        print(args.dataset, fold['fold'], 'verified', max_difference, flush=True)
    args.output.write_text(json.dumps(output) + '\n')


if __name__ == '__main__':
    main()
