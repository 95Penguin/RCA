"""Use the unchanged official OCSID trainer with experimental feature injection.

Only the input dataset adapter is wrapped; splits, optimizer, epochs, validation
selection, root auxiliary loss, operation branch and seeds are unchanged.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path('/Users/95penguin/project/NextMinuteRCA')
sys.path.insert(0, str(ROOT / 'src'))
from ocsid import training


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=('baseline', 'ledger', 'simple', 'mask'), required=True)
    parser.add_argument('--features', type=Path, required=True)
    args, remaining = parser.parse_known_args()
    table = json.loads(args.features.read_text())
    original = training.Re2WindowDataset

    class FeatureDataset(original):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            if args.variant == 'baseline':
                return
            extra = np.zeros((*self.features.shape[:-1], 2), dtype=np.float32)
            for i, (case, minute) in enumerate(zip(self.case_ids, self.timestamp_epochs)):
                row = table['cases'].get(case, {}).get(str(int(minute)))
                if row is None:
                    continue
                values = row.get('cpca' if args.variant == 'ledger' else 'simple', {})
                for j, service in enumerate(self.services):
                    extra[i, j, 0] = 0. if args.variant == 'mask' else values.get(service, 0.)
                    extra[i, j, 1] = float(service in row['observed'])
            self.features = np.concatenate((self.features, extra), axis=-1)
            # Both are placed in the existing trace modality. The observed_*
            # prefix also preserves the binary mask in official standardization.
            self.feature_columns = tuple(self.feature_columns) + (
                'trace_cpca_responsibility', 'observed_entry_ledger')

    training.Re2WindowDataset = FeatureDataset
    validation_outputs = []
    calibrate = training._calibrate
    def capture_validation(vp, vy, tp, ty):
        validation_outputs.append({'probabilities': vp.tolist(), 'labels': vy.tolist()})
        return calibrate(vp, vy, tp, ty)
    training._calibrate = capture_validation
    began = time.monotonic()
    result = training.main(remaining)
    output = Path(remaining[remaining.index('--output') + 1])
    doc = json.loads(output.read_text())
    doc['ledger_pilot'] = {'variant': args.variant, 'features': str(args.features),
                           'protocol': table['protocol'],
                           'seconds': time.monotonic() - began,
                           'note': 'Only input adapter changed; original trainer and CPCA ranking unchanged.'}
    for fold, validation in zip(doc['folds'], validation_outputs):
        fold['validation_predictions_for_fpr'] = validation
    output.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + '\n')
    return result


if __name__ == '__main__':
    raise SystemExit(main())
