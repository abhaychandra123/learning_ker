"""Verify and summarize the four completed single-geometry capacity arms."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from compare_combined import write_csv
from synthetic_data import digest, write_json

ARMS=('tied_16','tied_32','untied_16','untied_32')


def summarize(root):
    summaries=[]; paired=[]; reference=None; keysets={}; provenance={}
    for arm in ARMS:
        run=root/arm
        setup=json.loads((run/'setup.json').read_text())
        status=json.loads((run/'status.json').read_text())
        saved=torch.load(run/'final.pt',map_location='cpu',weights_only=True)
        if status['status']!='complete' or not status['actual_evaluated']:
            raise ValueError(f'{arm} has not completed actual evaluation')
        if saved['contract']!=setup['contract'] or saved['model_config']!=setup['model_config'] or saved['epoch']!=status['completed_epochs']:
            raise ValueError('Checkpoint/setup/status mismatch')
        if f"{setup['model_config']['sharing']}_{setup['model_config']['layers']}"!=arm:
            raise ValueError('Arm name/model mismatch')
        common=dict(setup['contract'])
        common['config']={k:v for k,v in common['config'].items() if k not in
            ('sharing','layers','batch_size','checkpoint_mode')}
        signature=dict(contract=common,scale_uT=setup['scale_uT'],pad=setup['pad'],epochs=saved['epoch'])
        if reference is None: reference=signature
        elif reference!=signature: raise ValueError('Arms differ beyond planned depth/sharing/runtime settings')
        history=json.loads((run/'training.json').read_text())
        summary=dict(arm=arm,parameters=setup['parameter_count'],epochs=saved['epoch'],
            initial_train_mse=history[0]['post_update_mse'],final_train_mse=history[-1]['post_update_mse'],
            training_seconds=status['training_seconds'],clipped_updates=sum(r['gradient_clipped'] is True for r in history))
        for domain,filename in [('clean_synthetic','synthetic_final.csv'),('actual_comsol','actual_final.csv')]:
            with (run/filename).open() as stream: rows=list(csv.DictReader(stream))
            keys=[(r.get('sample_id',r.get('pair_id')),r['component']) for r in rows]
            if not rows or len(set(keys))!=len(keys): raise ValueError('Empty/duplicate evaluation rows')
            if domain in keysets and keysets[domain]!=set(keys): raise ValueError('Unmatched evaluation cases')
            keysets[domain]=set(keys)
            for metric in ('nrmse_range','relative_l2','amplitude_slope'):
                values=[float(r[metric]) for r in rows]
                if not np.isfinite(values).all(): raise ValueError('Nonfinite summary metrics')
                summary[f'{domain}_mean_{metric}']=float(np.mean(values))
            paired.extend(dict(arm=arm,domain=domain,geometry=r['geometry'],component=r['component'],
                input_z_um=r['input_z_um'],target_z_um=r['target_z_um'],
                **{m:r[m] for m in ('rmse_uT','nrmse_range','relative_l2','amplitude_slope')}) for r in rows)
        summaries.append(summary)
        provenance[arm]={name:digest(run/name) for name in
            ('setup.json','status.json','final.pt','training.json','synthetic_final.csv','actual_final.csv')}
    output=root/'comparison'; output.mkdir(exist_ok=True)
    write_csv(output/'summary.csv',summaries); write_csv(output/'per_case.csv',paired)
    write_json(output/'results.json',dict(summary=summaries,source_sha256=provenance,
        limits='One geometry, one seed, clean training-only fit diagnostic. Same-geometry COMSOL transfer. '
               'Equal updates, unequal compute. No independent generalization or calibrated confidence claim. '
               'Actual inputs were not used for training or checkpoint selection.'))
    for row in summaries:
        print(f"{row['arm']}: {row['parameters']} parameters; clean L2={100*row['clean_synthetic_mean_relative_l2']:.3f}%; "
              f"actual L2={100*row['actual_comsol_mean_relative_l2']:.3f}%")
    return summaries


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',type=Path,required=True)
    summarize(parser.parse_args().experiment)
