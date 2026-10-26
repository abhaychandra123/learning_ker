"""Verify a legacy process, stop only that run, snapshot its checkpoint, launch v2.

Run on the dedicated GPU server. Original data/run directories remain intact.
At most the in-progress epoch is discarded; completed epochs and Adam state
are copied to a fresh experiment. No generic pkill or process-name-only kills.
"""
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import time

import torch
from synthetic_data import digest,write_json


def training_command_matches(command,output):
    try:
        words=shlex.split(command)
        return (any(Path(w).name=='train_synthetic.py' for w in words)
                and '--output' in words
                and Path(words[words.index('--output')+1]).resolve()==Path(output).resolve())
    except (ValueError,IndexError):
        return False


def process_table():
    raw=subprocess.check_output(['ps','-eo','pid=,ppid=,stat=,args='],text=True)
    return {int(a):dict(parent=int(b),state=c,command=d)
            for line in raw.splitlines() if len(parts:=line.strip().split(None,3))==4
            for a,b,c,d in [parts]}


def main():
    project=Path(__file__).resolve().parents[1]
    os.chdir(project)
    old=project/'runs/synthetic_data_hypothesis_v1/augmented_1080'
    new=project/'runs/synthetic_data_hypothesis_fast_v2'
    if new.exists():
        raise FileExistsError('Optimized experiment exists. Inspect its logs/status; do not switch twice.')
    checkpoint=old/'last.pt'
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if not 0<=saved['epoch']<200:
        raise ValueError('Expected an incomplete legacy 200-epoch run')
    # Verify every legacy source before stopping any process.
    for name,checksum in saved['contract']['source_sha256'].items():
        if digest(project/'scripts'/name)!=checksum:
            raise ValueError('Legacy source differs before migration: '+name)
    from argparse import Namespace
    from synthetic_data import SyntheticDataset
    from train_synthetic_fast import validate_migration
    ds=SyntheticDataset(project/'runs/synthetic_v1')
    expected=Namespace(subset='full',device='cuda',layers=16,kernel_size=9,lr=1e-4,
                       lambda_ista=1e-4,seed=0,save_predictions=True,epochs=200)
    # Check checkpoint against actual immutable metadata/archives before stopping.
    from synthetic_experiment import verify_actual_run
    _,hashes=verify_actual_run(project/'runs/combined_full_200',ds)
    validate_migration(saved,ds,hashes,expected)
    pidfile=project/'logs/synthetic_data_hypothesis_v1.pid'
    launcher=int(pidfile.read_text().strip())
    table=process_table()
    matches=[pid for pid,p in table.items() if not p['state'].startswith('Z')
             and training_command_matches(p['command'],old)]
    if len(matches)!=1:
        raise RuntimeError('Expected exactly one active legacy augmented trainer; inspect processes manually.')
    child=matches[0]
    if (launcher not in table or table[child]['parent']!=launcher
            or 'hpc/run_synthetic_experiment.sh' not in table[launcher]['command']):
        raise RuntimeError('Launcher/trainer identity mismatch; no process was stopped.')
    new.mkdir(parents=True,exist_ok=False)
    write_json(new/'transition.json',dict(status='stopping_legacy',launcher_pid=launcher,trainer_pid=child,
                                        checkpoint_epoch_before_stop=saved['epoch']))
    # Prevent the wrapper starting its control arm while its child exits.
    os.kill(launcher,signal.SIGSTOP)
    try:
        # Recheck identity after stopping the parent; refuse a stale PID.
        current=process_table()
        if child not in current or not training_command_matches(current[child]['command'],old):
            raise RuntimeError('Trainer exited during transition; inspect preserved original run.')
        os.kill(child,signal.SIGTERM)
        for _ in range(150):
            current=process_table()
            if child not in current or current[child]['state'].startswith('Z'):
                break
            time.sleep(.2)
        else:
            raise RuntimeError('Legacy trainer has not exited; no new GPU run launched.')
        os.kill(launcher,signal.SIGTERM)
    finally:
        try:
            os.kill(launcher,signal.SIGCONT)
        except ProcessLookupError:
            pass
    shutil.copy2(checkpoint,new/'legacy_last.pt')
    saved=torch.load(new/'legacy_last.pt',map_location='cpu',weights_only=True)
    validate_migration(saved,ds,hashes,expected)
    write_json(new/'transition.json',dict(status='legacy_checkpoint_preserved',launcher_pid=launcher,
               trainer_pid=child,epoch=saved['epoch'],sha256=digest(new/'legacy_last.pt'),
               lost_work='Only work since the latest completed-epoch checkpoint may be discarded.'))
    env=dict(os.environ,QDM_PYTHON=str(Path.home()/'qdm-env/bin/python'),QDM_EXPERIMENT=str(new))
    with (new/'launch.log').open('xb') as log, open(os.devnull,'rb') as stdin:
        process=subprocess.Popen(['bash','hpc/run_synthetic_fast.sh'],cwd=project,env=env,
                                 stdin=stdin,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    (new/'launcher.pid').write_text(str(process.pid)+'\n')
    print(f'Preserved epoch {saved["epoch"]}; optimized launcher PID {process.pid}.')
    print(f'Log: {new}/launch.log. Check it for GPU benchmark and epoch progress.')


if __name__=='__main__':
    main()
