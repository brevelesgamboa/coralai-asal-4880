"""Paired 400x400 reversal experiment using the established headless runner."""
import argparse
import copy
from importlib.metadata import version
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.compare_food_movement import ENV
from scripts.compare_trail_reference import read_json, require, run, save_json, sha256

PREVIOUS = ROOT/'results/week13/food_movement_comparison/seed42_20261118'
NAMES = ('continues','reverses')


def displacement(name, update):
    moves=update//125
    return 2*moves if name=='continues' else 2*(min(moves,4)-max(moves-4,0))


def configurations():
    return {name: read_json(ROOT / 'configs/food_direction_change' / f'{name}.json') for name in NAMES}


def setup(output, configs, reference_dir):
    from scripts.compare_trail_reference import load_resource
    import numpy as np
    initial=load_resource(reference_dir/'local/faster/run',0)['capacity']
    ys=np.where(initial>0)[1]
    for name,config in configs.items():
        save_json(output/'configs'/f'{name}.json',config)
        for u in range(1001):
            delta=displacement(name,u)
            require(int(ys.min())+delta>=0 and int(ys.max())+delta<400,'Resource boundary crossing')
    save_json(output/'design.json',{'grid':[400,400],'seed':42,'completed_updates':1000,
        'continues':{str(u):displacement('continues',u) for u in range(0,1001,125)},
        'reverses':{str(u):displacement('reverses',u) for u in range(0,1001,125)},
        'offset_units':'y cells relative to initial reserves; +y is down',
        'only_treatment_difference':'env_options.reverse_after_update=500, absent for continues',
        'phase':'Movement at 500 is downward in both; first upward movement is 625. Reserve translation before release, after NN/physics/evolution and before radiation/cleanup.',
        'initial_y_bounds':[int(ys.min()),int(ys.max())],'resource_wraps':0,
        'initialization':'Two fresh independent seeded starts; exact tensors, genomes, controller parameters, registries and Python/NumPy/Torch RNG states checked. No incomplete tensor-only fork.',
        'fork_not_used':'Trail resume explicitly unsupported by runner: checkpoints omit environment reserves/state. No reliable complete midpoint fork exists.',
        'capture':'0,25,...,1000, after all completed-update operations; resource and event logs every update',
        'source_region':'remaining reserve > 0, distinct from already available energy',
        'previous_faster_reference':str(reference_dir)})


def provenance(output, reference_dir):
    files=[*ROOT.glob('scripts/*food*.py'),ROOT/'tests/test_food_direction.py',ROOT/'scripts/compare_trail_reference.py',
        ROOT/'scripts/demo_controller_lookup_fix.py',ROOT/'scripts/evaluate_survival.py',ROOT/'scripts/verify_controller_lookup.py']
    files+=list((ROOT/'external/coralai/coralai').glob('*.py'))
    files+=list((ROOT/'external/coralai/experiments/coral_dev').glob('*.py'))
    files+=[ROOT/'external/coralai/experiments/coral_dev/neat.config']
    files += list((ROOT/'configs/food_direction_change').glob('*.json'))
    save_json(output/'provenance.json',{'branch':subprocess.check_output(['git','branch','--show-current'],text=True).strip(),
        'baseline_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'invocation':shlex.join([sys.executable,*sys.argv]),'environment':ENV,'python':sys.version,
        'versions':{n:version(n) for n in ('torch','taichi','numpy','Pillow','neat-python','matplotlib')},
        'source_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in sorted(set(files))},
        'previous_faster_reference':str(reference_dir),
        'run_commands':{n:read_json(output/'local'/n/'command.json') for n in ('timing',*NAMES)},
        'simulator_provenance':{n:read_json(output/'local'/n/'run/meta.json')['simulator_provenance'] for n in NAMES},
        'nondeterminism':'Serial CPU requested; atomic/kernel ordering can vary. Initial equality and actual common-prefix/repeat differences are measured, not assumed.'})
    (output/'requirements-used.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    (output/'source_changes.patch').write_text(subprocess.check_output(
        ['git','diff','HEAD','--','external/coralai/experiments/coral_dev/environments.py'],text=True))
    reference = shlex.quote(os.path.relpath(reference_dir, ROOT))
    (output/'reproduce.sh').write_text('''#!/bin/sh
set -eu
repo_dir=$(CDPATH= cd -- "$(dirname -- "$0")/../../../.." && pwd)
cd "$repo_dir"
python_bin=${CORALAI_PYTHON:-.venv-novelty/bin/python}
reference_dir=REFERENCE_PATH
if [ "$#" -ge 2 ]; then reference_dir=$2; fi
exec "$python_bin" -B scripts/compare_food_direction.py --reference-dir "$reference_dir" --output-dir "${1:?Pass a fresh output directory}"
'''.replace('REFERENCE_PATH', reference))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--reference-dir',type=Path,default=PREVIOUS,help='Completed speed result containing local/faster/run')
    p.add_argument('--stage',choices=('all','run','analyze'),default='all')
    args=p.parse_args()
    output=args.output_dir.resolve()
    os.environ.update(ENV)
    if args.stage!='analyze':
        output.mkdir(parents=True,exist_ok=False)
        for sub in ('local','checks','configs'):(output/sub).mkdir()
    configs=configurations()
    if args.stage!='analyze':
        setup(output,configs,args.reference_dir.resolve())
        calibration=copy.deepcopy(configs['continues'])
        calibration.update(steps=50,max_screenshots=3,checkpoint_interval=50)
        wall=run(calibration,output/'local/timing')
        elapsed=read_json(output/'local/timing/run/meta.json')['elapsed_seconds']
        timing={'grid':[400,400],'completed_updates':50,'wall_seconds':wall,'simulation_capture_seconds':elapsed,
            'linear_pair_estimate_seconds':elapsed*40,'decision':'Keep established grid, duration and physics.'}
        save_json(output/'timing.json',timing)
        print('Timing:',json.dumps(timing),flush=True)
        walls={}
        for name,c in configs.items():
            print('Running',name,flush=True)
            walls[name]=run(c,output/'local'/name)
            print('Completed',name,walls[name],flush=True)
        save_json(output/'completed.json',{'run_wall_seconds':walls})
    if args.stage!='run':
        from scripts.analyze_food_direction import analyze
        analyze(output,configs,args.reference_dir.resolve())
        provenance(output,args.reference_dir.resolve())
    print('Saved',output,flush=True)


if __name__=='__main__':main()
