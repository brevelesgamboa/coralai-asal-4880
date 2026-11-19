"""Run a bounded three-speed trail experiment; no model or ASAL search."""
import argparse
import copy
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'external/coralai'),
                str(ROOT / 'external/coralai/coralai/dependencies/PyTorch-NEAT'),
                str(ROOT / 'external/coralai/experiments/coral_dev')]
from scripts.compare_trail_reference import audit_defaults, read_json, require, run, save_json, sha256

NAMES = ('fixed', 'slower', 'faster')
ENV = {'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'TI_CPU_MAX_NUM_THREADS': '1',
       'MPLCONFIGDIR': str(ROOT / '.cache/matplotlib'),
       'TI_OFFLINE_CACHE_FILE_PATH': str(ROOT / '.cache/taichi-food')}


def configurations():
    return {name: read_json(ROOT / 'configs/food_movement_comparison' / f'{name}.json') for name in NAMES}


def preflight(output, configs):
    from environments import make_env
    pair = {n: read_json(ROOT / f'configs/trail_reference_comparison/{n}.json') for n in ('fixed', 'dynamic')}
    audit_defaults(pair, output)
    (output / 'reference.json').replace(output / 'baseline_reference.json')
    env = make_env('trail', 4, configs['fixed']['env_options'])
    trail, progress = env._build_trail(400, 400, torch.device('cpu'))
    reserve = trail * env.max_capacity * (1 + env.goal_boost * progress)
    ys = torch.where(reserve > 0)[1]
    (output / 'configs').mkdir(exist_ok=True)
    schedules = {}
    for name, config in configs.items():
        expected = copy.deepcopy(configs['fixed'])
        interval = config['env_options']['shift_interval']
        expected['env_options']['shift_interval'] = interval
        require(config == expected, 'Treatment changed a setting besides movement frequency')
        displacement = 2 * (1000 // interval) if interval else 0
        require(int(ys.max()) + displacement < 400, 'Schedule would cross the boundary')
        for offset in range(0, displacement + 1, 2):
            shifted = torch.roll(reserve, offset, dims=1)
            require(float(shifted.double().sum()) == float(reserve.double().sum()), 'Translation lost reserves')
            if offset:
                require(not bool((shifted[:, :offset] > 0).any()), 'Schedule wraps reserves')
        schedules[name] = {'shift_interval_updates': interval, 'shift_distance_cells': 2,
            'direction': '+y', 'mean_cells_per_update': 2 / interval if interval else 0,
            'translation_updates': list(range(interval, 1001, interval)) if interval else [],
            'final_displacement_y_cells': displacement,
            'final_y_bounds_inclusive': [int(ys.min()) + displacement, int(ys.max()) + displacement]}
        save_json(output / 'configs' / f'{name}.json', config)
    save_json(output / 'design.json', {'grid': [400, 400], 'seed': 42, 'completed_updates': 1000,
        'initial_y_bounds_inclusive': [int(ys.min()), int(ys.max())], 'schedules': schedules,
        'treatment': 'Only movement frequency changes; slower/faster refer to food reserves, not cells.',
        'selection': '125 is the established schedule; 250 halves its frequency. Faster than 125 risks crossing the edge; no route or boundary change is introduced.',
        'initial_reserve_energy_units': float(reserve.double().sum()), 'resource_wraps': 0,
        'units': 'Grid cells, completed updates, simulator energy and infrastructure units.',
        'source_region': 'Remaining reserve > 0 after shift and release; distinct from available substrate energy.',
        'phase': 'After NN/physics/evolution/chunk death, resource shift/release, radiation, cleanup/cull.',
        'initialization': 'Independent seeded initialization with exact tensor/controller/registry/Python/NumPy/Torch RNG equality verified from initial checkpoints. Trail resume is unsupported.',
        'limits': 'Single seed. Serial CPU requested; atomic/kernel ordering may still vary. Occupancy changes do not establish learning, adaptation or individual locomotion.'})


def provenance(output):
    files = [str(p.relative_to(ROOT)) for p in (ROOT / 'external/coralai/coralai').glob('*.py')]
    files += [str(p.relative_to(ROOT)) for p in (ROOT / 'external/coralai/experiments/coral_dev').glob('*.py')]
    files += ['external/coralai/experiments/coral_dev/neat.config',
              'scripts/compare_food_movement.py', 'scripts/analyze_food_movement.py', 'tests/test_food_movement.py', '.gitignore',
              'scripts/compare_trail_reference.py', 'scripts/demo_controller_lookup_fix.py',
              'scripts/evaluate_survival.py', 'configs/trail_reference_comparison/fixed.json',
              'configs/trail_reference_comparison/dynamic.json']
    files += [f'configs/food_movement_comparison/{name}.json' for name in NAMES]
    save_json(output / 'provenance.json', {
        'baseline_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'branch': subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip(),
        'baseline_remote': 'origin/main; e7491627d5099e844a91f761f3f1843d89fc7fc7 verified with git ls-remote',
        'controller_fix_commit': '6eedc6c52f8c282876cbdafd8494fd5d0682bb12 (equivalent to 9777ef0)',
        'source_sha256': {f: sha256(ROOT / f) for f in sorted(set(files)) if (ROOT / f).exists()},
        'versions': {p: version(p) for p in ('torch', 'taichi', 'numpy', 'Pillow', 'neat-python', 'matplotlib')},
        'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(),
        'environment': ENV, 'invocation': shlex.join([sys.executable, *sys.argv]),
        'child_commands': {n: read_json(output / 'local' / n / 'command.json') for n in ('timing', *NAMES)},
        'simulator_provenance': {n: read_json(output / 'local' / n / 'run/meta.json')['simulator_provenance'] for n in NAMES},
        'reuse': 'Unmodified trail environment, corrected controllers, headless runner, comparison run/load/RGB helpers and event_analysis instrumentation.',
        'new_this_week': 'Three-frequency design, matched-mask controls and exact three-way analysis; presentation is a separate saved-state command.',
        'nondeterminism': 'Serial CPU environment requested; later numerical identity is not guaranteed. Initial states and available RNG states are verified, and pre-movement divergence is measured.'})
    (output / 'requirements-used.txt').write_text(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True))
    (output / 'reproduce.sh').write_text('''#!/bin/sh
set -eu
# Reuse the existing scientific Python environment.
repo_dir=$(CDPATH= cd -- "$(dirname -- "$0")/../../../.." && pwd)
cd "$repo_dir"
# Essential packages and exact versions are recorded in requirements-used.txt.
# No CLIP/model download or ASAL search is needed.
python_bin=${CORALAI_PYTHON:-.venv-novelty/bin/python}
exec "$python_bin" -B scripts/compare_food_movement.py --output-dir "${1:?Pass a fresh output directory}"
''')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--stage', choices=('all', 'run', 'analyze'), default='all')
    args = p.parse_args()
    output = args.output_dir.resolve()
    os.environ.update(ENV)
    configs = configurations()
    if args.stage != 'analyze':
        output.mkdir(parents=True, exist_ok=False)
        (output / 'local').mkdir()
        preflight(output, configs)
        cal = copy.deepcopy(configs['fixed'])
        cal.update(steps=50, max_screenshots=3, checkpoint_interval=50)
        wall = run(cal, output / 'local/timing')
        elapsed = read_json(output / 'local/timing/run/meta.json')['elapsed_seconds']
        save_json(output / 'timing.json', {'grid': [400, 400], 'updates': 50,
            'wall_seconds': wall, 'simulation_capture_seconds': elapsed,
            'linear_three_run_seconds': elapsed * 60, 'decision': 'Keep required grid, physics and duration.'})
    if args.stage != 'analyze':
        timing = read_json(output / 'timing.json')
        print('Full-size timing:', json.dumps(timing), flush=True)
        wall = {}
        for name, config in configs.items():
            print(f"Running {name}: 400x400, 1000 updates; interval {config['env_options']['shift_interval']}", flush=True)
            wall[name] = run(config, output / 'local' / name)
            print(f'Completed {name} in {wall[name]:.1f}s', flush=True)
        save_json(output / 'completed.json', {'run_wall_seconds': wall})
    if args.stage != 'run':
        from scripts.analyze_food_movement import analyze
        analyze(output, configs)
        provenance(output)
    print('Saved:', output, flush=True)


if __name__ == '__main__':
    main()
