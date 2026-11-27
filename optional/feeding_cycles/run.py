"""Bounded optional feeding-cycle comparison using the existing headless runner."""
import argparse
import copy
import csv
from importlib.metadata import distributions
import itertools
import json
import os
from pathlib import Path
import platform
import re
import shlex
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'external/coralai'),
               str(ROOT / 'external/coralai/coralai/dependencies/PyTorch-NEAT'),
               str(ROOT / 'external/coralai/experiments/coral_dev')]
ENV = {'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'TI_CPU_MAX_NUM_THREADS': '1',
       'MPLCONFIGDIR': str(HERE / '.local/matplotlib'),
       'TI_OFFLINE_CACHE_FILE_PATH': str(HERE / '.local/taichi')}
os.environ.update(ENV)

import numpy as np
from PIL import Image, ImageDraw
import torch
from environments import make_env
from coralai.provenance import effective_configuration
from scripts.compare_trail_reference import (run, read_json, save_json, require,
    load_state, load_resource, headless_image, sha256, write_csv)
from scripts.analyze_food_movement import initial_equality, rows
from scripts.demo_controller_lookup_fix import event_analysis
from scripts.render_food_gui_style import (checked_rgb, viewer_for_saved_states,
    state_panel, centered, legend, encode, PAGE, ORANGE, WHITE)

NAMES = ('A_fixed_steady', 'B_fixed_pulsed', 'C_moving_steady', 'D_moving_pulsed')
LABELS = ('A | Fixed source, steady', 'B | Fixed source, pulsed',
          'C | Moving source, steady', 'D | Moving source, pulsed')
UPDATES = list(range(0, 1001, 25))


def phase(name, update):
    if update == 0:
        return 'Initial'
    if 'steady' in name:
        return 'Steady'
    return 'Feeding' if (update - 1) % 200 < 100 else 'Paused'


def preflight(configs, protocol):
    base = read_json(ROOT / protocol['reference_config'])
    for name, config in configs.items():
        expected = copy.deepcopy(base)
        expected['env_options']['shift_interval'] = 125 if 'moving' in name else 0
        if 'pulsed' in name:
            expected['env_options']['release_schedule'] = dict(
                cycle_updates=200, feeding_updates=100, rate_multiplier=2.0)
        require(config == expected, 'A setting outside the specified treatments changed')
    env = make_env('trail', 4, configs[NAMES[0]]['env_options'])
    mask, _ = env._build_trail(400, 400, torch.device('cpu'))
    ys = torch.where(mask > 0)[1]
    require(int(ys.max()) + 16 < 400, 'Movement would wrap reserves')
    return {'initial_source_cells': int(mask.sum()),
            'initial_y_bounds_inclusive': [int(ys.min()), int(ys.max())],
            'final_moving_y_bounds_inclusive': [int(ys.min()) + 16, int(ys.max()) + 16],
            'nominal_requested_release': int(mask.sum()) * 20.0, 'wraps': 0}


def plot_results(output, protocol, summary):
    """Check compact evidence and plot the recorded points, without a rollout."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.ticker import FuncFormatter, MultipleLocator

    path = output / 'measurements.csv'
    recorded = summary.get('artifacts', {}).get(path.name)
    if recorded:
        require(sha256(path) == recorded['sha256'], 'Measurements changed since the recorded analysis')
    with path.open(newline='') as stream:
        records = list(csv.DictReader(stream))
    data = {n: {} for n in NAMES}
    feeding_updates = {u for lo, hi in protocol['feeding_update_ranges_inclusive'] for u in range(lo, hi+1)}
    paused_updates = {u for lo, hi in protocol['paused_update_ranges_inclusive'] for u in range(lo, hi+1)}
    require(not feeding_updates & paused_updates and feeding_updates | paused_updates == set(range(1, 1001)),
            'Recorded feeding/pause ranges must partition updates 1..1000')
    require(all((phase(NAMES[1], u) == 'Feeding') == (u in feeding_updates) for u in range(1, 1001)),
            'Recorded ranges disagree with completed-update release timing')
    for row in records:
        name, update, count = row['condition'], float(row['completed_update']), float(row['occupied_cells'])
        require(name in data and update.is_integer() and count.is_integer() and 0 <= count <= 160000,
                'Invalid measured condition, update, or occupied-cell count')
        update = int(update)
        require(update not in data[name], 'Duplicate measurement')
        require(row['release_phase'] == phase(name, update), 'CSV phase label is misaligned')
        data[name][update] = row
    require(all(sorted(points) == UPDATES for points in data.values()), 'Expected all 41 captures for each condition')
    for observation in summary['matched_observations']:
        for name, count in observation['occupied_cells'].items():
            require(int(float(data[name][observation['completed_update']]['occupied_cells'])) == count,
                    'CSV count disagrees with the recorded matched observation')
    for name in NAMES:
        for key in ('cumulative_requested', 'cumulative_delivered', 'cumulative_clipped', 'remaining_capacity'):
            require(float(data[name][1000][key]) == summary['resource_budgets'][name][key], 'Recorded resource budget changed')
    values = {n: np.array([float(data[n][u]['occupied_cells']) for u in UPDATES]) for n in NAMES}
    feed_ends = protocol['feeding_end_comparisons']
    pause_ends = protocol['cycle_end_comparisons']
    counts = {u: {n: int(float(data[n][u]['occupied_cells'])) for n in NAMES} for u in UPDATES}
    feeding_higher = all(counts[u][pulse] > counts[u][steady] for u in feed_ends
                         for steady, pulse in ((NAMES[0], NAMES[1]), (NAMES[2], NAMES[3])))
    paused_lower = all(counts[u][pulse] < counts[u][steady] for u in pause_ends
                       for steady, pulse in ((NAMES[0], NAMES[1]), (NAMES[2], NAMES[3])))
    interactions = [(counts[u][NAMES[3]]-counts[u][NAMES[1]]) -
                    (counts[u][NAMES[2]]-counts[u][NAMES[0]]) for u in pause_ends]
    delivered = [float(data[n][1000]['cumulative_delivered']) for n in NAMES]
    require(max(delivered)-min(delivered) < summary['checks']['material_delivery_tolerance'], 'Material delivery mismatch')

    colors = ('#2468A2', '#CA5600')
    shade = '#8CBFA3'
    spans = [(lo-0.5, hi+0.5) for lo, hi in protocol['feeding_update_ranges_inclusive']]
    full_top = np.ceil(max(v.max() for v in values.values()) / 10000) * 10000
    later = np.array(UPDATES) >= 200
    inset_limits = (np.floor(min(v[later].min() for v in values.values()) / 1000)*1000-1000,
                    np.ceil(max(v[later].max() for v in values.values()) / 1000)*1000+1000)
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.titleweight': 'bold'}):
        figure, axes = plt.subplots(1, 2, figsize=(14.4, 6.8), sharex=True, sharey=True)
        figure.subplots_adjust(left=0.075, right=0.985, bottom=0.22, top=0.79, wspace=0.14)
        figure.suptitle('Food timing changes occupied-cell trajectories', y=0.98, fontsize=19, weight='bold')
        figure.text(0.5, 0.92, 'Exploratory • one matched starting world • 400 × 400 • seed 42', ha='center', color='#46515A')
        for axis, title, names in zip(axes, ('(a) Fixed source', '(b) Moving source'), (NAMES[:2], NAMES[2:])):
            inset = axis.inset_axes([0.35, 0.46, 0.61, 0.43])
            for target in (axis, inset):
                for left, right in spans:
                    target.axvspan(left, right, color=shade, alpha=0.24, linewidth=0, zorder=0)
                for name, color in zip(names, colors):
                    line, = target.plot(UPDATES, values[name], color=color, linewidth=1.9,
                                        marker='o', markersize=2.8, label=name.split('_')[-1].capitalize())
                    require(np.array_equal(line.get_xdata(), UPDATES) and np.array_equal(line.get_ydata(), values[name]),
                            'Plotted coordinates differ from CSV measurements')
                target.grid(axis='y', color='#DCE1E5', linewidth=0.7, zorder=0)
            axis.set(title=title, xlim=(0, 1000), ylim=(0, full_top))
            axis.xaxis.set_major_locator(MultipleLocator(200))
            axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:,.0f}'))
            inset.set(xlim=(200, 1000), ylim=inset_limits, title='Later cycles: updates 200–1000')
            inset.title.set(fontsize=9, weight='normal')
            inset.tick_params(labelsize=8, pad=2)
            inset.xaxis.set_major_locator(MultipleLocator(200))
            inset.yaxis.set_major_locator(MultipleLocator(5000))
            inset.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value/1000:g}k'))
            axis.indicate_inset_zoom(inset, edgecolor='#78848C', alpha=0.5, linewidth=0.8)
        axes[0].set_ylabel('Occupied cells (genome >= 0)', labelpad=10)
        handles, labels = axes[0].get_legend_handles_labels()
        figure.legend(handles+[Patch(facecolor=shade, alpha=0.35)], labels+['Pulsed feeding period'],
                      loc='upper center', bbox_to_anchor=(0.5, 0.89), ncol=3, frameon=False)
        figure.supxlabel('Completed simulation updates', y=0.14, fontsize=12)
        figure.text(0.5, 0.082, 'Shading describes the pulsed schedule; steady feeding continues throughout. Feed: 1–100, then pause: 101–200; repeat.',
                    ha='center', fontsize=10)
        figure.text(0.5, 0.04, 'All recorded 25-update measurements are shown; lines connect points without smoothing. Insets expand the same later-cycle data.',
                    ha='center', fontsize=10, color='#46515A')
        figure.savefig(output / 'feeding_cycles.png', dpi=170, facecolor='white')
        plt.close(figure)
    return {'input_sha256': sha256(path), 'plotted_measurements_verified': len(records),
            'feeding_spans_half_update_edges': spans, 'full_axes': {'x': [0, 1000], 'y': [0, float(full_top)]},
            'inset_axes': {'x': [200, 1000], 'y': [float(v) for v in inset_limits]},
            'boundary_alignment': 'Integer updates 1..100 feed and 101..200 pause; shading edges at 0.5/100.5, 200.5/300.5, etc. Update 0 is unshaded.',
            'verified_counts': {str(u): counts[u] for u in (900, 1000)},
            'pulsed_higher_at_every_feeding_end': feeding_higher, 'pulsed_lower_at_every_pause_end': paused_lower,
            'complete_cycle_interactions': interactions, 'interaction_changes_sign': min(interactions) < 0 < max(interactions),
            'final_delivered_range': max(delivered)-min(delivered)}


def analyze(output, configs, protocol):
    local = output / 'local'
    runs = {n: local / n / 'run' for n in NAMES}
    checks = initial_equality(runs)
    totals, cumulative = event_analysis(local, configs)
    logs = {n: {int(r['completed_update']): r for r in rows(p / 'resource_metrics.csv')}
            for n, p in runs.items()}
    initial = load_resource(runs[NAMES[0]], 0)['capacity']
    support = int((initial > 0).sum())
    tolerance = protocol['accounting']
    budgets, phases = {}, []
    max_replay_error = max_addition_error = 0.0
    for name in NAMES:
        require(list(logs[name]) == list(range(1001)), 'Missing completed updates')
        expected = torch.from_numpy(initial.copy())
        requested = 0.0
        reserve_bound = 0.0
        half_ulp = float(np.spacing(initial.max())) / 2
        for update in range(1001):
            if update:
                if 'moving' in name and update % 125 == 0:
                    expected = torch.roll(expected, 2, dims=1)
                rate = 0.0 if phase(name, update) == 'Paused' else (0.04 if 'pulsed' in name else 0.02)
                injection = expected.clamp(max=rate)
                requested += float(injection.double().sum())
                expected.sub_(injection).clamp_(min=0)
                if rate: reserve_bound += half_ulp * support
            m = logs[name][update]
            error = max(abs(m['cumulative_requested'] - requested),
                        abs(m['remaining_capacity'] - float(expected.double().sum())))
            max_replay_error = max(max_replay_error, error)
            require(error < tolerance['exact_float32_replay_absolute_tolerance'], 'Reserve replay mismatch')
            addition_error = abs(m['cumulative_requested'] - m['cumulative_delivered'] - m['cumulative_clipped'])
            max_addition_error = max(max_addition_error, addition_error)
            require(addition_error < tolerance['delivery_accounting_absolute_tolerance'], 'Unexplained supply difference')
            require(m['cumulative_clipped'] == 0, 'Energy clipped during injection')
            require(m['occupied_cells'] == m['occupied_inside_source'] + m['occupied_outside_source'], 'Bad occupancy partition')
            if update in UPDATES:
                saved = load_resource(runs[name], update)
                require(np.array_equal(saved['capacity'], expected.numpy()), 'Reserve location mismatch')
                if update:
                    require(np.array_equal(saved['last_injection'], injection.numpy()), 'Release timing/location mismatch')
            if update and phase(name, update) == 'Paused':
                require(m['cumulative_requested'] == logs[name][update-1]['cumulative_requested'], 'Release during pause')
        b = read_json(runs[name] / 'meta.json')['resource_budget']
        b['reserve_depletion_minus_requested'] = b['initial_capacity'] - b['remaining_capacity'] - b['cumulative_requested']
        b['reserve_rounding_bound'] = reserve_bound
        b['requested_minus_delivered'] = b['cumulative_requested'] - b['cumulative_delivered']
        require(abs(b['reserve_depletion_minus_requested']) <= reserve_bound, 'Unexplained reserve loss')
        budgets[name] = b
        for start in range(0, 1000, 100):
            end = start + 100
            a, z = logs[name][start], logs[name][end]
            phases.append({'condition': name, 'updates_inclusive': [start+1, end],
                'cycle_half': 'Feeding window' if start % 200 == 0 else 'Pause window',
                'release_phase': phase(name, end),
                'occupied_start': int(a['occupied_cells']), 'occupied_end': int(z['occupied_cells']),
                'occupancy_change': int(z['occupied_cells'] - a['occupied_cells']),
                'delivered': z['cumulative_delivered'] - a['cumulative_delivered'],
                'colonization_gains': cumulative[name][end]['local'] - cumulative[name][start]['local'],
                'radiation_gains': cumulative[name][end]['forced'] - cumulative[name][start]['forced']})
    nominal = support * 20.0
    material = nominal * tolerance['material_release_difference_relative_tolerance']
    request_range = max(b['cumulative_requested'] for b in budgets.values()) - min(b['cumulative_requested'] for b in budgets.values())
    delivery_range = max(b['cumulative_delivered'] for b in budgets.values()) - min(b['cumulative_delivered'] for b in budgets.values())
    require(request_range < material and delivery_range < material, 'Material final supply mismatch; treatment claims withheld')
    require(all(abs(b['cumulative_requested'] - nominal) < material for b in budgets.values()), 'Nominal release mismatch')
    pairs, measurements, previous, pre_movement = [], [], {}, []
    for update in UPDATES:
        occupied, states = {}, {}
        for name in NAMES:
            state = states[name] = load_state(runs[name], update)
            require(tuple(state.shape) == (1, 14, 400, 400) and bool(torch.isfinite(state).all()), 'Bad saved state')
            occupied[name] = state[0, 13].numpy() >= 0
            m = logs[name][update]
            source = load_resource(runs[name], update)['capacity'] > 0
            require(int(occupied[name].sum()) == m['occupied_cells'], 'Occupancy timing mismatch')
            require(int((occupied[name] & source).sum()) == m['occupied_inside_source'], 'Source/count mismatch')
            for channel, key in ((0, 'total_energy'), (1, 'total_infra')):
                require(abs(float(state[0, channel].double().sum()) - m[key]) < 1e-7, 'State/log mismatch')
            with Image.open(runs[name] / 'screenshots' / f'step_{update:07d}.png') as image:
                require(np.array_equal(np.asarray(image), np.asarray(headless_image(state))), 'Saved screenshot mismatch')
            if update == 1000:
                require(torch.equal(state, torch.load(runs[name] / 'checkpoint_0001000/substrate.pt', weights_only=True)), 'Final checkpoint mismatch')
            reference = NAMES[0] if name in NAMES[:2] else NAMES[2]
            motion_reference = NAMES[0] if 'steady' in name else NAMES[1]
            measurements.append({'condition': name, **m, 'release_phase': phase(name, update),
                'source_displacement_y': 2 * (update // 125) if 'moving' in name else 0,
                'changed_occupied_locations_since_previous_capture': int((occupied[name] != previous[name]).sum()) if name in previous else 0,
                'cumulative_colonization_gains': cumulative[name][update]['local'],
                'cumulative_radiation_gains': cumulative[name][update]['forced'],
                'differing_locations_vs_same_motion_steady': int((occupied[name] != occupied[reference]).sum()),
                'differing_locations_vs_same_schedule_fixed': int((occupied[name] != occupied[motion_reference]).sum()),
                'fixed_occupancy_in_this_source_mask': int((occupied[motion_reference] & source).sum())})
            previous[name] = occupied[name].copy()
        for a, b in itertools.combinations(NAMES, 2):
            either = occupied[a] | occupied[b]
            pairs.append({'completed_update': update, 'a': a, 'b': b,
                'differing_occupied_locations': int((occupied[a] != occupied[b]).sum()),
                'jaccard': float((occupied[a] & occupied[b]).sum() / either.sum()) if either.any() else 1.0})
        if update < 125:
            pre_movement.append({'update': update, **{schedule: float((states[a] - states[b]).abs().max())
                for schedule, a, b in (('steady', NAMES[0], NAMES[2]), ('pulsed', NAMES[1], NAMES[3]))}})
    write_csv(output / 'measurements.csv', measurements)
    comparisons = []
    for update in range(100, 1001, 100):
        counts = {n: int(logs[n][update]['occupied_cells']) for n in NAMES}
        steady_effect = counts[NAMES[2]] - counts[NAMES[0]]
        pulsed_effect = counts[NAMES[3]] - counts[NAMES[1]]
        comparisons.append({'completed_update': update, 'cycle_half': 'Feeding end' if update % 200 else 'Complete cycle / pause end',
            'occupied_cells': counts, 'moving_minus_fixed_steady': steady_effect,
            'moving_minus_fixed_pulsed': pulsed_effect,
            'movement_x_pulsing_difference_of_differences': pulsed_effect - steady_effect})
    checks.update(per_update_resource_checks=4004, snapshot_checks=164, event_checks=4000,
        max_float32_reserve_replay_error=max_replay_error, max_injection_accounting_residual=max_addition_error,
        pre_first_movement_state_differences=pre_movement,
        nominal_requested_release=nominal, final_requested_range=request_range,
        final_delivered_range=delivery_range, material_delivery_tolerance=material,
        equal_requested_and_delivered_within_tolerance=True, clipping=0, wraps=0)
    return {'checks': checks, 'resource_budgets': budgets, 'phase_comparisons': phases,
            'matched_observations': comparisons, 'spatial_comparisons': pairs,
            'cell_event_totals': totals}, logs


def render(output, logs, replace=False):
    movie = output / 'overview.mp4'
    require(replace or not movie.exists(), 'Refusing to overwrite overview.mp4')
    if replace:
        backup = output / 'local/previous_media'
        backup.mkdir(exist_ok=True)
        for name in ('overview.mp4', 'endpoint.png'):
            target = backup / name
            if not target.exists():
                target.write_bytes((output / name).read_bytes())
    frames = output / 'local/overview_frames'
    frames.mkdir(exist_ok=True)
    vis = viewer_for_saved_states()
    max_error = 0.0
    for update in UPDATES:
        canvas = Image.new('RGB', (1760, 2080), PAGE)
        draw = ImageDraw.Draw(canvas)
        centered(draw, 'Optional feeding cycles | 400 x 400 | seed 42', 880, 14, 34, True)
        centered(draw, f'Completed update {update:04d} / 1000', 880, 62, 30)
        for index, (name, label) in enumerate(zip(NAMES, LABELS)):
            x, y = 40 + (index % 2) * 880, 118 + (index // 2) * 920
            centered(draw, label, x+400, y, 27, True)
            status = phase(name, update)
            rate = '0 / source cell' if status == 'Paused' else ('0.04 / source cell' if status == 'Feeding' else '0.02 / source cell')
            if status == 'Initial': rate = 'release starts at update 1'
            centered(draw, f'{status} | {rate}', x+400, y+39, 24, color=ORANGE if 'pulsed' in name else WHITE)
            run_dir = output / 'local' / name / 'run'
            state = load_state(run_dir, update)
            resource = load_resource(run_dir, update)
            rgb, error, _ = checked_rgb(state, vis)
            max_error = max(max_error, error)
            panel, _ = state_panel(rgb, resource['capacity'].T > 0, 'overview')
            canvas.paste(panel, (x, y+78))
            centered(draw, f"Occupied cells (genome >= 0): {int(logs[name][update]['occupied_cells']):,}", x+400, y+880, 23)
        legend(draw, 880, 1982, 21, 'gui')
        centered(draw, 'Auto-scaled E / infrastructure / rotation. Brightness and colored pixels are not population.', 880, 2024, 20)
        path = frames / f'update_{update:04d}.png'
        canvas.save(path)
        if update == 1000: canvas.save(output / 'endpoint.png')
    timeline = [0] * 10 + [u for u in UPDATES[1:] for _ in range(4)] + [1000] * 20
    with tempfile.TemporaryDirectory(dir=output / 'local', prefix='render-') as temporary:
        temporary = Path(temporary)
        encoded = temporary / 'overview.mp4'
        command = encode(frames, timeline, encoded, temporary / 'sequence')
        info = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-count_frames', '-show_entries',
            'stream=width,height,nb_read_frames,r_frame_rate', '-of', 'json', str(encoded)], text=True))
        require(int(info['streams'][0]['nb_read_frames']) == len(timeline), 'Video frame count mismatch')
        decoded = temporary / 'decoded_endpoint.png'
        subprocess.run(['ffmpeg', '-v', 'error', '-i', str(encoded), '-vf', f'select=eq(n\\,{len(timeline)-1})',
            '-frames:v', '1', str(decoded)], check=True)
        require(Image.open(decoded).size == (1760, 2080), 'Decoded endpoint size mismatch')
        encoded.replace(movie)
        decoded.replace(output / 'local/decoded_endpoint.png')
    return {'command': command, 'probe': info, 'max_gui_kernel_rgb_error': max_error,
            'unique_synchronized_states': len(UPDATES), 'duration_seconds': len(timeline)/10,
            'phase_labels': 'Release at the displayed completed update; initial state has no release yet.',
            'fallback': 'endpoint.png is the exact lossless source of the final video frame.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--analyze-only', action='store_true', help='Verify compact results and regenerate the figure; no raw captures or simulation')
    parser.add_argument('--render-media', action='store_true', help='With --analyze-only, also regenerate media from retained local captures')
    args = parser.parse_args()
    output = args.output_dir.resolve()
    require(output.is_relative_to(HERE / 'results'), 'Keep outputs in optional/feeding_cycles/results/')
    configs = {n: read_json(HERE / 'configs' / f'{n}.json') for n in NAMES}
    protocol = read_json(HERE / 'configs/protocol.json')
    design = preflight(configs, protocol)
    if args.render_media and not args.analyze_only:
        parser.error('--render-media requires --analyze-only; fresh runs already render media')
    if args.analyze_only:
        result = read_json(output / 'summary.json')
        require(result['protocol'] == protocol, 'Saved protocol differs from the configured experiment')
        require(result['preflight'] == design, 'Saved design differs from the configured experiment')
        for name, record in result['artifacts'].items():
            require(sha256(output / name) == record['sha256'], f'Recorded artifact changed: {name}')
        result['figure'] = plot_results(output, protocol, result)
        if args.render_media:
            logs = {n: {int(r['completed_update']): r for r in rows(output / 'local' / n / 'run/resource_metrics.csv')}
                    for n in NAMES}
            result['review_media'] = render(output, logs, replace=True)
        result.setdefault('review', {}).update({'analysis_command': 'optional/feeding_cycles/.local/venv/bin/python optional/feeding_cycles/run.py --output-dir '
            + str(output.relative_to(ROOT)) + ' --analyze-only',
            'analysis_source_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in
                (HERE / 'run.py', HERE / 'test_feeding_cycles.py', HERE / 'README.md')},
            'simulation_execution_record': 'Original provenance, checks, resource_budgets, and float32 residuals are preserved; this analysis ran no simulation.'})
        for name in ('feeding_cycles.png', *(['overview.mp4', 'endpoint.png'] if args.render_media else [])):
            path = output / name
            result['artifacts'][name] = {'bytes': path.stat().st_size, 'sha256': sha256(path)}
        save_json(output / 'summary.json', result)
        print(f'Compact analysis and figure saved: {output}', flush=True)
        return
    output.mkdir(parents=True, exist_ok=False)
    (output / 'local').mkdir()
    # Persist the complete protocol and configs before the timing or outcome runs.
    save_json(output / 'summary.json', {'status': 'Protocol fixed; outcomes not yet inspected',
        'protocol': protocol, 'preflight': design})
    save_json(output / 'effective_configs.json', {n: effective_configuration(c) for n, c in configs.items()})
    calibration = copy.deepcopy(configs[NAMES[0]])
    calibration.update(steps=50, max_screenshots=3, checkpoint_interval=50)
    run(calibration, output / 'local/timing')
    timing = read_json(output / 'local/timing/run/meta.json')
    estimate = timing['elapsed_seconds'] * 80
    print(f'50-update rollout: {timing["elapsed_seconds"]:.2f}s; four-run estimate {estimate:.1f}s', flush=True)
    require(estimate < 3600, 'Estimated simulation time exceeds one hour; report feasibility before proceeding')
    for name, config in configs.items():
        print(f'Running {name}: 400 x 400, 1000 completed updates', flush=True)
        run(config, output / 'local' / name)
    save_json(output / 'effective_configs.json', {n: read_json(output / 'local' / n / 'run/meta.json')['effective_config'] for n in NAMES})
    result, logs = analyze(output, configs, protocol)
    result.update(status='Complete; exploratory results pending review', protocol=protocol, preflight=design)
    matched = result['matched_observations']
    feeding = [r for r in matched if r['completed_update'] % 200]
    paused = [r for r in matched if not r['completed_update'] % 200]
    phase_pattern = all(r['occupied_cells'][pulse] > r['occupied_cells'][steady]
        for r in feeding for steady, pulse in ((NAMES[0], NAMES[1]), (NAMES[2], NAMES[3]))) and all(
        r['occupied_cells'][pulse] < r['occupied_cells'][steady]
        for r in paused for steady, pulse in ((NAMES[0], NAMES[1]), (NAMES[2], NAMES[3])))
    interactions = [r['movement_x_pulsing_difference_of_differences'] for r in paused]
    interaction_pattern = ('changes sign' if min(interactions) < 0 < max(interactions)
                           else 'does not change sign in this run')
    last = matched[-1]
    spatial = {(r['a'], r['b']): r['differing_occupied_locations']
               for r in result['spatial_comparisons'] if r['completed_update'] == 1000}
    result['interpretation'] = {
        'question_in_plain_language': 'We previously changed where food appeared. This optional experiment changes when it becomes available and tests whether that interacts with movement.',
        'phase_response': ('At every sampled feeding end, pulsed occupancy exceeds steady occupancy for both motion settings; at every complete-cycle/pause end it is lower.'
            if phase_pattern else 'The feeding-end and pause-end rankings are mixed; inspect matched_observations.') + ' Counts use genome >= 0.',
        'movement_interaction': f"The movement-by-pulsing difference of differences {interaction_pattern} across complete cycles. At update 1000, moving minus fixed is {last['moving_minus_fixed_steady']} occupied cells under steady release and {last['moving_minus_fixed_pulsed']} under pulsed release (interaction {last['movement_x_pulsing_difference_of_differences']:+d}). One seed cannot establish a general interaction.",
        'spatial_response': f'At update 1000, fixed versus moving differs at {spatial[NAMES[0], NAMES[2]]} occupied locations under steady release and {spatial[NAMES[1], NAMES[3]]} under pulsed release; endpoint totals do not describe spatial occupancy.',
        'usefulness': ('Useful as an optional demonstration that feeding phase changes the apparent endpoint ranking.' if phase_pattern
            else 'Phase-aware observations remain necessary to interpret these exploratory results.') + ' Keep exploratory pending review; evidence for a general movement interaction is insufficient.'}
    result['media'] = render(output, logs)
    result['figure'] = plot_results(output, protocol, result)
    timing = read_json(output / 'local/timing/run/meta.json')
    result['timing'] = {'pilot_updates': 50, 'grid': [400, 400], 'pilot_seconds': timing['elapsed_seconds'],
        'four_full_run_estimate_seconds': timing['elapsed_seconds'] * 80,
        'actual_rollout_seconds': {n: read_json(output / 'local' / n / 'run/meta.json')['elapsed_seconds'] for n in NAMES}}
    sources = [HERE / 'run.py', HERE / 'test_feeding_cycles.py', *(HERE / 'configs').glob('*.json')]
    result['provenance'] = {'source_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'branch': subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip(),
        'worktree': str(ROOT), 'source_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in sources},
        'simulator': read_json(output / 'local' / NAMES[0] / 'run/meta.json')['simulator_provenance'],
        'versions': {d.metadata['Name']: d.version for d in distributions()},
        'python': sys.version, 'platform': platform.platform(), 'environment': ENV,
        'ffmpeg': subprocess.check_output(['ffmpeg', '-version'], text=True).splitlines()[0],
        'run_command': shlex.join([sys.executable, '-B', 'optional/feeding_cycles/run.py', '--output-dir', str(output.relative_to(ROOT))]),
        'commands': {n: read_json(output / 'local' / n / 'command.json') for n in ('timing', *NAMES)},
        'reproduction_command': 'optional/feeding_cycles/.local/venv/bin/python optional/feeding_cycles/run.py --output-dir optional/feeding_cycles/results/seed42_reproduction'}
    validation = HERE / '.local/validation.txt'
    if validation.exists():
        text = validation.read_text()
        result['focused_validation'] = {'result': re.findall(r'Ran \d+ tests in [\d.]+s\n\n(?:OK|FAILED[^\n]*)', text),
            'command': "PYTHONPATH='tests:external/coralai:external/coralai/coralai/dependencies/PyTorch-NEAT:.' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TI_CPU_MAX_NUM_THREADS=1 optional/feeding_cycles/.local/venv/bin/python -m unittest optional.feeding_cycles.test_feeding_cycles test_controller_lookup test_controller_integration test_dynamic_energy test_food_presentation -q",
            'log_sha256': sha256(validation)}
    result['limitations'] = ['One seed and one bounded schedule; no causal mechanism or universal strategy ranking established.',
        'Serial CPU requested; Taichi RNG/kernel/atomic ordering is not fully captured by Python/NumPy/Torch RNG equality.',
        'Float32 reserve subtraction and energy addition residuals are retained without correction; see resource_budgets.',
        'Initial background infrastructure and stored energy can buffer pauses; radiation reseeds throughout.',
        'Captures are every 25 updates; per-update event/resource logs verify intervening feeding and movement timing.']
    result['artifacts'] = {p.name: {'bytes': p.stat().st_size, 'sha256': sha256(p)}
        for p in (output / 'effective_configs.json', output / 'measurements.csv', output / 'overview.mp4', output / 'endpoint.png', output / 'feeding_cycles.png')}
    save_json(output / 'summary.json', result)
    print(f'Complete: {output}', flush=True)


if __name__ == '__main__':
    main()
