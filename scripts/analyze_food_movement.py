"""Exact saved-state checks and readable visuals for the food-speed comparison."""
import csv
import itertools
import json
import pickle
import shutil

import numpy as np
from PIL import Image
import torch

from scripts.compare_trail_reference import (headless_image, load_resource, load_state,
    read_json, require, save_json, sha256, write_csv)
from scripts.demo_controller_lookup_fix import event_analysis

NAMES = ('fixed', 'slower', 'faster')


def rows(path):
    with path.open() as stream:
        return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(stream)]


def initial_equality(runs):
    first_run = next(iter(runs.values()))
    base = torch.load(first_run / 'initial_state.pt', weights_only=True, map_location='cpu')
    reserve = load_resource(first_run, 0)['capacity']
    with (first_run / 'checkpoint_0000000/population.pkl').open('rb') as stream:
        first = pickle.load(stream)
    for run in runs.values():
        require(torch.equal(base, torch.load(run / 'initial_state.pt', weights_only=True, map_location='cpu')), 'Initial tensor differs')
        require(np.array_equal(reserve, load_resource(run, 0)['capacity']), 'Initial reserves differ')
        with (run / 'checkpoint_0000000/population.pkl').open('rb') as stream:
            other = pickle.load(stream)
        require(first['genome_keys'] == other['genome_keys'] == list(range(50)), 'Initial registries differ')
        require(pickle.dumps(first['genomes']) == pickle.dumps(other['genomes']), 'Initial genomes differ')
        for key in ('weights', 'biases'):
            require(len(first[key]) == len(other[key]), 'Initial controller counts differ')
            require(all(torch.equal(a, b) for a, b in zip(first[key], other[key])), 'Initial controller tensors differ')
        require(torch.equal(first['rng_state']['torch'], other['rng_state']['torch']), 'Torch RNG differs')
        a, b = first['rng_state']['numpy'], other['rng_state']['numpy']
        require(a[0] == b[0] and np.array_equal(a[1], b[1]) and a[2:] == b[2:], 'NumPy RNG differs')
        require(first['rng_state']['python'] == other['rng_state']['python'], 'Python RNG differs')
    return {'starting_states_bitwise_equal': True, 'starting_reserves_bitwise_equal': True,
        'initial_controller_parameters_equal': True, 'initial_genomes_equal': True,
        'initial_registries_equal': True, 'initial_rng_states_equal': True,
        'initial_registered_controllers': len(first['genomes']),
        'initial_state_sha256': {n: sha256(run / 'initial_state.pt') for n, run in runs.items()}}


def same_mask_counts(reference_occupied, moving_occupied, moving_source, initial_source):
    """Evaluate both ecosystems in the SAME source mask, plus an unmoving mask."""
    return {'moving_occupied_in_current_source': int((moving_occupied & moving_source).sum()),
        'fixed_occupied_in_same_moving_source': int((reference_occupied & moving_source).sum()),
        'moving_minus_fixed_in_same_source': int((moving_occupied & moving_source).sum()) - int((reference_occupied & moving_source).sum()),
        'moving_occupied_in_initial_source': int((moving_occupied & initial_source).sum()),
        'fixed_occupied_in_initial_source': int((reference_occupied & initial_source).sum()),
        'current_source_cells': int(moving_source.sum())}


def analyze(output, configs):
    local = output / 'local'
    measurements = output / 'measurements'
    measurements.mkdir(exist_ok=True)
    steps = configs['fixed']['steps']
    updates = list(range(0, steps + 1, configs['fixed']['screenshot_interval']))
    validation = initial_equality({n: local / n / 'run' for n in NAMES})
    logs = {}
    for name in NAMES:
        for source, target in (('resource_metrics.csv', f'{name}_resources.csv'), ('cell_events.csv', f'{name}_events.csv')):
            shutil.copy2(local / name / 'run' / source, measurements / target)
        logs[name] = {int(r['completed_update']): r for r in rows(local / name / 'run/resource_metrics.csv')}
        require(list(logs[name]) == list(range(steps + 1)), 'Incomplete resource log')
    totals, cumulative = event_analysis(local, configs)
    shutil.copy2(local / 'event_summary.json', output / 'event_summary.json')
    initial = load_resource(local / 'fixed/run', 0)['capacity']
    initial_source = initial > 0
    expected = torch.from_numpy(initial.copy())
    requested = 0.0
    # Full per-update resource accounting, independent of the ecosystem rollout.
    for update in range(steps + 1):
        if update:
            injection = expected.clamp(max=configs['fixed']['env_options']['inject_rate'])
            requested += float(injection.double().sum())
            expected.sub_(injection).clamp_(min=0)
        for name in NAMES:
            r = logs[name][update]
            require(abs(r['cumulative_requested'] - requested) < 1e-7, 'Unexpected reserve release')
            require(abs(r['remaining_capacity'] - float(expected.double().sum())) < 1e-7, 'Unexpected reserve loss/creation')
            require(abs(r['cumulative_requested'] - r['cumulative_delivered'] - r['cumulative_clipped']) < 0.02, 'Unexplained energy addition discrepancy')
            require(r['occupied_cells'] == r['occupied_inside_source'] + r['occupied_outside_source'], 'Occupancy partition mismatch')
    snapshots, pairs, mask_controls = [], [], []
    previous = {}
    expected = torch.from_numpy(initial.copy())
    last_update = 0
    for update in updates:
        for _ in range(last_update, update):
            expected.sub_(expected.clamp(max=configs['fixed']['env_options']['inject_rate'])).clamp_(min=0)
        states, resources, occupied = {}, {}, {}
        for name in NAMES:
            run_dir = local / name / 'run'
            state = states[name] = load_state(run_dir, update)
            resource = resources[name] = load_resource(run_dir, update)
            require(tuple(state.shape) == (1, 14, 400, 400), 'Unexpected grid/channels')
            require(bool(torch.isfinite(state).all()), 'Nonfinite simulation state')
            require(bool((state[0, 13] == state[0, 13].floor()).all()), 'Noninteger genome IDs')
            occupied[name] = state[0, 13].numpy() >= 0
            source = resource['capacity'] > 0
            options = configs[name]['env_options']
            interval = options['shift_interval']
            offset = update // interval * options['shift_cells'] if interval else 0
            require(np.array_equal(resource['capacity'], torch.roll(expected, offset, dims=1).numpy()), 'Reserve location/depletion mismatch')
            if offset:
                require(not bool((resource['capacity'][:, :offset] > 0).any()), 'Wrapped reserves detected')
            m = logs[name][update]
            require(m['occupied_cells'] == int(occupied[name].sum()), 'State/count phase mismatch')
            require(m['occupied_inside_source'] == int((occupied[name] & source).sum()), 'State/source phase mismatch')
            require(abs(m['total_energy'] - float(state[0, 0].double().sum())) < 1e-7, 'State/energy phase mismatch')
            require(abs(m['total_infra'] - float(state[0, 1].double().sum())) < 1e-7, 'State/infra phase mismatch')
            with Image.open(run_dir / 'screenshots' / f'step_{update:07d}.png') as image:
                require(np.array_equal(np.asarray(image), np.asarray(headless_image(state))), 'State/screenshot mismatch')
            snapshots.append({'condition': name, **m, 'source_displacement_y_cells': offset,
                'changed_occupied_locations_since_previous_capture': int((previous[name] != occupied[name]).sum()) if name in previous else 0,
                'state_sha256': sha256(run_dir / 'frame_states' / f'update_{update:07d}.npz'),
                'resource_sha256': sha256(run_dir / 'frame_states' / f'resource_{update:07d}.npz')})
            previous[name] = occupied[name].copy()
            if update == steps:
                require(torch.equal(state, torch.load(run_dir / f'checkpoint_{steps:07d}/substrate.pt', weights_only=True, map_location='cpu')), 'Final checkpoint mismatch')
        for a, b in itertools.combinations(NAMES, 2):
            both, either = occupied[a] & occupied[b], occupied[a] | occupied[b]
            pairs.append({'completed_update': update, 'condition_a': a, 'condition_b': b,
                'differing_occupied_locations': int((occupied[a] != occupied[b]).sum()),
                'occupied_intersection': int(both.sum()), 'occupied_union': int(either.sum()),
                'occupancy_jaccard': float(both.sum() / either.sum()) if either.any() else 1.0,
                'maximum_abs_state_difference': float((states[a] - states[b]).abs().max())})
        for name in ('slower', 'faster'):
            mask_controls.append({'completed_update': update, 'condition': name,
                **same_mask_counts(occupied['fixed'], occupied[name], resources[name]['capacity'] > 0, initial_source)})
        last_update = update
    write_csv(output / 'snapshot_measurements.csv', snapshots)
    write_csv(output / 'paired_measurements.csv', pairs)
    write_csv(output / 'source_mask_controls.csv', mask_controls)
    summary, budgets = [], {}
    for name in NAMES:
        first, last = logs[name][0], logs[name][steps]
        budget = read_json(local / name / 'run/meta.json')['resource_budget']
        budget['reserve_depletion_minus_requested'] = budget['initial_capacity'] - budget['remaining_capacity'] - budget['cumulative_requested']
        budget['requested_minus_delivered'] = budget['cumulative_requested'] - budget['cumulative_delivered']
        budgets[name] = budget
        interval = configs[name]['env_options']['shift_interval']
        summary.append({'condition': name, 'shift_interval_updates': interval,
            'movement_cells_per_update': 2/interval if interval else 0, 'final_displacement_y_cells': 2*(1000//interval) if interval else 0,
            'initial_occupied_cells': int(first['occupied_cells']), 'final_occupied_cells': int(last['occupied_cells']),
            'final_occupied_inside_current_source': int(last['occupied_inside_source']),
            'final_occupied_outside_current_source': int(last['occupied_outside_source']),
            'local_colonization_gains': totals[name]['colonization_gains'], 'radiation_reseeding_gains': totals[name]['radiation_gains'],
            'actual_energy_delivered': last['cumulative_delivered'], 'requested_energy': last['cumulative_requested'],
            'clipped_energy': last['cumulative_clipped'], 'remaining_reserve': last['remaining_capacity'],
            'final_available_energy': last['total_energy'], 'final_infrastructure': last['total_infra']})
    write_csv(output / 'comparison.csv', summary)
    delivery_range = max(max(logs[n][u]['cumulative_delivered'] for n in NAMES) - min(logs[n][u]['cumulative_delivered'] for n in NAMES) for u in range(steps+1))
    save_json(output / 'resource_accounting.json', {'runs': budgets,
        'maximum_actual_delivery_range_across_conditions_energy_units': delivery_range,
        'maximum_delivery_range_fraction_of_final_delivery': delivery_range / logs['fixed'][steps]['cumulative_delivered'],
        'initial_available_energy': logs['fixed'][0]['total_energy'],
        'initial_infrastructure': logs['fixed'][0]['total_infra'],
        'accounting': 'Actual float64 after-minus-before sum around float32 energy injection/clamp, logged each update. Reserves and requested release independently checked every update. No rescaling.',
        'float32_reserve_depletion_residual': 'Repeated float32 subtraction leaves a nonzero depletion-minus-requested residual; preserved in each budget. Equal in all three treatments.',
        'interpretation': 'Use measured delivery range before attributing differences to movement. Remaining reserves are future supply; available energy has already been released.'})
    pre_motion = {}
    for a, b in itertools.combinations(NAMES, 2):
        boundary = min(c['env_options']['shift_interval'] for n, c in configs.items() if n in (a, b) and c['env_options']['shift_interval'])
        p = [r for r in pairs if r['condition_a'] == a and r['condition_b'] == b]
        early = [r for r in p if r['completed_update'] < boundary]
        pre_motion[f'{a}_vs_{b}'] = {'before_completed_update': boundary,
            'maximum_abs_state_difference': max(r['maximum_abs_state_difference'] for r in early),
            'maximum_differing_occupied_locations': max(r['differing_occupied_locations'] for r in early),
            'first_captured_update_with_occupancy_difference': next((r['completed_update'] for r in p if r['differing_occupied_locations']), None)}
    validation.update({'per_update_resource_budget_records_checked': (steps+1)*3,
        'event_update_records_checked': steps*3, 'reserve_location_checks': len(updates)*3,
        'state_count_source_rgb_checks': len(updates)*3, 'final_checkpoint_matches': 3,
        'same_moving_mask_control_records': len(mask_controls), 'pairwise_spatial_records': len(pairs),
        'saved_completed_updates': updates,
        'resource_wraps': 0, 'pre_movement_checks': pre_motion})
    save_json(output / 'validation.json', validation)
    plot_trajectories(output, logs, mask_controls, pairs, cumulative)
    print(json.dumps({'comparison': summary, 'delivery_range': delivery_range, 'pre_movement': pre_motion}, indent=2), flush=True)


def plot_trajectories(output, logs, controls, pairs, cumulative):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = dict(zip(NAMES, ('#596579', '#168a99', '#d47b1b')))
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), constrained_layout=True)
    for name in NAMES:
        axes[0].plot(list(logs[name]), [r['occupied_cells'] for r in logs[name].values()], label=name.title(), color=colors[name])
    axes[0].set(ylabel='Occupied cells', title='One matched seed: movement frequency changes occupancy patterns')
    axes[0].legend()
    for name in ('slower', 'faster'):
        data = [r for r in controls if r['condition'] == name]
        axes[1].plot([r['completed_update'] for r in data], [r['moving_occupied_in_current_source'] for r in data], color=colors[name], label=f'{name.title()}: current food region')
        axes[1].plot([r['completed_update'] for r in data], [r['fixed_occupied_in_same_moving_source'] for r in data], '--', color=colors[name], label=f'Fixed ecosystem: same {name} mask')
        data = [r for r in pairs if r['condition_a'] == 'fixed' and r['condition_b'] == name]
        axes[2].plot([r['completed_update'] for r in data], [r['differing_occupied_locations'] for r in data], color=colors[name], label=f'Fixed vs {name}')
    axes[1].set(ylabel='Occupied cells within source')
    axes[1].legend(fontsize=9, ncol=2)
    axes[2].set(ylabel='Locations with different occupancy', xlabel='Completed update')
    axes[2].legend()
    for ax in axes:
        ax.grid(alpha=.2)
    fig.savefig(output / 'trajectories.png', dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for name in NAMES:
        updates = list(cumulative[name])
        axes[0].plot(updates, [cumulative[name][u]['local'] for u in updates], label=name.title(), color=colors[name])
        axes[1].plot(updates, [cumulative[name][u]['forced'] for u in updates], label=name.title(), color=colors[name])
    axes[0].set(title='Local colonization gains', ylabel='Cumulative empty-to-occupied transitions', xlabel='Completed update')
    axes[1].set(title='Radiation reseeding gains', xlabel='Completed update')
    for ax in axes:
        ax.legend(); ax.grid(alpha=.2)
    fig.savefig(output / 'mechanisms.png', dpi=150)
    plt.close(fig)
