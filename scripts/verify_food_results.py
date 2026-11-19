"""Check compact Week 13 evidence; optionally inspect local captures and decode videos."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text())


def rows(path):
    with path.open() as stream:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(stream)]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def decode_endpoint(video, fallback, expected):
    from PIL import Image
    probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-count_frames',
        '-show_streams', '-show_format', '-of', 'json', str(video)], text=True))
    stream = probe['streams'][0]
    w, h = expected['size']
    require((stream['width'], stream['height']) == (w, h), 'Wrong video proportions')
    require(int(stream['nb_read_frames']) == expected['decoded_frames'], 'Wrong number of frames')
    require(stream['avg_frame_rate'] == '10/1', 'Wrong playback speed')
    require(abs(float(probe['format']['duration'])-expected['duration_seconds']) < .01, 'Wrong duration')
    proc = subprocess.Popen(['ffmpeg', '-v', 'error', '-xerror', '-i', str(video),
        '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for _ in range(expected['decoded_frames']):
            final = bytearray()
            while len(final) < w*h*3:
                chunk = proc.stdout.read(w*h*3-len(final))
                require(bool(chunk), 'Truncated video')
                final.extend(chunk)
        require(not proc.stdout.read(1), 'Extra video frames')
        require(proc.wait() == 0, proc.stderr.read().decode())
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        proc.stdout.close()
        proc.stderr.close()
    if fallback:
        with Image.open(fallback) as image:
            require(image.size == (w, h) and image.convert('RGB').tobytes() == final,
                    'Fallback is not the exact final decoded frame')


def verify(root, *, raw=False, decode=False):
    reports = {}
    for group, names in (('food_movement_comparison', ('fixed', 'slower', 'faster')),
                         ('food_direction_change', ('continues', 'reverses'))):
        result = root/group/'seed42_20261118'
        with (result/'snapshot_measurements.csv').open() as stream:
            snapshots = {(r['condition'], int(float(r['completed_update']))): r for r in csv.DictReader(stream)}
        require(set(snapshots) == {(n, u) for n in names for u in range(0, 1001, 25)}, 'Incomplete captures')
        validation = read_json(result/'validation.json')
        require(validation['resource_wraps'] == 0, 'Unsupported boundary crossing')
        resource_logs = {}
        for name in names:
            config = read_json(result/'configs'/f'{name}.json')
            require(config == read_json(ROOT/'configs'/group/f'{name}.json'), 'Effective config differs')
            suffix = 'resources' if len(names) == 3 else 'resource_metrics'
            log = rows(result/'measurements'/f'{name}_{suffix}.csv')
            require([r['completed_update'] for r in log] == list(range(1001)), 'Incomplete resource log')
            resource_logs[name] = log
            for r in log:
                require(r['occupied_cells'] == r['occupied_inside_source']+r['occupied_outside_source'], 'Occupancy partition mismatch')
                require(abs(r['cumulative_requested']-r['cumulative_delivered']-r['cumulative_clipped']) < .02, 'Resource accounting discrepancy')
            suffix = 'events' if len(names) == 3 else 'cell_events'
            events = rows(result/'measurements'/f'{name}_{suffix}.csv')
            require([e['completed_update'] for e in events] == list(range(1, 1001)), 'Incomplete event log')
            for e in events:
                u = int(e['completed_update'])
                require(e['occupied_before'] == log[u-1]['occupied_cells'] and e['occupied_after'] == log[u]['occupied_cells'], 'Event phase differs from occupancy log')
                require(e['occupied_after']-e['occupied_before'] == e['net_gains']-e['net_losses'], 'Net event accounting mismatch')
            totals = read_json(result/'event_summary.json')[name]
            for field in ('colonization_gains', 'radiation_gains', 'net_gains', 'net_losses'):
                require(sum(e[field] for e in events) == totals[field], 'Event summary mismatch')
            for u in range(0, 1001, 25):
                row = snapshots[name, u]
                require(all(float(row[k]) == v for k, v in log[u].items()), 'Snapshot/log phase mismatch')
                if raw:
                    import numpy as np
                    folder = result/'local'/name/'run/frame_states'
                    for prefix, field in (('update', 'state_sha256'), ('resource', 'resource_sha256')):
                        require(digest(folder/f'{prefix}_{u:07d}.npz') == row[field], 'Saved capture hash changed')
                    with np.load(folder/f'update_{u:07d}.npz', allow_pickle=False) as archive:
                        state = archive['state']
                        require(int(archive['completed_update']) == u and state.shape == (1, 14, 400, 400), 'Wrong saved state/update')
                        require(np.isfinite(state).all(), 'Nonfinite state')
                        occupied = state[0, 13] >= 0
                    with np.load(folder/f'resource_{u:07d}.npz', allow_pickle=False) as archive:
                        require(int(archive['completed_update']) == u, 'Wrong source update')
                        source = archive['capacity'] > 0
                    require(int(occupied.sum()) == float(row['occupied_cells']), 'Saved occupancy differs')
                    require(int((occupied & source).sum()) == float(row['occupied_inside_source']), 'Source/state alignment differs')
        if len(names) == 2:
            require(all(resource_logs['continues'][u] == resource_logs['reverses'][u] for u in range(625)), 'Common prefix changed')
            pair = rows(result/'paired_measurements.csv')
            require(all(r['maximum_abs_state_difference'] == 0 for r in pair if r['completed_update'] < 625), 'Saved common prefix changed')
            regions = []
            with (result/'source_region_counts.csv').open() as stream:
                regions = list(csv.DictReader(stream))
            for r in regions:
                require(int(r['reverses_minus_continues']) == int(r['reverses_occupied'])-int(r['continues_occupied']), 'Shared-mask count mismatch')
        reports[group] = dict(resource_rows=len(names)*1001, event_rows=len(names)*1000,
                             captured_states=len(snapshots), local_raw_verified=raw)
    ready = root/'presentation_ready'
    sources = root/'food_movement_comparison/seed42_20261118/presentation_assets/original_color/verification.json'
    speed = read_json(sources)['videos']
    direction = read_json(root/'food_direction_change/seed42_20261118/presentation_assets/original_color/verification.json')['videos']
    for relative, fallback, expected in (
        ('01_food_speed/food_speed.mp4', '01_food_speed/food_speed_fallback.png', speed['overview']),
        ('02_direction_change/direction_change.mp4', '02_direction_change/direction_change_fallback.png', direction['closeup']),
        ('backup/speed_closeup.mp4', None, speed['closeup'])):
        require(digest(ready/relative) == expected['video_sha256'], 'Selected video differs from verified export')
        if fallback:
            require(digest(ready/fallback) == expected['fallback_sha256'], 'Selected fallback differs')
        if decode:
            decode_endpoint(ready/relative, ready/fallback if fallback else None, expected)
    require(digest(ready/'03_findings/final_comparison.png') == direction['closeup']['fallback_sha256'], 'Findings must use the reversal endpoint')
    from PIL import Image
    with Image.open(ready/'backup/reversal_occupancy.png') as image:
        require(image.size == (1920, 1080), 'Wrong occupancy backup experiment/shape')
    reports['presentation'] = dict(selected_video_hashes_verified=3, exact_reversal_findings=True, decoded=decode)
    return reports


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results-root', type=Path, default=ROOT/'results/week13')
    p.add_argument('--raw', action='store_true', help='Require ignored local saved-state captures')
    p.add_argument('--decode', action='store_true', help='Require ffmpeg/ffprobe and fully decode selected videos')
    args = p.parse_args()
    print(json.dumps(verify(args.results_root, raw=args.raw, decode=args.decode), indent=2))


if __name__ == '__main__':
    main()
