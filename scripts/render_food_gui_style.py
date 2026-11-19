"""Export GUI-style food comparisons from existing captures; no simulation entry point."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import taichi as ti
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'external/coralai'),
                str(ROOT/'external/coralai/coralai/dependencies/PyTorch-NEAT')]
from coralai.visualization import Visualization
from scripts.compare_trail_reference import gui_values, rgb_image, load_state, load_resource, require, sha256

WORLD = (0, 0, 400, 400)
CLOSEUP = (300, 100, 400, 200)
PAGE = (17, 22, 31)
EMPTY = (20, 29, 40)
CYAN = (69, 211, 230)
ORANGE = (255, 172, 74)
WHITE = (238, 243, 248)
FPS = 10

def font(size, bold=False):
    path = '/usr/share/fonts/dejavu-sans-fonts/DejaVuSans' + ('-Bold' if bold else '') + '.ttf'
    if not Path(path).exists():
        path = '/usr/share/fonts/truetype/dejavu/DejaVuSans' + ('-Bold' if bold else '') + '.ttf'
    return ImageFont.truetype(path, size) if Path(path).exists() else ImageFont.load_default(size=size)

def outline_pixels(source, roi, scale, width):
    """Draw actual cell-edge boundaries before cropping; no false crop border.

    Arrays are in image (y,x) order. Vertical edges lie on x grid lines;
    horizontal edges lie on y grid lines. All enlargement is integer nearest.
    """
    x0, y0, x1, y1 = roi
    h, w = source.shape
    vertical = np.zeros((h, w + 1), dtype=bool)
    horizontal = np.zeros((h + 1, w), dtype=bool)
    vertical[:, 0], vertical[:, -1] = source[:, 0], source[:, -1]
    vertical[:, 1:-1] = source[:, :-1] != source[:, 1:]
    horizontal[0], horizontal[-1] = source[0], source[-1]
    horizontal[1:-1] = source[:-1] != source[1:]
    out = np.zeros(((y1-y0)*scale, (x1-x0)*scale), dtype=bool)
    for edges, is_vertical in ((vertical[y0:y1, x0:x1+1], True),
                               (horizontal[y0:y1+1, x0:x1], False)):
        rows, cols = np.nonzero(edges)
        for along in range(scale):
            for across in range(width):
                yy = rows*scale + (along if is_vertical else across)
                xx = cols*scale + (across if is_vertical else along)
                inside = (yy < out.shape[0]) & (xx < out.shape[1])
                out[yy[inside], xx[inside]] = True
    return out

def centered(draw, text, center, y, size, bold=False, color=WHITE):
    face = font(size, bold)
    draw.text((round(center-draw.textlength(text, font=face)/2), y), text, font=face, fill=color)

def locator(initial_source):
    image = np.empty((400, 400, 3), dtype=np.uint8)
    image[:] = EMPTY
    image[initial_source] = (103, 121, 140)
    image = Image.fromarray(image).resize((200, 200), Image.Resampling.NEAREST)
    d = ImageDraw.Draw(image)
    x0, y0, x1, y1 = CLOSEUP
    d.rectangle((x0//2, y0//2, x1//2-1, y1//2-1), outline=WHITE, width=3)
    d.rectangle((0, 0, 199, 199), outline=(103, 121, 140), width=1)
    return image

def geometry_checks():
    source = np.zeros((400, 400), dtype=bool)
    source[120:130, 340:350] = True
    outlines = outline_pixels(source, CLOSEUP, 8, 2)
    require(outlines[20*8, 45*8] and outlines[25*8, 40*8], 'x/y orientation fixture failed')
    require(not outlines[10, 10], 'Misplaced outline')
    source[:] = True
    require(not outline_pixels(source, CLOSEUP, 8, 2)[:, 0].any(), 'Crop created a false left source boundary')
    return True

def encode(source_frames, timeline, output, sequence):
    sequence.mkdir()
    for i, update in enumerate(timeline):
        (sequence / f'frame_{i:04d}.png').symlink_to(source_frames / f'update_{update:04d}.png')
    command = ['ffmpeg', '-v', 'error', '-framerate', str(FPS), '-i', str(sequence/'frame_%04d.png'),
               '-frames:v', str(len(timeline)), '-an', '-c:v', 'libx264', '-preset', 'medium',
               '-crf', '16', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(output)]
    subprocess.run(command, check=True, capture_output=True)
    return command

def cue(update):
    if update<500:return 'Both sources move down together'
    if update<625:return 'Turn enabled; the next source shift will be upward'
    if update==625:return 'First upward source shift'
    return 'Right-hand source is returning upward'


SPECS = {
    'speeds': dict(directory='food_movement_comparison', names=('fixed', 'slower', 'faster'),
                   labels=('Fixed', 'Slower', 'Faster'), size=(2560, 1440), xs=(40, 880, 1720), py=330),
    'direction': dict(directory='food_direction_change', names=('continues', 'reverses'),
                      labels=('Keeps moving down', 'Turns back upward'), size=(1920, 1080), xs=(100, 1020), py=180),
}
CHANNELS = (0, 1, 12)
UPDATES = list(range(0, 1001, 25))


def json_write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def viewer_for_saved_states():
    # Only the display kernel is used. No window, experiment, controller, or physics is constructed.
    ti.init(arch=ti.cpu, cpu_max_num_threads=1, offline_cache=False)
    vis = Visualization.__new__(Visualization)
    vis.w = vis.h = 400
    vis.scale = 1
    vis.image = ti.Vector.field(3, dtype=ti.f32, shape=(400, 400))
    vis.chinds = torch.tensor(CHANNELS)
    vis.norm_mode = vis.view_mode = 0
    return vis


def checked_rgb(state, vis):
    require(tuple(state.shape) == (1, 14, 400, 400), 'Missing required saved state channels/grid')
    require(np.isfinite(state[0, list(CHANNELS)].numpy()).all(), 'Nonfinite display channels')
    vis.substrate = SimpleNamespace(mem=state, torch_device=torch.device('cpu'))
    scales, mask = vis._compute_render_params()
    require(mask.tolist() == [1, 1, 1], 'Viewer RGB mask changed')
    vis.write_to_renderer(state, scales, vis.chinds, mask, vis.norm_mode, 400)
    actual = vis.image.to_numpy().transpose(1, 0, 2)
    exported = gui_values(state)
    error = float(np.max(np.abs(actual - exported)))
    require(error <= 2e-7, 'GUI-compatible exporter differs from interactive display kernel')
    return np.asarray(rgb_image(exported)), error, scales.tolist()


def state_panel(rgb, source, kind):
    roi, scale = (WORLD, 2) if kind == 'overview' else (CLOSEUP, 8)
    x0, y0, x1, y1 = roi
    expected = np.repeat(np.repeat(rgb[y0:y1, x0:x1], scale, axis=0), scale, axis=1)
    pixels = expected.copy()
    outline = outline_pixels(source, roi, scale, 2)
    pixels[outline] = ORANGE
    require(pixels.shape == (800, 800, 3), 'Non-square simulation panel')
    require(np.array_equal(pixels[~outline], expected[~outline]), 'Underlying GUI channels changed')
    # Independently recover the cell-edge locations from the full saved mask.
    # Checking before clipping keeps the close-up from inventing a boundary at its left edge.
    padded = np.pad(source.astype(np.int8), 1)
    vy, vx = np.nonzero(np.diff(padded, axis=1)[1:-1])
    hy, hx = np.nonzero(np.diff(padded, axis=0)[:, 1:-1])
    reference = np.zeros((800, 800), dtype=bool)
    for yy, xx, vertical in ((vy, vx, True), (hy, hx, False)):
        yy, xx = (yy-y0)*scale, (xx-x0)*scale
        for along in range(scale):
            for across in range(2):
                rows = yy + (along if vertical else across)
                cols = xx + (across if vertical else along)
                keep = (rows >= 0) & (cols >= 0) & (rows < 800) & (cols < 800)
                reference[rows[keep], cols[keep]] = True
    require(np.array_equal(outline, reference), 'Source outline is not aligned to saved grid edges')
    return Image.fromarray(pixels), outline


def legend(draw, center, y, size, view):
    items = [('Energy', (255, 80, 80)), ('Infrastructure', (80, 235, 100)),
             ('Rotation', (110, 155, 255)), ('Source outline', ORANGE)]
    if view == 'occupancy':
        items = [('Occupied cells', CYAN), ('Source outline', ORANGE)]
    face = font(size)
    widths = [30 + draw.textlength(label, font=face) for label, _ in items]
    gap = 44
    x = round(center - (sum(widths) + gap*(len(items)-1))/2)
    for (label, color), width in zip(items, widths):
        draw.rectangle((x, y+8, x+18, y+26), fill=None if color == ORANGE else color,
                       outline=color, width=2)
        draw.text((x+30, y), label, font=face, fill=WHITE)
        x += round(width) + gap


def render(spec, key, update, rgbs, sources, kind, inset, view='gui'):
    canvas = Image.new('RGB', spec['size'], PAGE)
    d = ImageDraw.Draw(canvas)
    is_speed = key == 'speeds'
    title = 'GUI-style' if view == 'gui' else 'Occupancy'
    note = ('Auto-scaled channels; colored pixels may be empty. Brightness is not population.'
            if view == 'gui' else 'Cyan = occupied cells. Orange outline = current source region.')
    centered(d, f'{title}  |  Completed update {update:04d}', spec['size'][0]//2,
             78 if is_speed else 15, 44 if is_speed else 32)
    if not is_speed:
        centered(d, cue(update), 960, 127, 27, color=ORANGE if update >= 500 else WHITE)
    outlines = {}
    for x, name, label in zip(spec['xs'], spec['names'], spec['labels']):
        centered(d, label, x+400, 204 if is_speed else 65, 68 if is_speed else 48, True)
        image, outlines[name] = state_panel(rgbs[name], sources[name], kind)
        canvas.paste(image, (x, spec['py']))
    if is_speed:
        center = 1120 if kind == 'closeup' else 1280
        legend(d, center, 1225, 34, view)
        centered(d, note,
                 center, 1300, 28)
        if kind == 'closeup':
            canvas.paste(inset, (2280, 1190))
            centered(d, 'Close-up location', 2380, 1392, 20)
    else:
        legend(d, 960, 985, 25, view)
        centered(d, note,
                 960, 1033, 23)
        if kind == 'closeup':
            canvas.paste(inset.resize((100, 100), Image.Resampling.NEAREST), (910, 860))
    return canvas, outlines


def timeline_for(key):
    result = []
    for update in UPDATES:
        count = 30 if update == 1000 else 12 if update == 0 else 3
        if key == 'direction' and update in (500, 625):
            count = 12 if update == 500 else 9
        result.extend([update]*count)
    return result


def verify_video(path, frames, timeline, fallback, spec, key, kind, work):
    probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-count_frames',
        '-show_streams', '-show_format', '-of', 'json', str(path)], text=True))
    stream = probe['streams'][0]
    width, height = spec['size']
    require((stream['width'], stream['height']) == spec['size'], 'Video proportions changed')
    require(int(stream['nb_read_frames']) == len(timeline), 'Video frame count mismatch')
    require(stream['avg_frame_rate'] == '10/1', 'Playback FPS mismatch')
    duration = float(probe['format']['duration'])
    require(abs(duration-len(timeline)/FPS) < .01, 'Playback duration mismatch')
    decoder = subprocess.Popen(['ffmpeg', '-v', 'error', '-i', str(path), '-f', 'rawvideo',
        '-pix_fmt', 'rgb24', '-'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    max_panel = max_outline = max_label = 0.0
    min_label_iou = 1.0
    previous, original = None, None
    checked = set()
    representative = (0, 500, 625, 1000) if key == 'direction' else (0, 500, 1000)
    byte_count = width*height*3
    try:
        for i, update in enumerate(timeline):
            raw = bytearray()
            while len(raw) < byte_count:
                piece = decoder.stdout.read(byte_count-len(raw))
                require(bool(piece), 'Truncated video')
                raw.extend(piece)
            decoded = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)
            # Check every decoded frame, including all opening, turn, and endpoint holds.
            if previous != update:
                with Image.open(frames/f'update_{update:04d}.png') as image:
                    original = np.asarray(image.convert('RGB'))
            label_end = 290 if key == 'speeds' else 170
            err = np.abs(decoded[:label_end].astype(np.int16)-original[:label_end].astype(np.int16))
            label_error = float(err.mean())
            require(label_error < 5, 'Unclear completed-update labels or direction cue')
            max_label = max(max_label, label_error)
            for x in spec['xs']:
                y0, y1 = (195, 290) if key == 'speeds' else (60, 125)
                expected_label = original[y0:y1, x:x+800]
                actual_label = decoded[y0:y1, x:x+800]
                def foreground(rgb):
                    return (rgb.min(2) > 150) & (np.ptp(rgb.astype(np.int16), axis=2) < 25)
                wanted, got = foreground(expected_label), foreground(actual_label)
                require(wanted.sum() > 1000, 'Condition label is missing')
                iou = float((wanted & got).sum() / (wanted | got).sum())
                require(iou > .97, 'Condition lettering is distorted')
                min_label_iou = min(min_label_iou, iou)
                area = np.s_[spec['py']:spec['py']+800, x:x+800]
                expected, actual = original[area], decoded[area]
                error = np.abs(actual.astype(np.int16)-expected.astype(np.int16))
                panel_error = float(error.mean())
                require(panel_error < 12, 'Excessive GUI color error in encoded video')
                max_panel = max(max_panel, panel_error)
                annotation = np.all(expected == ORANGE, axis=2)
                if annotation.any():
                    outline_error = float(error[annotation].mean())
                    require(outline_error < 50, 'Source outline lost during encoding')
                    max_outline = max(max_outline, outline_error)
            if previous != update and update in representative:
                Image.fromarray(decoded).save(work/f'{kind}_decoded_{update:04d}.png')
            checked.add(update)
            previous = update
        require(not decoder.stdout.read(1), 'Unexpected extra video frames')
        require(decoder.wait() == 0, decoder.stderr.read().decode())
    finally:
        if decoder.poll() is None:
            decoder.kill()
            decoder.wait()
        decoder.stdout.close()
        decoder.stderr.close()
    Image.fromarray(decoded).save(fallback)
    with Image.open(fallback) as image:
        require(np.array_equal(np.asarray(image), decoded), 'Endpoint fallback mismatch')
    require(sorted(checked) == UPDATES, 'Omitted saved capture')
    return dict(decoded_frames=len(timeline), duration_seconds=duration, fps=FPS,
        size=spec['size'], all_frames_checked=True, all_41_completed_updates_verified=True,
        max_panel_mean_rgb_error=max_panel, max_outline_mean_rgb_error=max_outline,
        max_label_mean_rgb_error=max_label, minimum_condition_label_foreground_iou=min_label_iou,
        fallback_equals_final_decoded_frame=True,
        video_sha256=sha256(path), fallback_sha256=sha256(fallback))


def captions(key, view):
    colors = ('Red = energy; green = infrastructure; blue = rotation. Each channel uses its current '
              'full-world maximum + 1e-8 before cropping. Colored pixels may be unoccupied; '
              'brightness is not population. GUI-style, not an exact reconstruction of unknown Week 11 display settings.'
              if view == 'gui' else 'Cyan = occupied cells (genome >= 0).')
    phase = ('Fixed / Slower / Faster: shifts are 0 / +2 every 250 / +2 every 125 updates.' if key == 'speeds' else
             'Keeps moving down / Turns back upward. Both move down through 500; first upward shift at 625.')
    return (f'{colors}\n{phase}\nOrange outlines show remaining-capacity > 0, not reserve quantity or available energy.\n'
            'Overview: 400x400 at 2x. Close-up: x=[300,400), y=[100,200) at 8x, shared crop, no smoothing. '
            'Right = +x; down = +y. All 41 saved completed updates retained, with opening/endpoint holds.\n'
            'PNGs match decoded video endpoints. One matched seed does not establish learning, locomotion, or a best strategy. '
            'Controller corrections affect actual behavior; rendering changes presentation only.\n')


def export(key, vis, preview, *, results_dir=None, output_dir=None, view='gui'):
    spec = SPECS[key]
    result = results_dir.resolve() if results_dir else ROOT/'results/week13'/spec['directory']/'seed42_20261118'
    assets = output_dir.resolve() if output_dir else result/'presentation_assets'/('original_color' if view == 'gui' else 'occupancy')
    require(preview or not assets.exists(), f'Refusing to overwrite {assets}')
    work = Path(tempfile.mkdtemp(prefix='original-color-', dir=result/'local'))
    for kind in ('overview', 'closeup'):
        (work/kind).mkdir()
    with (result/'snapshot_measurements.csv').open() as stream:
        evidence = {(r['condition'], int(float(r['completed_update']))): r for r in csv.DictReader(stream)}
    inset = locator(load_resource(result/'local'/spec['names'][0]/'run', 0)['capacity'].T > 0)
    checks, inputs = [], {}
    updates = (0, 625, 1000) if preview else UPDATES
    for update in updates:
        rgbs, sources = {}, {}
        for name in spec['names']:
            run = result/'local'/name/'run'
            for prefix, field in (('update', 'state_sha256'), ('resource', 'resource_sha256')):
                path = run/'frame_states'/f'{prefix}_{update:07d}.npz'
                require(path.is_file(), f'Required saved capture missing: {path}')
                actual_hash = sha256(path)
                require(actual_hash == evidence[name, update][field], f'Completed experiment input changed: {path}')
                inputs[str(path.relative_to(result))] = actual_hash
            state, resource = load_state(run, update), load_resource(run, update)
            if view == 'gui':
                rgbs[name], error, maxima = checked_rgb(state, vis)
            else:
                require(tuple(state.shape) == (1, 14, 400, 400), 'Missing saved occupancy state')
                occupied = state[0, 13].numpy().T >= 0
                rgbs[name] = np.empty((400, 400, 3), dtype=np.uint8)
                rgbs[name][:] = EMPTY
                rgbs[name][occupied] = CYAN
                error, maxima = 0.0, []
            require(int((state[0, 13] >= 0).sum()) == int(float(evidence[name, update]['occupied_cells'])),
                    'State/measurement occupancy mismatch')
            sources[name] = resource['capacity'].T > 0
            require(sources[name].shape == (400, 400), 'Saved source mask shape mismatch')
            checks.append(dict(condition=name, completed_update=update, maxima_rgb=maxima,
                               maximum_viewer_float_error=error,
                               source_displacement_y_cells=evidence[name, update]['source_displacement_y_cells']))
        for kind in ('overview', 'closeup'):
            frame, outlines = render(spec, key, update, rgbs, sources, kind, inset, view)
            frame.save(work/kind/f'update_{update:04d}.png', compress_level=3)
            np.savez_compressed(work/kind/f'outlines_{update:04d}.npz', **outlines)
        print(f'{key}: checked and rendered saved update {update}', flush=True)
    if preview:
        print(f'PREVIEW {key}: {work}', flush=True)
        return
    assets.mkdir(parents=True)
    timeline = timeline_for(key)
    videos, commands = {}, {}
    for kind in ('overview', 'closeup'):
        print(f'{key}: encoding {kind}', flush=True)
        commands[kind] = encode(work/kind, timeline, assets/f'{kind}.mp4', work/f'{kind}_sequence')
        print(f'{key}: decoding and verifying {kind}', flush=True)
        videos[kind] = verify_video(assets/f'{kind}.mp4', work/kind, timeline,
                                   assets/f'{kind}_final.png', spec, key, kind, work)
    (assets/'captions.txt').write_text(captions(key, view))
    code = ('scripts/render_food_gui_style.py', 'scripts/compare_trail_reference.py',
            'external/coralai/coralai/visualization.py', 'external/coralai/experiments/coral_dev/experiment.py')
    json_write(assets/'verification.json', dict(videos=videos, encoder_commands=commands,
        renderer_source_sha256={p: sha256(ROOT/p) for p in code},
        saved_inputs_sha256=inputs, capture_checks=checks,
        saved_state_viewer_kernel_checks=checks if view == 'gui' else [], view=view,
        source_outline_alignment='Independent full-mask cell-edge check for every rendered panel; 2 display pixels.',
        normalized_before_crop=True, rgb_channels=dict(energy=0, infrastructure=1, rotation=12),
        normalization=('raw / (full-world channel max + 1e-8); default RGB mask [1,1,1]'
                       if view == 'gui' else 'genome >= 0; fixed cyan occupancy color'),
        timeline_completed_updates=timeline, closeup_half_open_xy=CLOSEUP,
        orientation='right=+x, down=+y, state[x,y] transposed to image[y,x]',
        square_panel_pixels=800, rendered_source_frames=str(work)))
    print(f'EXPORTED {assets}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', choices=('all', *SPECS), default='all')
    parser.add_argument('--results-dir', type=Path, help='Fresh or saved result directory for the selected experiment')
    parser.add_argument('--output-dir', type=Path, help='Fresh export directory; existing exports are never overwritten')
    parser.add_argument('--view', choices=('gui', 'occupancy'), default='gui')
    parser.add_argument('--preview-only', action='store_true')
    args = parser.parse_args()
    require(geometry_checks(), 'Existing outline geometry checks failed')
    if args.experiment == 'all' and (args.results_dir or args.output_dir):
        parser.error('--results-dir/--output-dir require one --experiment')
    vis = viewer_for_saved_states() if args.view == 'gui' else None
    for key in SPECS if args.experiment == 'all' else (args.experiment,):
        export(key, vis, args.preview_only, results_dir=args.results_dir,
               output_dir=args.output_dir, view=args.view)


if __name__ == '__main__':
    main()
