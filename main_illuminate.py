"""ASAL illumination with identified candidates and replayable JAX Coralai archives."""
import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

import argparse
from functools import partial
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import resource
import subprocess
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm.auto import tqdm

import substrates
import foundation_models
from rollout import rollout_simulation
import asal_metrics
import util

ROOT = Path(__file__).resolve().parent
SCHEMA_VERSION = 2
CLIP_REVISION = '3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268'
WORLD_SOURCES = ('substrates/coralai_new.py', 'substrates/__init__.py', 'rollout.py',
                 'foundation_models/clip.py', 'foundation_models/pixels.py')

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--seed', type=int, default=0, help='optimizer randomness only')
parser.add_argument('--eval_seed', type=int, default=42, help='one fixed world key for every candidate')
parser.add_argument('--save_dir', type=Path, required=True, help='new output directory; existing paths are refused')
parser.add_argument('--substrate', default='boids')
parser.add_argument('--substrate_config', type=Path, help='JSON constructor overrides for JAX Coralai')
parser.add_argument('--rollout_steps', type=int)
parser.add_argument('--foundation_model', choices=('clip', 'pixels', 'dino'), default='clip')
parser.add_argument('--clip_model_path', type=Path, help='local snapshot; otherwise use the pinned HF revision')
parser.add_argument('--clip_revision', default=CLIP_REVISION)
parser.add_argument('--offline', action='store_true')
parser.add_argument('--k_nbrs', type=int, default=2)
parser.add_argument('--n_child', type=int, default=32)
parser.add_argument('--pop_size', type=int, default=256)
parser.add_argument('--n_iters', type=int, default=1000)
parser.add_argument('--sigma', type=float, default=0.1)
parser.add_argument('--record_selection', action='store_true', help='retain pool embeddings for independent selection verification; bounded pilots only')


def parse_args(*args, **kwargs):
    result = parser.parse_args(*args, **kwargs)
    if not (result.pop_size >= 2 and result.n_child >= 1 and result.n_iters >= 0
            and 1 <= result.k_nbrs < result.pop_size and np.isfinite(result.sigma) and result.sigma > 0):
        parser.error('Require pop_size >= 2, n_child >= 1, n_iters >= 0, 1 <= k_nbrs < pop_size, and finite sigma > 0')
    if result.rollout_steps is not None and result.rollout_steps < 1:
        parser.error('rollout_steps must be positive')
    return result


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def array_hash(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def world_fingerprint():
    return {name: sha256(ROOT / name) for name in WORLD_SOURCES}


def create_model(name, model_path=None, revision=CLIP_REVISION, offline=False):
    if name == 'dino':
        return foundation_models.create_foundation_model(name), {'name': name, 'identity': 'legacy factory DINO',
            'replay_supported': False}
    if name == 'pixels':
        return foundation_models.Pixels(), {'name': 'pixels', 'identity': '8x8 block-mean RGB, L2 normalized',
            'preprocessing': {'input': 'float32 HWC [0,1]', 'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}}
    from huggingface_hub import snapshot_download
    identity = 'openai/clip-vit-base-patch32'
    path = Path(model_path) if model_path else Path(snapshot_download(identity, revision=revision,
        local_files_only=offline, allow_patterns=['*.json', '*.txt', 'flax_model.msgpack']))
    model = foundation_models.CLIP(model_path=str(path))
    return model, {'name': 'clip', 'identity': identity, 'revision': revision,
        'checkpoint_sha256': sha256(path / 'flax_model.msgpack'), 'original_model_path': str(path.resolve()),
        'processor_sha256': {p.name: sha256(p) for p in (path / 'config.json', path / 'preprocessor_config.json')},
        'preprocessing': {'input': 'float32 HWC [0,1]', 'render_size': 224, 'resize': 'JAX bilinear',
            'mean': np.asarray(model.img_mean).tolist(), 'std': np.asarray(model.img_std).tolist(),
            'embedding': 'Flax CLIP image features, L2 normalized; no uint8 quantization before embedding'}}


def evaluator(substrate, model, steps):
    rollout = partial(rollout_simulation, s0=None, substrate=substrate, fm=model,
                      rollout_steps=steps, time_sampling='final', img_size=224, return_state=False)
    def evaluate(key, params, candidate_id):
        result = rollout(key, params)
        return dict(params=params, rgb=result['rgb'], z=result['z'],
                    candidate_id=candidate_id, eval_key=key)
    return jax.jit(evaluate)


def select_archive(pool, k_nbrs, n_child):
    """Original greedy similarity selection; all candidate fields travel together."""
    size = pool['z'].shape[0]
    distances = -pool['z'] @ pool['z'].T
    distances = distances.at[jnp.arange(size), jnp.arange(size)].set(jnp.inf)
    def remove(matrix, _):
        index = matrix.sort(axis=-1)[:, :k_nbrs].mean(axis=-1).argmin()
        return matrix.at[:, index].set(jnp.inf).at[index, :].set(jnp.inf), index
    _, killed = jax.lax.scan(remove, distances, None, length=n_child)
    kept = jnp.setdiff1d(jnp.arange(size), killed, assume_unique=True, size=size-n_child)
    return jax.tree.map(lambda values: values[kept], pool), kept, killed


def effective_hashes(params, substrate):
    result = []
    for vector in params:
        tree = substrate.param_reshaper.reshape_single(jnp.asarray(vector))
        if hasattr(substrate.substrate, '_base'):
            tree = jax.tree.map(lambda base, delta: base+delta, substrate._base, tree)
        result.append(array_hash(np.concatenate([np.asarray(x).ravel() for x in jax.tree.leaves(tree)])))
    return result


def save_archive(output, pop, substrate, metadata, history, traces):
    host = jax.device_get(pop)
    archive = dict(host)
    archive['rgb_float_sha256'] = np.asarray([array_hash(x) for x in host['rgb']], dtype='S64')
    archive['rgb'] = np.rint(np.clip(host['rgb'], 0, 1)*255).astype(np.uint8)
    archive['effective_parameter_sha256'] = np.asarray(effective_hashes(host['params'], substrate), dtype='S64')
    if hasattr(substrate.substrate, '_base'):
        archive.update({'base_'+key: np.asarray(value) for key, value in substrate._base.items()})
    with (output / 'archive.npz.tmp').open('wb') as stream:
        np.savez_compressed(stream, **archive)
    (output / 'archive.npz.tmp').replace(output / 'archive.npz')
    np.savez_compressed(output / 'history.npz', loss=np.asarray(history, dtype=np.float32))
    if traces:
        np.savez_compressed(output / 'selection.npz', **{k: np.stack([t[k] for t in traces]) for k in traces[0]})
    metadata['archive_sha256'] = sha256(output / 'archive.npz')
    util.save_json(output, 'meta', metadata)
    return archive


def load_archive(output, verify_world=True):
    output = Path(output)
    if not (output / 'meta.json').is_file():
        raise ValueError('Legacy/unidentified archive: meta.json with versioned settings, candidate IDs and evaluation keys is required; replay settings cannot be guessed.')
    metadata = util.load_json(output, 'meta')
    if metadata.get('schema_version') != SCHEMA_VERSION:
        raise ValueError('Legacy archive schema is not a verified replay of the corrected implementation.')
    if verify_world and metadata['world_source_sha256'] != world_fingerprint():
        raise ValueError('Simulator/renderer source fingerprint differs from the saved world; use the recorded implementation.')
    if sha256(output / 'archive.npz') != metadata['archive_sha256']:
        raise ValueError('Archive checksum does not match its metadata')
    with np.load(output / 'archive.npz', allow_pickle=False) as source:
        archive = dict(source)
    return metadata, archive


def export_gallery(output, metadata=None, archive=None):
    """Map every archive row to a labeled tile, across as many pages as needed."""
    output = Path(output)
    if metadata is None: metadata, archive = load_archive(output)
    count = len(archive['candidate_id'])
    p_hashes = [s.decode() for s in archive['effective_parameter_sha256']]
    i_hashes = [array_hash(image) for image in archive['rgb']]
    counts = {'archive_entries': count, 'unique_effective_parameters': len(set(p_hashes)),
              'unique_rendered_images': len(set(i_hashes)), 'duplicate_parameter_entries': count-len(set(p_hashes)),
              'duplicate_image_entries': count-len(set(i_hashes))}
    pages, entries = [], []
    page_count = (count+63)//64
    font = ImageFont.load_default(size=13)
    name = metadata.get('substrate', {}).get('name', 'coralai')
    title = 'JAX Coralai' if name == 'coralai' else f'ASAL {name}'
    for page in range(page_count):
        start, end = page*64, min((page+1)*64, count)
        cols = max(2, min(8, int(np.ceil(np.sqrt(end-start)))))
        sheet = Image.new('RGB', (cols*224, 70+int(np.ceil((end-start)/cols))*250), '#11161f')
        draw = ImageDraw.Draw(sheet)
        draw.text((10, 10), f'{title} atlas | {metadata["foundation_model"]["name"]} | page {page+1}/{page_count}', font=font, fill='white')
        draw.text((10, 33), f'{count} entries | {counts["unique_rendered_images"]} unique images | fixed evaluation key', font=font, fill='#c7d3dc')
        filename = f'gallery_{page+1:03d}.png'
        for index in range(start, end):
            x, y = (index-start)%cols*224, 70+(index-start)//cols*250
            sheet.paste(Image.fromarray(archive['rgb'][index]), (x, y))
            cid = int(archive['candidate_id'][index])
            draw.text((x+4, y+227), f'row {index:03d} | candidate {cid:06d}', font=font, fill='white')
            entries.append({'archive_index': index, 'candidate_id': cid, 'eval_key': archive['eval_key'][index].tolist(),
                'params': f'archive.npz:params[{index}]', 'effective_parameter_sha256': p_hashes[index],
                'image_sha256': i_hashes[index], 'page': filename, 'image_box': [x, y, x+224, y+224]})
        sheet.save(output / filename)
        pages.append(filename)
    mapping = {'schema_version': SCHEMA_VERSION, 'counts': counts, 'pages': pages, 'entries': entries,
        'replay': 'python illumination_results/res.py --run_dir RUN_DIR --candidate_ids ID [ID ...]'}
    util.save_json(output, 'gallery', mapping)
    return mapping


def replay_archive(output, candidate_ids, model_path=None, offline=False):
    from substrates.coralai_new import IMPLEMENTATION_VERSION
    output = Path(output)
    metadata, archive = load_archive(output)
    description = metadata['substrate']
    if description['name'] != 'coralai' or description['implementation_version'] != IMPLEMENTATION_VERSION:
        raise ValueError('Verified replay currently requires a corrected, versioned JAX Coralai archive.')
    sub = substrates.FlattenSubstrateParameters(substrates.create_substrate('coralai', **description['settings']))
    sub.substrate._base = {key: jnp.asarray(archive['base_'+key]) for key in ('w1','b1','w2','b2')}
    fm_record = metadata['foundation_model']
    if fm_record['name'] not in ('clip', 'pixels'):
        raise ValueError('Verified replay supports the recorded CLIP or pixels model; legacy DINO loading is not revision-pinned.')
    if model_path is None and Path(fm_record.get('original_model_path', '')).is_dir() and fm_record.get('original_model_path'):
        model_path = fm_record['original_model_path']
    model, model_record = create_model(fm_record['name'], model_path, fm_record.get('revision', CLIP_REVISION), offline)
    for key in ('checkpoint_sha256', 'processor_sha256', 'preprocessing'):
        if model_record.get(key) != fm_record.get(key): raise ValueError(f'Replay model mismatch: {key}')
    evaluate = evaluator(sub, model, metadata['arguments']['rollout_steps'])
    lookup = {int(cid): i for i, cid in enumerate(archive['candidate_id'])}
    results = []
    for cid in candidate_ids:
        if cid not in lookup: raise ValueError(f'Candidate {cid} is not in the saved archive; available IDs: {sorted(lookup)}')
        index = lookup[cid]
        fresh = jax.device_get(evaluate(jnp.asarray(archive['eval_key'][index]), jnp.asarray(archive['params'][index]), jnp.int32(cid)))
        image = np.rint(np.clip(fresh['rgb'], 0, 1)*255).astype(np.uint8)
        results.append({'candidate_id': cid, 'archive_index': index,
            'eval_key': archive['eval_key'][index].tolist(),
            'float_image_sha256_matches': array_hash(fresh['rgb']) == archive['rgb_float_sha256'][index].decode(),
            'uint8_image_max_abs_difference': int(np.abs(image.astype(int)-archive['rgb'][index].astype(int)).max()),
            'embedding_exact_match': bool(np.array_equal(fresh['z'], archive['z'][index])),
            'embedding_max_abs_difference': float(np.abs(fresh['z']-archive['z'][index]).max())})
    util.save_json(output, 'replay', {'entries': results, 'jax_backend': jax.default_backend(),
        'note': 'Measured reconstruction in this environment; bitwise identity across other backends/versions is not promised.'})
    return results


def main(args):
    started = time.perf_counter()
    output = args.save_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    options = json.loads(args.substrate_config.read_text()) if args.substrate_config else {}
    raw_substrate = substrates.create_substrate(args.substrate, **options)
    substrate = substrates.FlattenSubstrateParameters(raw_substrate)
    args.rollout_steps = args.rollout_steps if args.rollout_steps is not None else substrate.rollout_steps
    raw_substrate.rollout_steps = args.rollout_steps
    model, fm_record = create_model(args.foundation_model, args.clip_model_path, args.clip_revision, args.offline)
    from substrates.coralai_new import IMPLEMENTATION_VERSION
    sources = (*WORLD_SOURCES, 'main_illuminate.py', 'illumination_results/res.py', 'asal_metrics.py')
    metadata = {'schema_version': SCHEMA_VERSION, 'status': 'running',
        'arguments': {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
        'evaluation': {'protocol': 'fixed world; independent optimizer RNG', 'optimizer_seed': args.seed,
            'simulation_seed': args.eval_seed, 'simulation_key': np.asarray(jax.random.PRNGKey(args.eval_seed)).tolist()},
        'substrate': {'name': args.substrate, 'implementation_version': IMPLEMENTATION_VERSION if args.substrate=='coralai' else 'factory-defaults',
            'settings': raw_substrate.settings() if args.substrate=='coralai' else None,
            'parameter_count': substrate.n_params,
            'parameter_tree': {k: list(v.shape) for k,v in raw_substrate._base.items()} if args.substrate=='coralai' else None,
            'initialization': 'Coralai: zero energy/com, patch-seeded infrastructure/reserves, patch-masked random genome IDs, eight random perimeter headings; t=0.',
            'controller': 'Coralai: per-genome 54-input MLP, tanh hidden layer, 10 outputs; offsets plus saved base tensors; center then heading-relative perimeter inputs.',
            'renderer': 'Coralai: genome palette, infrastructure brightness, energy glow, reserve background; HWC RGB bilinear resize to 224.'},
        'foundation_model': fm_record, 'world_source_sha256': world_fingerprint(),
        'provenance': {'source_revision': subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'source_sha256': {p: sha256(ROOT/p) for p in sources},
            'command': [sys.executable, *sys.argv], 'cwd': str(ROOT),
            'versions': {p:version(p) for p in ('jax','jaxlib','flax','evosax','transformers','numpy','Pillow')},
            'backend': jax.default_backend(), 'devices': [str(d) for d in jax.devices()]},
        'storage': 'archive.npz: complete controller offsets + base tensors, float32 embeddings, uint8 previews and float-image hashes; no pretrained CLIP weights.',
        'initial_parameters': 'candidate 0: zero offsets; other candidates: independent Gaussian offsets with mutation sigma',
        'limitations': ['Distinct JAX implementation, not simulator equivalence or NEAT evolution.',
            'Image/CLIP diversity does not prove behavioral diversity; color and initialization effects remain untested.',
            'Selection still builds a quadratic similarity matrix and repeatedly sorts its rows; giant archives are not validated.',
            'Optional selection traces retain pool embeddings across iterations; use only for bounded verification.']}
    util.save_json(output, 'meta', metadata)
    evaluate = evaluator(substrate, model, args.rollout_steps)
    batch = jax.jit(jax.vmap(evaluate))
    optimizer_key, init_key = jax.random.split(jax.random.PRNGKey(args.seed))
    params = args.sigma*jax.random.normal(init_key, (args.pop_size, substrate.n_params))
    params = params.at[0].set(jnp.zeros(substrate.n_params))
    if len(set(effective_hashes(np.asarray(params), substrate))) != args.pop_size:
        raise ValueError('Initialization did not produce distinct effective parameters; no repeated mutation attempts were made.')
    keys = jnp.broadcast_to(jax.random.PRNGKey(args.eval_seed), (args.pop_size, 2))
    initial_started = time.perf_counter()
    pop = batch(keys, params, jnp.arange(args.pop_size, dtype=jnp.int32))
    jax.block_until_ready(pop)
    metadata['initial_evaluation_seconds'] = time.perf_counter()-initial_started

    @jax.jit
    def do_iter(pop, key, first_id):
        parent_key, noise_key = jax.random.split(key)
        parents = jax.random.randint(parent_key, (args.n_child,), 0, args.pop_size)
        params = pop['params'][parents] + args.sigma*jax.random.normal(noise_key, (args.n_child, substrate.n_params))
        children = batch(jnp.broadcast_to(keys[0], (args.n_child,2)), params, first_id+jnp.arange(args.n_child,dtype=jnp.int32))
        pool = jax.tree.map(lambda a,b: jnp.concatenate((a,b)), pop, children)
        selected, kept, killed = select_archive(pool, args.k_nbrs, args.n_child)
        return selected, {'loss': asal_metrics.calc_illumination_score(selected['z']),
            'pool_ids': pool['candidate_id'], 'pool_z': pool['z'], 'kept_ids': pool['candidate_id'][kept],
            'killed_ids': pool['candidate_id'][killed]}

    history, traces, iteration_seconds = [], [], []
    save_interval = max(1, args.n_iters//10)
    for iteration in tqdm(range(args.n_iters)):
        tick = time.perf_counter()
        optimizer_key, key = jax.random.split(optimizer_key)
        pop, info = do_iter(pop, key, jnp.int32(args.pop_size+iteration*args.n_child))
        info = jax.device_get(info)
        if not np.isfinite(info['pool_z']).all(): raise ValueError('Nonfinite candidate embeddings')
        history.append(float(info['loss']))
        iteration_seconds.append(time.perf_counter()-tick)
        if args.record_selection: traces.append({k:v for k,v in info.items() if k!='loss'})
        metadata.update(completed_iterations=iteration+1, last_candidate_id=args.pop_size+(iteration+1)*args.n_child-1)
        if (iteration+1) % save_interval == 0:
            save_archive(output, pop, substrate, metadata, history, traces)
    metadata.update(status='complete', completed_iterations=args.n_iters,
        evaluated_candidates=args.pop_size+args.n_iters*args.n_child, iteration_seconds=iteration_seconds,
        elapsed_seconds=time.perf_counter()-started, peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    archive = save_archive(output, pop, substrate, metadata, history, traces)
    mapping = export_gallery(output, metadata, archive)
    metadata['counts'] = mapping['counts']
    metadata['output_bytes'] = sum(p.stat().st_size for p in output.iterdir() if p.is_file())
    util.save_json(output, 'meta', metadata)
    print(json.dumps({'output':str(output), **mapping['counts'], 'seconds':metadata['elapsed_seconds'], 'peak_rss_mib':metadata['peak_rss_mib']}))
    return metadata


if __name__ == '__main__':
    main(parse_args())
