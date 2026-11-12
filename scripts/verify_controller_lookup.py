"""Deterministic controller assignment check; no ecosystem rollout."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import taichi as ti
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "external/coralai"), str(ROOT / "external/coralai/coralai/dependencies/PyTorch-NEAT")]


def fixture(module, n=8):
    """Real registration/kernel, bias-only networks, one sensor/one neighbor.

    Using one neighbor isolates lookup assignment from the separate coral
    neighborhood-layout defect. Active IDs 0, 2, 7 are deliberately sparse.
    """
    indices = ti.types.struct(genome=ti.i32, rot=ti.i32).field(shape=())
    indices[None] = dict(genome=13, rot=12)
    mem = torch.zeros(1, 14, 8, 8)
    mem[0, 13] = -1
    sub = SimpleNamespace(mem=mem, ti_indices=indices, torch_device=torch.device("cpu"), w=8, h=8)
    ev = module.SpaceEvolver.__new__(module.SpaceEvolver)
    ev.substrate = sub
    ev.torch_device = sub.torch_device
    ev.genomes, ev.genome_keys, ev.ages = [], [], []
    ev.combined_weights, ev.combined_biases = [], []
    ev._next_key = 0
    ev._cached_cw = ev._cached_cb = ev._key_to_local_cache = None
    ev._weights_dirty = ev._key_map_dirty = True
    ev._scratch = {}
    ev.sense_chinds = [0]
    ev.act_chinds = [2]
    ev.dir_kernel = torch.tensor([[1, 0]], dtype=torch.int32)
    ev.dir_order = torch.tensor([0], dtype=torch.int32)
    ev.timestep = ev.time_last_cull = 0
    ev.energy_offset = 0.0
    ev.create_torch_net = lambda genome: genome
    for key in range(n):
        network = SimpleNamespace(weights=torch.zeros(1, 1, 2), biases=torch.tensor([[[10.0 + key]]]))
        assert ev.add_organism_get_key(network) == key
    return ev


def assigned_outputs(module, ev):
    output = torch.full((1, 8, 8), -999.0)
    w, b = ev.get_combined_weights()
    module.apply_weights_and_biases(ev.substrate.mem, output, torch.tensor(ev.sense_chinds), w, b,
                                  ev.dir_kernel, ev.dir_order, ev.substrate.ti_indices, ev._get_key_to_local())
    return output


def check(module):
    ev = fixture(module)
    ids = [0, 2, 7, -1, -2]
    for x, key in enumerate(ids): ev.substrate.mem[0, 13, x, 0] = key
    out = assigned_outputs(module, ev)
    actual = [float(out[0, x, 0]) for x in range(len(ids))]
    expected = [10.0, 12.0, 17.0, 0.0, 0.0]
    return {"cell_genome_ids": ids, "registered_ids": ev.genome_keys,
            "lookup": ev._get_key_to_local().tolist(), "expected_outputs": expected,
            "actual_outputs": actual, "passed": actual == expected,
            "fixture": "Real registration and forward kernel; bias-only controllers; one sensor/one neighbor to isolate lookup."}


def layout_check(module):
    ev = fixture(module)
    ev.sense_chinds = [0, 1, 8, 9, 10, 11]
    ev.dir_kernel = torch.tensor([[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]])
    ev.dir_order = torch.tensor([0, -1, 1])
    ev.combined_weights = [torch.zeros(1, 1, 24) for _ in ev.genomes]
    ev._invalidate_weight_cache()
    ev.substrate.mem[0, 13, 0, 0] = 0
    try:
        value = float(assigned_outputs(module, ev)[0, 0, 0])
        return {"stored_inputs_per_controller": 24, "passed": value == 10.0, "actual_output": value}
    except Exception as error:
        return {"stored_inputs_per_controller": 24, "passed": False,
                "error": type(error).__name__ + ": " + str(error)[:700]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "external/coralai/coralai/evolver.py")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expect-broken", action="store_true")
    args = parser.parse_args()
    ti.init(arch=ti.cpu, cpu_max_num_threads=1, debug=True)
    spec = importlib.util.spec_from_file_location("coralai.lookup_verification", args.source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    report = check(module)
    report["coral_layout"] = layout_check(module)
    report["source"] = str(args.source.resolve())
    report["source_sha256"] = hashlib.sha256(args.source.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    assert report["passed"] != args.expect_broken, "Unexpected controller-assignment result"
    assert report["coral_layout"]["passed"] != args.expect_broken, "Unexpected coral-layout result"


if __name__ == "__main__": main()
