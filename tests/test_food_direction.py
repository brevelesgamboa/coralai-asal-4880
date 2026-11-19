"""Opt-in reversal, reserve-only translation, and real runner capture phase."""
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'external/coralai/experiments/coral_dev')]
from environments import TrailEnvironment, make_env
from scripts.compare_trail_reference import load_state, load_resource


def substrate(shape=400):
    mem = torch.zeros(1, 14, shape, shape)
    mem[0, 1] = 1
    mem[0, 13] = -1
    return SimpleNamespace(mem=mem, torch_device=torch.device('cpu'),
        ti_indices={None: SimpleNamespace(energy=0, infra=1, genome=13)})


class FoodDirectionTests(unittest.TestCase):
    def test_reversal_is_opt_in_and_default_metadata_is_unchanged(self):
        env = TrailEnvironment()
        self.assertEqual(env.reverse_after_update, 0)
        self.assertNotIn('reverse_after_update', env.to_dict())
        for value in (-1, True, 1.5, None):
            with self.assertRaises(ValueError):
                make_env('trail', options={'reverse_after_update': value})
        self.assertEqual(make_env('trail', options={'reverse_after_update':500}).to_dict()['reverse_after_update'],500)

    def test_exact_schedule_before_release_and_no_boundary_crossing(self):
        subs = [substrate(), substrate()]
        envs = [TrailEnvironment(shift_interval=125, shift_cells=2, reverse_after_update=x) for x in (0,500)]
        for env, sub in zip(envs, subs):
            env.seed(sub)
        initial=envs[0]._capacity.clone()
        depleted=initial.clone()
        unchanged=[s.mem[:,1:].clone() for s in subs]
        offsets=[0,0]
        for u in range(1,1001):
            for i,(env,sub) in enumerate(zip(envs,subs)):
                if u % 125 == 0:
                    offsets[i] += -2 if i==1 and u>500 else 2
                env.step(sub)
                expected = torch.roll(depleted, offsets[i], dims=1)
                self.assertTrue(torch.equal(env._last_injection, expected.clamp(max=.02)))
                self.assertTrue(torch.equal(sub.mem[:,1:], unchanged[i]))
                support=torch.where(env._capacity>0)[1]
                self.assertGreaterEqual(int(support.min()),18)
                self.assertLessEqual(int(support.max()),398)
            depleted.sub_(depleted.clamp(max=.02)).clamp_(min=0)
            for env,offset in zip(envs,offsets):
                self.assertTrue(torch.equal(env._capacity,torch.roll(depleted,offset,dims=1)))
                self.assertEqual(env.resource_state()['remaining_capacity'],float(depleted.double().sum()))
            if u<=624:
                self.assertTrue(torch.equal(subs[0].mem,subs[1].mem))
            if u==500:self.assertEqual(offsets,[8,8])
            if u==625:self.assertEqual(offsets,[10,6])
        self.assertEqual(offsets,[16,0])

    def test_reversal_does_not_translate_previously_available_energy(self):
        sub=substrate(32)
        env=TrailEnvironment(shift_interval=1,shift_cells=2,reverse_after_update=1)
        env.seed(sub)
        for _ in range(3):
            before=sub.mem.clone()
            env.step(sub)
            self.assertTrue(torch.equal(sub.mem[0,0],(before[0,0]+env._last_injection).clamp(0,100)))
            self.assertTrue(torch.equal(sub.mem[:,1:],before[:,1:]))

    def test_real_headless_capture_after_reversed_shift(self):
        base=json.loads((ROOT/'configs/food_movement_comparison/faster.json').read_text())
        base.update(shape=32,steps=4,screenshot_interval=1,max_screenshots=5,checkpoint_interval=4)
        base['env_options'].update(shift_interval=2,reverse_after_update=2)
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp)
            (tmp/'config.json').write_text(json.dumps(base))
            result=subprocess.run([sys.executable,'-B',str(ROOT/'external/coralai/experiments/coral_dev/headless_coral.py'),
                '--config',str(tmp/'config.json'),'--output-dir',str(tmp/'run')],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            with (tmp/'run/resource_metrics.csv').open() as stream:
                counts={int(r['completed_update']):int(r['occupied_cells']) for r in csv.DictReader(stream)}
            original=load_resource(tmp/'run',0)['capacity']
            expected=torch.from_numpy(original.copy())
            for u in range(5):
                if u:expected.sub_(expected.clamp(max=.02)).clamp_(min=0)
                offset=2 if u in (2,3) else 0
                self.assertTrue(np.array_equal(load_resource(tmp/'run',u)['capacity'],torch.roll(expected,offset,dims=1).numpy()))
                state=load_state(tmp/'run',u)
                self.assertEqual(int((state[0,13]>=0).sum()),counts[u])
            self.assertTrue(torch.equal(load_state(tmp/'run',4),torch.load(tmp/'run/checkpoint_0000004/substrate.pt',weights_only=True)))


if __name__=='__main__':unittest.main()
