"""Independent direction/controller fixtures and bounded archive/replay regressions."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

import main_illuminate as atlas
import substrates
from substrates.coralai_new import CoralaiSubstrate, N_IN

# Independent oracle: array-axis offsets, starting with positive first axis.
PERIMETER = ((1,0), (1,1), (0,1), (-1,1), (-1,0), (-1,-1), (0,-1), (1,-1))


class ControllerTests(unittest.TestCase):
    def test_heading_fix_does_not_change_nine_cell_resource_distribution(self):
        sub=CoralaiSubstrate(grid=5,n_genomes=2,hidden=2)
        energy=jnp.zeros((5,5)).at[2,2].set(18.)
        energy2,infra2=sub._energy_physics(energy,jnp.ones((5,5)))
        self.assertAlmostEqual(float(energy2.sum()),18.,places=5)
        self.assertAlmostEqual(float(infra2.sum()),25.,places=5)

    def test_all_headings_and_turns_claim_perimeter_including_wraps(self):
        sub = CoralaiSubstrate(grid=9, n_genomes=2, hidden=2)
        run = jax.jit(sub._explore)
        for parent in ((4,4), (0,0)):
            for heading in range(8):
                for action, turn in ((1,0), (2,-1), (3,1)):
                    with self.subTest(parent=parent, heading=heading, action=action):
                        genome = jnp.full((9,9), -1).at[parent].set(1)
                        rot = jnp.full((9,9), heading)
                        energy = jnp.zeros((9,9)).at[parent].set(1.)
                        explore = jnp.zeros((9,9,4)).at[parent+(action,)].set(1.)
                        infra2, energy2, genome2, rot2 = run(genome, rot, energy, jnp.zeros((9,9)), explore)
                        direction = (heading+turn)%8
                        dx,dy = PERIMETER[direction]
                        target = ((parent[0]+dx)%9, (parent[1]+dy)%9)
                        expected = np.asarray(genome).copy(); expected[target]=1
                        np.testing.assert_array_equal(genome2, expected)
                        self.assertEqual(int(rot2[target]), direction)
                        self.assertAlmostEqual(float(infra2[target]), .9, places=6)
                        self.assertAlmostEqual(float(energy2[parent]), 0., places=6)

    def test_sensing_is_center_then_heading_relative_with_54_inputs(self):
        sub = CoralaiSubstrate(grid=5, n_genomes=2, hidden=2)
        com = jnp.zeros((5,5,4)).at[...,0].set(jnp.arange(5)[:,None]*10+jnp.arange(5)[None,:])
        for heading in range(8):
            features = sub._sense_inputs(jnp.zeros((5,5)), jnp.zeros((5,5)), com, jnp.full((5,5),heading))
            expected = [22]
            for offset in range(8):
                dx,dy = PERIMETER[(heading+offset)%8]
                expected.append((2+dx)*10+2+dy)
            self.assertEqual(N_IN,54)
            np.testing.assert_array_equal(np.asarray(features[2,2]).reshape(9,6)[:,2],expected)

    def test_flatten_mapping_and_genome_specific_networks(self):
        sub = CoralaiSubstrate(grid=5, n_genomes=2, hidden=2)
        sub._base = jax.tree.map(jnp.zeros_like, sub._base)
        params = jax.tree.map(jnp.zeros_like, sub._base)
        params['w1'] = params['w1'].at[0,0,0].set(1.)
        params['w2'] = params['w2'].at[0,0,0].set(2.)
        params['b2'] = params['b2'].at[1].set(jnp.arange(10)+10.)
        flat = substrates.FlattenSubstrateParameters(sub)
        vector = flat.param_reshaper.flatten_single(params)
        restored = flat.param_reshaper.reshape_single(vector)
        for key in params: np.testing.assert_array_equal(restored[key],params[key])
        genome = jnp.full((5,5),-1).at[0,0].set(0).at[1,1].set(1)
        energy = jnp.full((5,5),jnp.expm1(.25))
        run = lambda p: sub._nn_forward(p,energy,jnp.zeros((5,5)),jnp.zeros((5,5,4)),genome,jnp.zeros((5,5),dtype=int))
        actual = run(restored)
        expected0 = np.zeros(10); expected0[0]=2*np.tanh(.25)
        np.testing.assert_allclose(actual[0,0],expected0,atol=1e-7)
        np.testing.assert_array_equal(actual[1,1],np.arange(10)+10)
        np.testing.assert_array_equal(actual[2,2],np.zeros(10))
        changed = dict(restored); changed['b2']=changed['b2'].at[1,9].add(3.)
        changed_vector=flat.param_reshaper.flatten_single(changed)
        after=run(flat.param_reshaper.reshape_single(changed_vector))
        self.assertEqual(float(after[1,1,9]),22.)
        np.testing.assert_array_equal(after[0,0],actual[0,0])

    def test_world_key_and_rollouts_do_not_share_mutable_state(self):
        raw=CoralaiSubstrate(grid=8,n_genomes=2,hidden=2)
        sub=substrates.FlattenSubstrateParameters(raw)
        key=jax.random.PRNGKey(42)
        params=jax.tree.map(jnp.zeros_like,raw._base)
        other=dict(params); other['b2']=other['b2'].at[:,1].set(4.)
        a=sub.param_reshaper.flatten_single(params); b=sub.param_reshaper.flatten_single(other)
        start_a=sub.init_state(key,a); start_b=sub.init_state(key,b)
        for name in start_a: np.testing.assert_array_equal(start_a[name],start_b[name])
        before=jax.tree.map(np.asarray,start_a)
        model,_=atlas.create_model('pixels')
        evaluate=atlas.evaluator(sub,model,2)
        first=evaluate(key,a,jnp.int32(0)); second=evaluate(key,b,jnp.int32(1)); repeated=evaluate(key,a,jnp.int32(2))
        np.testing.assert_array_equal(first['rgb'],repeated['rgb'])
        self.assertFalse(np.array_equal(first['rgb'],second['rgb']))
        for name in before: np.testing.assert_array_equal(before[name],start_a[name])
        np.testing.assert_array_equal(a,sub.param_reshaper.flatten_single(params))


class ArchiveTests(unittest.TestCase):
    def test_selection_keeps_identity_keys_and_params_together(self):
        # Entries 0/1 are identical; removing the first tied duplicate leaves 1,2,3.
        pool={'z':jnp.array([[1.,0.],[1.,0.],[0.,1.],[-1.,0.]]),
              'candidate_id':jnp.array([10,20,30,40]),'params':jnp.arange(8).reshape(4,2),
              'eval_key':jnp.array([[0,2],[0,3],[0,4],[0,5]],dtype=jnp.uint32)}
        selected,kept,killed=atlas.select_archive(pool,1,1)
        np.testing.assert_array_equal(killed,[0]); np.testing.assert_array_equal(kept,[1,2,3])
        for name in pool: np.testing.assert_array_equal(selected[name],np.asarray(pool[name])[[1,2,3]])

    def test_short_runs_save_final_and_protect_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); config=root/'config.json'
            config.write_text(json.dumps(dict(grid=8,n_genomes=2,hidden=2,rollout_steps=1)))
            for iterations in (0,1,5,9):
                output=root/f'iterations_{iterations}'
                args=atlas.parse_args(['--save_dir',str(output),'--substrate','coralai','--substrate_config',str(config),
                    '--foundation_model','pixels','--pop_size','3','--n_child','1','--k_nbrs','1','--n_iters',str(iterations)])
                with contextlib.redirect_stdout(io.StringIO()): atlas.main(args)
                metadata,archive=atlas.load_archive(output)
                self.assertEqual(metadata['completed_iterations'],iterations)
                self.assertEqual(metadata['status'],'complete')
                self.assertEqual(metadata['substrate']['settings']['rollout_steps'],1)
                self.assertEqual(len(archive['candidate_id']),3)
                self.assertEqual(archive['rgb'].dtype,np.uint8)
                self.assertEqual(metadata['counts']['unique_effective_parameters'],3)
                np.testing.assert_array_equal(archive['eval_key'],np.tile([0,42],(3,1)))
                before=atlas.sha256(output/'archive.npz')
                with self.assertRaises(FileExistsError): atlas.main(args)
                self.assertEqual(before,atlas.sha256(output/'archive.npz'))
                if iterations==1:
                    replay=atlas.replay_archive(output,[int(archive['candidate_id'][0])])
                    self.assertEqual(replay[0]['uint8_image_max_abs_difference'],0)
                    self.assertLess(replay[0]['embedding_max_abs_difference'],1e-5)
                    metadata['world_source_sha256']={}
                    (output/'meta.json').write_text(json.dumps(metadata))
                    with self.assertRaisesRegex(ValueError,'fingerprint'): atlas.load_archive(output)

    def test_legacy_archive_is_rejected_without_guessing(self):
        with self.assertRaisesRegex(ValueError,'Legacy/unidentified'):
            atlas.load_archive(atlas.ROOT/'illumination_results/test4')

    def test_export_has_all_65_entries_and_reports_image_duplicates(self):
        images=np.zeros((65,224,224,3),dtype=np.uint8)
        for i in range(65): images[i,...,0]=i
        images[1]=images[0]
        archive={'candidate_id':np.arange(65), 'rgb':images, 'eval_key':np.tile([0,42],(65,1)),
                 'effective_parameter_sha256':np.array([atlas.array_hash(np.array(i)) for i in range(65)],dtype='S64')}
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)
            gallery=atlas.export_gallery(output,{'foundation_model':{'name':'pixels'}},archive)
            self.assertEqual(len(gallery['pages']),2)
            self.assertEqual(len(gallery['entries']),65)
            self.assertEqual(gallery['counts']['unique_rendered_images'],64)
            self.assertEqual(gallery['counts']['duplicate_image_entries'],1)
            for entry in gallery['entries']:
                with Image.open(output/entry['page']) as sheet:
                    np.testing.assert_array_equal(sheet.crop(entry['image_box']),images[entry['archive_index']])

    def test_factory_preserves_declared_rollout_default(self):
        self.assertEqual(substrates.create_substrate('coralai').rollout_steps,500)


if __name__=='__main__': unittest.main()
