"""Distinct JAX Coralai implementation; not an equivalence-validated NEAT port.

Spatial offsets index array axes (first, second). Headings 0..7 index KERNEL[1:];
the center is a sensor/flow neighbor only. Sensors are center, forward, then the
remaining perimeter in kernel order. Actions retain DIR_ORDER's signed turns.
"""

import colorsys
import inspect

import numpy as np
import jax
import jax.numpy as jnp

# Same layout as experiment.py
KERNEL = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [-1, 1],
                   [-1, 0], [-1, -1], [0, -1], [1, -1]], dtype=np.int32)
DIR_ORDER = np.array([0, -1, 1], dtype=np.int32)   # fwd, left, right
N_DIRS = len(KERNEL) - 1                           # 8 perimeter headings
IMPLEMENTATION_VERSION = 'coralai-jax-v2-relative-heading'

N_SENSE = 6                  # energy, infra, com a/b/c/d
N_IN = N_SENSE * len(KERNEL) # 54: center + eight heading-relative neighbors
N_OUT = 10                   # invest, liquidate, explore x4, com x4


def _shift(x, dx, dy):
    """y[i, j] = x[(i+dx) % H, (j+dy) % W]; spatial axes are the first two."""
    return jnp.roll(x, shift=(-dx, -dy), axis=(0, 1))


def _scatter_shift(x, dx, dy):
    """Inverse of _shift: y[(i+dx) % H, (j+dy) % W] = x[i, j]."""
    return jnp.roll(x, shift=(dx, dy), axis=(0, 1))


def _disk(xs, ys, cx, cy, r):
    return (xs - cx) ** 2 + (ys - cy) ** 2 <= r ** 2


class CoralaiSubstrate:
    def __init__(self,
                 grid=512, n_genomes=4, hidden=16, base_seed=0, init_scale=0.1,
                 rollout_steps=500,
                 # --- physics (defaults = experiment.py / physics.py) ---
                 max_energy=1.5, max_infra=10.0,
                 claim_efficiency=0.9, loser_cost=0.2, min_bid=0.1,
                 defense_coeff=0.0, infra_decay=0.0, alive_threshold=0.05,
                 # --- PatchesEnvironment (defaults = environments.py) ---
                 n_patches=8, patch_radius_frac=0.07, max_capacity=30.0,
                 inject_rate=0.06, spawn_interval=250, n_spawn=None,
                 seed_infra=0.4,
                 # --- disturbances ---
                 chunk_interval=50, chunk_radius=5.0,
                 radiate_interval=50, radiate_spots=5,
                 radiate_dead_radius=4.0, radiate_live_radius=2.0):
        self.grid = grid
        self.n_genomes = n_genomes
        self.hidden = hidden
        self.base_seed = base_seed
        self.init_scale = init_scale
        self.rollout_steps = rollout_steps

        self.max_energy = max_energy
        self.max_infra = max_infra
        self.claim_efficiency = claim_efficiency
        self.loser_cost = loser_cost
        self.min_bid = min_bid
        self.defense_coeff = defense_coeff
        self.infra_decay = infra_decay
        self.alive_threshold = alive_threshold

        self.n_patches = n_patches
        self.patch_radius_frac = patch_radius_frac
        self.max_capacity = max_capacity
        self.inject_rate = inject_rate
        self.spawn_interval = spawn_interval
        self.n_spawn = n_patches if n_spawn is None else n_spawn
        self.seed_infra = seed_infra

        self.chunk_interval = chunk_interval
        self.chunk_radius = chunk_radius
        self.radiate_interval = radiate_interval
        self.radiate_spots = radiate_spots
        self.radiate_dead_radius = radiate_dead_radius
        self.radiate_live_radius = radiate_live_radius

        self._xs = jnp.arange(grid, dtype=jnp.float32)[:, None]
        self._ys = jnp.arange(grid, dtype=jnp.float32)[None, :]

        # Fixed random base network, one per genome id.
        k1, k2 = jax.random.split(jax.random.PRNGKey(base_seed))
        self._base = dict(
            w1=jax.random.normal(k1, (n_genomes, N_IN, hidden)) / np.sqrt(N_IN),
            b1=jnp.zeros((n_genomes, hidden)),
            w2=jax.random.normal(k2, (n_genomes, hidden, N_OUT)) / np.sqrt(hidden),
            b2=jnp.zeros((n_genomes, N_OUT)),
        )

        # One colour per genome id for rendering.
        self._palette = jnp.asarray(
            [colorsys.hsv_to_rgb((0.55 + k / n_genomes) % 1.0, 0.75, 1.0)
             for k in range(n_genomes)], dtype=jnp.float32)

    def settings(self):
        """Effective constructor settings, including the resolved spawn count."""
        return {name: getattr(self, name) for name in inspect.signature(type(self)).parameters}

    # ------------------------------------------------------------------
    # ASAL substrate interface
    # ------------------------------------------------------------------
    def default_params(self, rng):
        shapes = jax.tree.map(lambda a: a.shape, self._base)
        keys = jax.random.split(rng, 4)
        return {name: self.init_scale * jax.random.normal(k, shapes[name])
                for name, k in zip(sorted(shapes), keys)}

    def init_state(self, rng, params):
        H = self.grid
        k_g, k_r, k_p = jax.random.split(rng, 3)
        energy = jnp.zeros((H, H))
        infra = jnp.zeros((H, H))
        capacity = jnp.zeros((H, H))
        capacity, infra = self._add_patches(k_p, capacity, infra, self.n_patches)

        genome = jax.random.randint(k_g, (H, H), 0, self.n_genomes)
        rot = jax.random.randint(k_r, (H, H), 0, N_DIRS)
        # Life only exists where there is something to live on (patch cells
        # get infra at birth); everything else would die on step 1 anyway.
        genome = jnp.where(energy + infra > self.alive_threshold, genome, -1)

        return dict(energy=energy, infra=infra, com=jnp.zeros((H, H, 4)),
                    rot=rot, genome=genome, capacity=capacity,
                    t=jnp.zeros((), jnp.int32))

    def step_state(self, rng, state, params):
        k_chunk, k_spawn, k_rad = jax.random.split(rng, 3)
        t = state['t']
        energy, infra, com = state['energy'], state['infra'], state['com']
        rot, genome, capacity = state['rot'], state['genome'], state['capacity']

        # ---- 1. NN forward + activate_outputs ----------------------------
        raw = self._nn_forward(params, energy, infra, com, genome, rot)
        invest, liquidate, explore, com = self._activate(raw, genome)

        # ---- 2. physics ---------------------------------------------------
        delta = invest * energy - liquidate * infra               # invest_liquidate
        energy, infra = energy - delta, infra + delta
        infra, energy, genome, rot = self._explore(genome, rot, energy, infra, explore)
        energy, infra = self._energy_physics(energy, infra)
        genome = jnp.where(energy + infra > self.alive_threshold, genome, -1)

        # ---- 3. run_evolution --------------------------------------------
        if self.infra_decay > 0:
            infra = infra * (1.0 - self.infra_decay)
        energy = jnp.clip(energy, 0.0, 100.0)
        infra = jnp.clip(infra, 0.0, 100.0)
        genome = jax.lax.cond(t % self.chunk_interval == 0,
                              lambda g: self._kill_chunk(k_chunk, g),
                              lambda g: g, genome)

        # ---- 4. PatchesEnvironment.step ----------------------------------
        injection = jnp.minimum(capacity, self.inject_rate)
        energy = jnp.clip(energy + injection, 0.0, 100.0)
        capacity = capacity - injection
        capacity, infra = jax.lax.cond(
            (t + 1) % self.spawn_interval == 0,
            lambda c, i: self._add_patches(k_spawn, c, i, self.n_spawn),
            lambda c, i: (c, i), capacity, infra)

        # ---- 5. radiation -------------------------------------------------
        genome = jax.lax.cond((t % self.radiate_interval == 0) & (t > 0),
                              lambda g: self._radiate(k_rad, g),
                              lambda g: g, genome)

        return dict(energy=energy, infra=infra, com=com, rot=rot,
                    genome=genome, capacity=capacity, t=t + 1)

    def render_state(self, state, params, img_size=None):
        genome = state['genome']
        alive = genome >= 0
        cap = jnp.clip(state['capacity'] / self.max_capacity, 0.0, 1.0)
        # dim green glow = nutrient patches that still hold reserve
        bg = jnp.array([0.03, 0.04, 0.06]) + cap[..., None] * jnp.array([0.05, 0.30, 0.10])
        # living cells: genome hue, brightness from infrastructure, glow from energy
        col = self._palette[jnp.clip(genome, 0, self.n_genomes - 1)]
        bright = 0.45 + 0.55 * jnp.tanh(state['infra'])
        glow = 0.35 * jnp.tanh(state['energy'])
        fg = jnp.clip(col * bright[..., None] + glow[..., None], 0.0, 1.0)
        img = jnp.where(alive[..., None], fg, bg)
        if img_size is not None:
            img = jax.image.resize(img, (img_size, img_size, 3), method='bilinear')
        return img

    # ------------------------------------------------------------------
    # NN
    # ------------------------------------------------------------------
    def _sense_inputs(self, energy, infra, com, rot):
        sense = jnp.concatenate([jnp.log1p(jnp.maximum(energy, 0.0))[..., None],
                                 jnp.log1p(jnp.maximum(infra, 0.0))[..., None], com], axis=-1)
        perimeter = jnp.stack([_shift(sense, int(dx), int(dy)) for dx, dy in KERNEL[1:]], axis=-2)
        order = (rot[..., None] + jnp.arange(N_DIRS)) % N_DIRS
        relative = jnp.take_along_axis(perimeter, order[..., None], axis=-2)
        return jnp.concatenate([sense[..., None, :], relative], axis=-2).reshape((*rot.shape, N_IN))

    def _nn_forward(self, params, energy, infra, com, genome, rot):
        w1 = self._base['w1'] + params['w1']
        b1 = self._base['b1'] + params['b1']
        w2 = self._base['w2'] + params['w2']
        b2 = self._base['b2'] + params['b2']

        nbhd = self._sense_inputs(energy, infra, com, rot)
        h = jnp.tanh(jnp.einsum('hwi,kij->khwj', nbhd, w1) + b1[:, None, None, :])
        out = jnp.einsum('khwj,kjo->khwo', h, w2) + b2[:, None, None, :]     # (K,H,W,10)
        onehot = jax.nn.one_hot(genome, self.n_genomes)                      # dead -> zeros
        return jnp.einsum('hwk,khwo->hwo', onehot, out)

    # ------------------------------------------------------------------
    # physics.py ports
    # ------------------------------------------------------------------
    def _activate(self, raw, genome):
        invest_raw, liquidate_raw = raw[..., 0], raw[..., 1]
        explore_raw, com_raw = raw[..., 2:6], raw[..., 6:10]

        mean = com_raw.mean(axis=(0, 1), keepdims=True)
        var = com_raw.var(axis=(0, 1), keepdims=True)
        com = jax.nn.sigmoid((com_raw - mean) / jnp.sqrt(var + 1e-5))

        trade = jnp.tanh(invest_raw - liquidate_raw)
        invest, liquidate = jax.nn.relu(trade), jax.nn.relu(-trade)

        ev = jax.nn.relu(explore_raw)
        ev = ev.at[..., 0].set(ev.mean(axis=-1))          # "no" channel = mean
        explore = jax.nn.softmax(ev, axis=-1)

        alive = genome >= 0
        return (invest * alive, liquidate * alive,
                explore * alive[..., None], com)

    def _explore(self, genome, rot, energy, infra, explore):
        kernel = jnp.asarray(KERNEL[1:])
        dir_order = jnp.asarray(DIR_ORDER)

        max_act = jnp.argmax(explore, axis=-1)
        att_dir = (rot + dir_order[jnp.clip(max_act - 1, 0, 2)] + N_DIRS) % N_DIRS
        att_dx, att_dy = kernel[att_dir][..., 0], kernel[att_dir][..., 1]
        # a "serious attacker" pays loser_cost whether or not it wins
        serious = ((genome >= 0) & (max_act != 0)
                   & (energy >= self.min_bid))

        best_bid = jnp.where(genome >= 0,
                             energy + infra * self.defense_coeff, 0.0)
        win_off = jnp.full(genome.shape, -1, jnp.int32)
        win_genome, win_rot = genome, rot

        for k in range(1, len(KERNEL)):
            dx, dy = int(KERNEL[k, 0]), int(KERNEL[k, 1])
            points_here = (_shift(serious, dx, dy)
                           & (_shift(att_dx, dx, dy) + dx == 0)
                           & (_shift(att_dy, dx, dy) + dy == 0))
            bid = _shift(energy, dx, dy)
            better = points_here & (bid > best_bid)          # strict, like the kernel
            best_bid = jnp.where(better, bid, best_bid)
            win_off = jnp.where(better, k, win_off)
            win_genome = jnp.where(better, _shift(genome, dx, dy), win_genome)
            win_rot = jnp.where(better, _shift(att_dir, dx, dy), win_rot)

        won = win_off >= 0
        infra_delta = jnp.where(won, self.claim_efficiency * best_bid, 0.0)
        energy_delta = -self.loser_cost * energy * serious
        for k in range(1, len(KERNEL)):            # winner pays the rest of its bid
            dx, dy = int(KERNEL[k, 0]), int(KERNEL[k, 1])
            charge = jnp.where(win_off == k, (1.0 - self.loser_cost) * best_bid, 0.0)
            energy_delta = energy_delta - _scatter_shift(charge, dx, dy)

        return infra + infra_delta, energy + energy_delta, win_genome, win_rot

    def _energy_physics(self, energy, infra):
        offs = [(int(dx), int(dy)) for dx, dy in KERNEL]

        # flow_energy_up: energy is pushed to neighbours proportional to their infra
        s = sum(_shift(infra, dx, dy) for dx, dy in offs)
        ok = s > 1e-8
        ratio = jnp.where(ok, energy / jnp.maximum(s, 1e-8), 0.0)
        energy = (infra * sum(_scatter_shift(ratio, dx, dy) for dx, dy in offs)
                  + jnp.where(ok, 0.0, energy))

        # distribute_energy / distribute_infra: overflow spreads evenly over 3x3
        def spread(x, cap):
            over = x > cap
            share = jnp.where(over, x / len(KERNEL), 0.0)
            return jnp.where(over, 0.0, x) + sum(_scatter_shift(share, dx, dy)
                                                 for dx, dy in offs)

        return spread(energy, self.max_energy), spread(infra, self.max_infra)

    # ------------------------------------------------------------------
    # environment / disturbances
    # ------------------------------------------------------------------
    def _add_patches(self, rng, capacity, infra, n):
        radius = self.patch_radius_frac * self.grid
        centers = jax.random.uniform(rng, (n, 2), minval=0.0, maxval=float(self.grid))

        def body(carry, c):
            capacity, infra = carry
            new = _disk(self._xs, self._ys, c[0], c[1], radius) & (capacity == 0)
            capacity = capacity + new * self.max_capacity
            infra = jnp.clip(infra + new * self.seed_infra, 0.0, 100.0)
            return (capacity, infra), None

        (capacity, infra), _ = jax.lax.scan(body, (capacity, infra), centers)
        return capacity, infra

    def _kill_chunk(self, rng, genome):
        c = jax.random.randint(rng, (2,), 0, self.grid)
        mask = _disk(self._xs, self._ys, c[0], c[1], self.chunk_radius)
        return jnp.where(mask, -1, genome)

    def _radiate(self, rng, genome):
        k_xy, k_g = jax.random.split(rng)
        xy = jax.random.randint(k_xy, (self.radiate_spots, 2), 0, self.grid)
        gid = jax.random.randint(k_g, (self.radiate_spots,), 0, self.n_genomes)

        def body(g, inp):
            c, new_id = inp
            g = jnp.where(_disk(self._xs, self._ys, c[0], c[1], self.radiate_dead_radius), -1, g)
            g = jnp.where(_disk(self._xs, self._ys, c[0], c[1], self.radiate_live_radius), new_id, g)
            return g, None

        genome, _ = jax.lax.scan(body, genome, (xy, gid))
        return genome


if __name__ == "__main__":   # python -m substrates.coralai_new
    import time
    sub = CoralaiSubstrate()
    rng = jax.random.PRNGKey(0)
    params = jax.tree.map(jnp.zeros_like, sub.default_params(rng))
    state = sub.init_state(rng, params)
    step = jax.jit(sub.step_state)
    t0 = time.time()
    for i in range(sub.rollout_steps):
        rng, r = jax.random.split(rng)
        state = step(r, state, params)
        if i % 100 == 0:
            print(f"step {i:4d}  alive={int((state['genome'] >= 0).sum()):5d}  "
                  f"E={float(state['energy'].sum()):8.1f}  I={float(state['infra'].sum()):8.1f}")
    print("img", sub.render_state(state, params, img_size=224).shape, f"{time.time()-t0:.1f}s")
