import jax.numpy as jnp

import substrates
import jax
import pickle
from jax.flatten_util import ravel_pytree

pop = pickle.load(open("illumination_results/test4/pop.pkl", "rb"))
data = pickle.load(open("illumination_results/test4/data.pkl", "rb"))


base = substrates.create_substrate("coralai")   # NOT wrapped in Flatten
p = base.default_params(jax.random.PRNGKey(0))

_, unravel = ravel_pytree(p)
named = unravel(jnp.array(pop['params'][0]))  # unflatten the first individual in the population
print(jax.tree.map(lambda x: x.shape, named))