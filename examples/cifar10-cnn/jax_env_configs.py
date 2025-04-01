# some JAX configs
import os

# disable gpu memory pre-allocation
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# deterministic JAX ops. cf.: https://github.com/google/jax/issues/13672
os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
os.environ["TF_DETERMINISTIC_ops"] = "1"

import jax

jax.config.update("jax_threefry_partitionable", True)
