# MNIST classification with MLP

This is a simple example of MNIST classification task with an MLP model to 
showcase different features offered by OJAX-NN. This includes:
- Defining a simple MLP with `ojnn` layers and `ojnn.Sequential`
- Examples of SGD optimization scheduling both learning rate and momentum, 
achieved with either
  - A custom implementation using `ojnn.optim.Optimizer` and  `ojnn.optim.Scheduler`
  - An Optax based wrapper `ojnn.optim.SGD` and `ojnn.optim.inject_hyperparams`
- Loading and using a custom MNIST dataset class with `ojnn.io.NumpyDataset` 
and `ojnn.io.foreach_loop`
- Track metrics with `ojnn.io.MetricTracker`
- Log training / test progress with `ojnn.io.Logger`

This example can be run with 2 scripts:
- [run_custom.py](./run_custom.py): This example shows how a full training 
pipeline can be built solely with `ojnn` and standard JAX features including 
JAX transforms (`jit`, `grad`, etc.) and structured control flow primitives 
(`lax.cond`, `lax.while_loop`,`lax.fori_loop`, etc.) 
- [run_optax.py](./run_optax.py): Alternative version with Optax wrappers and
Optax features like schedulers, loss and `inject_hyperparams` extension