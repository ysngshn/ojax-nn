## OJAX-NN

OJAX-NN is a JAX-based neural network (NN) library which has the following features:

- __Simply JAX__: OJAX-NN is fully compatible with all standard JAX features, including transformations like `vmap`, `jit`, `grad`, and structured control flow primitives like `cond`, `scan`, `fori_loop`, etc. Simply use JAX as you know it, `jit` your whole training for efficiency, and combine JAX and OJAX-NN features however you wish. 
- __Fully packed__: OJAX-NN contains all necessary components for typical deep learning projects: a module system to define network architectures, optimizers for training, and `jit`-friendly input/output interactions (load data, monitor results, etc.).  
- __Easy customization__: OJAX-NN provides a set of flexible interfaces that are easy to extend and customize. You can easily design new network layers and architectures, implement custom optimizers, or use the latest datasets. 

OJAX-NN enables researchers and practitioners to easily set up deep learning projects and freely construct and test new prototypes.

> OJAX-NN is currently a work in progress. The main functionalities are here, but docs and tests are still lacking. Feel free to check out [The `examples` folder](./examples/) and read the source code to see how OJAX-NN can be used.

>OJAX-NN API might see some changes in the future (hopefully for the better!). Feedback is welcome!

## Installation

For now, you can clone this repo and install locally via, e.g., `pip install -e .`

## Why yet another JAX NN library?

There are already many awesome JAX NN libraries: [Flax](https://github.com/google/flax), [Equinox](https://github.com/patrick-kidger/equinox), [Haiku](https://github.com/deepmind/dm-haiku), [Stax](https://docs.jax.dev/en/latest/jax.example_libraries.stax.html), just to name a few. However, JAX NN libraries have some common "pain points" that none of the existing offerings can fully address, especially:

- _JAX feature incompatibility_: many libraries cannot incorporate standard JAX transformations like `grad` or `vmap`, and instead introduce their own variants. Users feel like using a different language that is only partially relevant to JAX.
- _Missing necessary "non-JAX" parts_: while JAX focuses on pure functional computations, a typical deep learning project also requires "impure" features like reading data from disk or monitoring results. However, this is often not supported by existing libraries, and users typically need to additionally install Pytorch/Tensorflow just to perform dataloading.

## Quick start

OJAX-NN is composed of three parts: `ojnn.modules`, `ojnn.optim`, and `ojnn.io`.

### ojnn.modules: defining neural network models

Defining model architectures and custom layers.

### ojnn.optim: deep learning optimizers

Optimize the model during training, integrate Optax & support custom optimizers.

### ojnn.io: input/output interface

Data loading, logging, monitoring training curves, saving results, etc.

## Examples

[The `examples` folder](./examples/) contains various working examples to showcase various OJAX-NN features and use cases. They can be run on a single GPU for quick experimentation. Please feel free to have a look ;)