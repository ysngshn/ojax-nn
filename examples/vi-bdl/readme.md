# Variational Inference methods for Bayesian deep learning

This example shows several practical variational inference methods implemented as custom NN optimizers using OJAX-NN. The available methods are
- [Bayes-by-backprop (BBB)](https://arxiv.org/abs/1505.05424): a classic from 2015 that is often implemented wrongly or used with suboptimal hyperparameters. Our implementation shows that it can work decently for NN training without "cold posterior", contrary to common (mis)beliefs;
- "Square gradient" variational online Newton (SGVON): the "gradient magnitude" variant of VON/VOGN described in [Khan et al.](https://arxiv.org/abs/1806.04854);
- Variational online Gauss Newton (VOGN): per-sample gradient variant of VON described in [Osawa et al.](https://arxiv.org/abs/1906.02506);
- Improved variational online Newton (IVON): yet another variant of VON which uses a Hessian estimator with second order correction term, cf. [Shen et al.](https://arxiv.org/abs/2402.17641)

We test these VI optimizers with ResNet-20 training using CIFAR-10 and report uncertainty metrics like NLL (loss), ECE, Brier score along with accuracy. 

Experiments can be run as follows:
```python
python run.py <opt_name> <init_lr>
```
e.g., `python run.py ivon 0.2`, `python run.py bbb 0.2`, `python run.py sgvon 0.02` or `python run.py vogn 0.02`.