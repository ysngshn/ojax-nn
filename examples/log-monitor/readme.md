# Example: log-monitor

This is a simple example to demonstrate the features from `ojnn.io` which helps
monitoring the training process. This includes:
- `ojnn.io.Logger`: a JAX-compatible `logging.Logger` counterpart to print log 
messages to console and/or log files. 
- `ojnn.io.MetricTracker`: a monitor that keeps track of running averages of 
values like metrics
- `ojnn.io.HostCSVMonitor`: a host monitor that records intermediate results 
into a csv file
- `ojnn.io.HostPlotMonitor`: a host monitor that interactively plots the 
results during training

We monitor the value from a noisy cosine function and log and monitor its
values. Notice that the log and monitor features offered from `ojnn.io` can be
used freely with JAX transformations and structured control flow primitives 
like `jax.lax.fori_loop`.

To run this example, simply run `python run.py` in this folder.