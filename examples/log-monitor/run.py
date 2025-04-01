from functools import partial
from time import sleep
from jax import jit
import jax.numpy as jnp
from jax.lax import fori_loop, cond
from jax.random import key as jrkey, normal as jrnormal, fold_in as jrfold_in
from ojnn.io import (
    Logger,
    HostCSVMonitor,
    HostPlotMonitor,
    MultiMonitor,
    MetricTracker,
    run_with_plot,
    host_callback,
)


def noisy_cosine(rngkey, step, total):
    return jnp.cos(jnp.pi * step / total) + 0.1 * jrnormal(rngkey, ())


if __name__ == "__main__":
    total_steps = 10000
    momentum = 0.99
    rngkey = jrkey(42)

    # logger to print info to console and log file
    logger = Logger(save_to="./run_log.txt")

    # metric tracker to track the running average and std
    tracker = MetricTracker(momentum=momentum, init_value=1.0)

    # monitor the value and its running stats
    monitor = MultiMonitor(
        csv_val=HostCSVMonitor("./noisy_cosine.csv", append=False).jaxify(),
        plot_val=HostPlotMonitor("noisy_cosine").jaxify(),
    )

    # run the process
    logger = logger.info("Process started.")
    tracker = tracker.start("value")

    @jit
    def loop_body(idx, state):
        rngkey, logger, tracker, monitor = state
        # get another noisy sample
        val = noisy_cosine(rngkey, idx, total_steps)
        # wait for 0.002 second, simulate costly compute
        host_callback(partial(sleep, 0.002))()
        # track the moving average of the value and monitor its stats
        tracker = tracker.log(value=val)
        avg = tracker.value
        monitor = monitor.log(step=idx, value=val, smoothed=avg)
        # print log status every 100 steps
        logger = cond(
            idx % 100 == 0,
            partial(
                logger.info,
                "step %d: value %.4f, smoothed %.4f",
                idx,
                val,
                avg,
            ),
            lambda: logger,
        )
        return jrfold_in(rngkey, idx), logger, tracker, monitor

    # context needed for plot monitor, launches a separate process for plotting
    with run_with_plot(".", block=True):
        monitor = monitor.start("step", "value", "smoothed")
        _, logger, tracker, monitor = fori_loop(
            0, total_steps, loop_body, (rngkey, logger, tracker, monitor)
        )
        monitor.stop()

    tracker.stop()
    logger.info("Done!")
    logger.close()
