"""run CIFAR-10 classification with ResNet, monitor training with matplotlib"""

from functools import partial
from time import perf_counter
import sys
import os
from os.path import join as opjoin

# - disable gpu memory pre-allocation
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# - deterministic ops, slower. cf.: https://github.com/google/jax/issues/13672
# os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true"
# os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
# os.environ["TF_DETERMINISTIC_ops"] = "1"

from jax import jit, Array, ShapeDtypeStruct
from jax.dtypes import canonicalize_dtype
from jax.numpy import argmax, take_along_axis, expand_dims, clip, pi, cos
from jax.nn import log_softmax, logsumexp
from jax.lax import fori_loop, cond, map
from jax.random import split as jrsplit, key as jrkey
import datasets
from ojnn import Module
from ojnn.optim import Scheduler, make_supervised_objective
from ojnn.io import (
    load,
    save,
    foreach_loop,
    Logger,
    HostCSVMonitor,
    HostPlotMonitor,
    MultiMonitor,
    MetricTracker,
    run_with_plot,
    host_callback,
)
from data_utils import (
    get_cifar10,
    CIFAR10_MEAN_STD,
    ComposeTransform,
    ImageToArray,
    Normalize,
    RandomHorizontalFlip,
    Pad,
    RandomCrop,
)
from models import resnet20
from vi_optim import VIOptimizer, BayesByBackprop, SGVON, VOGN, IVON
from calibration import brier_score, ECETracker


vi_optimizers: dict[str, type[VIOptimizer]] = {
    "bbb": BayesByBackprop,
    "sgvon": SGVON,
    "vogn": VOGN,
    "ivon": IVON,
}


def get_opt_name(optimizer: VIOptimizer) -> str:
    opt_name_map = {v: k for k, v in vi_optimizers.items()}
    return opt_name_map[type(optimizer)]


def make_warmup_cosine_schedule(
    value_start: float, warmup_steps: float, total_steps: float
):
    assert total_steps > warmup_steps

    def warmup_cosine_schedule(step):
        step = clip(step, max=total_steps)
        return cond(
            step < warmup_steps,
            lambda: (
                value_start * step
                + value_start / warmup_steps * (warmup_steps - step)
            )
            / warmup_steps,
            lambda: (
                0.5 * value_start * cos(pi * step / total_steps)
                + 0.5 * value_start
            ),
        )

    return warmup_cosine_schedule


def get_time() -> Array:
    return host_callback(
        perf_counter,
        debug_impl=False,
        result_shape_dtypes=ShapeDtypeStruct(
            shape=(), dtype=canonicalize_dtype(float)
        ),
    )()


def acc(output: Array, target: Array) -> Array:
    preds = argmax(output, axis=-1)
    return (preds == target).mean()


def cross_entropy(logits: Array, labels: Array) -> Array:
    return -take_along_axis(
        log_softmax(logits, axis=-1), expand_dims(labels, axis=-1), axis=-1
    ).mean()


# get SGD optimizer with warmup cosine learning rate scheduling
def get_optimizer_scheduler(
    opt_name: str,
    model: Module,
    rngkey: Array,
    lr: float,
    ess: float,
    momentum: float,
    momentum_hess: float | None,
    prior_prec: float,
    init_std: float,
    mc_samples: int,
    warmup_epochs: int,
    total_epochs: int,
) -> tuple[VIOptimizer, Scheduler]:
    # define VI optimizer
    if opt_name == "bbb" or momentum_hess is None:
        optimizer = vi_optimizers[opt_name](
            model.trainable_parameters(),
            rngkey=rngkey,
            lr=lr,
            ess=ess,
            momentum=momentum,
            prior_prec=prior_prec,
            init_std=init_std,
            mc_samples=mc_samples,
        )
    else:
        optimizer = vi_optimizers[opt_name](
            model.trainable_parameters(),
            rngkey=rngkey,
            lr=lr,
            ess=ess,
            momentum=momentum,
            momentum_hess=momentum_hess,
            prior_prec=prior_prec,
            init_std=init_std,
            mc_samples=mc_samples,
        )
    scheduler = Scheduler(
        optimizer,
        lr=make_warmup_cosine_schedule(lr, warmup_epochs, total_epochs),
    )
    return optimizer, scheduler


def train_transform(img, key):
    transform = ComposeTransform(
        [
            ImageToArray(),
            Normalize(*CIFAR10_MEAN_STD),
            RandomHorizontalFlip(),
            Pad(4),
            RandomCrop(32),
        ]
    )
    c = transform.rngkey_count
    new_keys = jrsplit(key, c + 1)
    output = transform(img, new_keys[1:])
    return output, new_keys[0]


def eval_transform(img):
    transform = ComposeTransform(
        [
            ImageToArray(),
            Normalize(*CIFAR10_MEAN_STD),
        ]
    )
    return transform(img, None)


def _rng_ff(objective, optimizer, model, inputs, targets, key):
    key, ffkey = jrsplit(key)
    optimizer, model, loss, logit_samples = optimizer.step(
        model, objective, inputs, targets, rngkey=ffkey
    )
    return (
        optimizer,
        model,
        loss,
        logsumexp(log_softmax(logit_samples, axis=-1), axis=0),
        key,
    )


def _deterministic_ff(objective, optimizer, model, inputs, targets, key):
    optimizer, model, loss, logit_samples = optimizer.step(
        model, objective, inputs, targets
    )
    return (
        optimizer,
        model,
        loss,
        logsumexp(log_softmax(logit_samples, axis=-1), axis=0),
        key,
    )


def train_step(model: Module, optimizer: VIOptimizer, key, data_batch):
    inputs, targets = data_batch
    inputs, key = train_transform(inputs, key)
    objective = make_supervised_objective(cross_entropy)
    if model.forward_rngkey_count > 0:
        optimizer, model, loss, logits, key = _rng_ff(
            objective, optimizer, model, inputs, targets, key
        )
    else:
        optimizer, model, loss, logits, key = _deterministic_ff(
            objective, optimizer, model, inputs, targets, key
        )
    return model, optimizer, key, loss, logits, targets


def eval_step(model: Module, data_batch):
    inputs, targets = data_batch
    inputs = eval_transform(inputs)
    model, logits = model(inputs)
    loss = cross_entropy(logits, targets)
    return model, loss, logits, targets


def train_epoch(
    trainset,
    evalset,
    trainkey,
    model: Module,
    optimizer: VIOptimizer,
    scheduler: Scheduler,
    logger,
    train_mon,
    eval_mon,
    epoch,
):
    # train epoch
    model = model.update_mode("train")
    key, skey = jrsplit(trainkey)
    tracker = MetricTracker().start("loss", "acc", "brier")
    ece_tracker = ECETracker()

    def _train_step(data_batch, state):
        model, optimizer, logger, tracker, ece_tracker, key, step = state
        model, optimizer, key, loss, logits, targets = train_step(
            model, optimizer, key, data_batch.values()
        )
        train_acc = acc(logits, targets)
        train_brier = brier_score(logits, targets)
        tracker = tracker.log(loss=loss, acc=train_acc, brier=train_brier)
        ece_tracker = ece_tracker.log(logits, targets)
        cond(
            step % 100 == 0,
            partial(
                logger.info,
                "- step %s: loss=%.4f, acc=%.4f, brier=%.4f, ece=%.4f",
                step,
                tracker["loss"],
                tracker["acc"],
                tracker["brier"],
                ece_tracker.ece,
            ),
            lambda: logger,
        )

        return model, optimizer, logger, tracker, ece_tracker, key, step + 1

    start_time = get_time()
    scheduler, optimizer = scheduler.schedule(optimizer)
    model, optimizer, logger, tracker, ece_tracker, key, _ = foreach_loop(
        trainset,
        _train_step,
        (model, optimizer, logger, tracker, ece_tracker, key, 0),
        shuffle_key=skey,
    )
    train_mon = train_mon.log(
        epoch=epoch,
        loss=tracker["loss"],
        acc=tracker["acc"],
        brier=tracker["brier"],
        ece=ece_tracker.ece,
    )
    logger = logger.info(
        "Train: loss=%.4f, acc=%.4f, brier=%.4f, ece=%.4f, time: %.1f sec",
        tracker["loss"],
        tracker["acc"],
        tracker["brier"],
        ece_tracker.ece,
        get_time() - start_time,
    )

    # eval epoch

    model = model.update_mode("eval")
    tracker = MetricTracker().start("loss", "acc", "brier")
    ece_tracker = ECETracker()

    def _eval_step(data_batch, state):
        model, tracker, ece_tracker = state
        model, loss, logits, targets = eval_step(model, data_batch.values())
        eval_acc = acc(logits, targets)
        eval_brier = brier_score(logits, targets)
        tracker = tracker.log(loss=loss, acc=eval_acc, brier=eval_brier)
        ece_tracker = ece_tracker.log(logits, targets)
        return model, tracker, ece_tracker

    start_time = get_time()
    model, tracker, ece_tracker = foreach_loop(
        evalset, _eval_step, (model, tracker, ece_tracker)
    )
    eval_mon = eval_mon.log(
        epoch=epoch,
        loss=tracker["loss"],
        acc=tracker["acc"],
        brier=tracker["brier"],
        ece=ece_tracker.ece,
    )
    logger = logger.info(
        "Eval : loss=%.4f, acc=%.4f, brier=%.4f, ece=%.4f, time: %.1f sec",
        tracker["loss"],
        tracker["acc"],
        tracker["brier"],
        ece_tracker.ece,
        get_time() - start_time,
    )

    model = model.update_mode("train")
    return key, model, optimizer, scheduler, logger, train_mon, eval_mon


def train(
    trainset,
    evalset,
    epochs: int,
    trainkey,
    model: Module,
    optimizer: VIOptimizer,
    scheduler: Scheduler,
    logger,
    train_mon,
    eval_mon,
):

    def _epoch_fn(epoch, state):
        trainkey, model, optimizer, scheduler, logger, train_mon, eval_mon = (
            state
        )
        logger = logger.info("### epoch %d ###", epoch)
        trainkey, model, optimizer, scheduler, logger, train_mon, eval_mon = (
            train_epoch(
                trainset,
                evalset,
                trainkey,
                model,
                optimizer,
                scheduler,
                logger,
                train_mon,
                eval_mon,
                epoch,
            )
        )
        return (
            trainkey,
            model,
            optimizer,
            scheduler,
            logger,
            train_mon,
            eval_mon,
        )

    logger = logger.info("*** training starts ***")

    _, model, optimizer, scheduler, logger, train_mon, eval_mon = fori_loop(
        0,
        epochs,
        _epoch_fn,
        (trainkey, model, optimizer, scheduler, logger, train_mon, eval_mon),
    )

    logger = logger.info("*** training complete ***")

    return model, optimizer, scheduler, logger, train_mon, eval_mon


def test(
    testset,
    bayes_samples: int,
    model: Module,
    optimizer: VIOptimizer,
    logger: Logger,
    bayes_sample_batchsize: int = 1,
    save_folder: str | None = None,
):
    model = model.update_mode("eval")
    opt_name = get_opt_name(optimizer)

    logger = logger.info("### MAP test started ###")
    tracker = MetricTracker().start("nll", "acc", "brier")
    ece_tracker = ECETracker()

    def _map_step(data_batch, state):
        model, tracker, ece_tracker = state
        model, loss, logits, targets = eval_step(model, data_batch.values())
        tracker = tracker.log(
            nll=loss,
            acc=acc(logits, targets),
            brier=brier_score(logits, targets),
        )
        ece_tracker = ece_tracker.log(logits, targets)
        return model, tracker, ece_tracker

    start_time = get_time()
    model, tracker, ece_tracker = foreach_loop(
        testset, _map_step, (model, tracker, ece_tracker)
    )
    map_time = get_time()
    logger.info(
        "MAP Test:   nll=%.4f, acc=%.4f, brier=%.4f, ece=%.4f, time: %.1f s",
        tracker["nll"],
        tracker["acc"],
        tracker["brier"],
        ece_tracker.ece,
        map_time - start_time,
    )
    if save_folder:
        ece_tracker.plot_reliability_diagram(
            saveas=opjoin(save_folder, f"reliability_{opt_name}_map.pdf")
        )

    logger = logger.info("### Bayes test started ###")
    tracker = MetricTracker().start("nll", "acc", "brier")
    ece_tracker = ECETracker()

    def _bayes_step(data_batch, state):
        model, optimizer, tracker, ece_tracker = state
        keys = jrsplit(optimizer.rngkey, bayes_samples + 1)
        optimizer = optimizer.update(rngkey=keys[0])
        old_mean = model.trainable_parameters()

        def _sample_predict(key):
            psample = optimizer.sample_parameters(old_mean, key)
            msample = model.load_states(psample)
            _, _, logits, _ = eval_step(msample, data_batch.values())
            return logits

        logit_samples = map(
            _sample_predict, keys[1:], batch_size=bayes_sample_batchsize
        )
        bayes_logits = logsumexp(log_softmax(logit_samples, axis=-1), axis=0)
        _, targets = tuple(data_batch.values())
        tracker = tracker.log(
            nll=cross_entropy(bayes_logits, targets),
            acc=acc(bayes_logits, targets),
            brier=brier_score(bayes_logits, targets),
        )
        ece_tracker = ece_tracker.log(bayes_logits, targets)

        return model, optimizer, tracker, ece_tracker

    model, optimizer, tracker, ece_tracker = foreach_loop(
        testset, _bayes_step, (model, optimizer, tracker, ece_tracker)
    )
    logger.info(
        "Bayes Test: nll=%.4f, acc=%.4f, brier=%.4f, ece=%.4f, time: %.1f s",
        tracker["nll"],
        tracker["acc"],
        tracker["brier"],
        ece_tracker.ece,
        get_time() - map_time,
    )
    if save_folder:
        ece_tracker.plot_reliability_diagram(
            saveas=opjoin(save_folder, f"reliability_{opt_name}_bayes.pdf")
        )

    logger = logger.info("### all tests complete ###")
    model = model.update_mode("train")
    return model, optimizer, logger


def get_param_count(model: Module) -> int:
    from jax.tree import flatten as tree_flatten

    return sum(p.size for p in tree_flatten(model.trainable_parameters())[0])


if __name__ == "__main__":
    # allow 1G max mem usage to use CIFAR-10 dataset in memory for speed up
    datasets.config.IN_MEMORY_MAX_SIZE = 1 * (10**9)

    # parse modelname from terminal
    if len(sys.argv) not in (3, 4):
        print(
            "expected usage: python run.py <opt_name> <init_lr> "
            "[<momentum_hess>]"
        )
        exit(1)
    opt_name, init_lr = sys.argv[1], float(sys.argv[2])
    momentum_hess = None if len(sys.argv) == 3 else float(sys.argv[3])
    if opt_name not in vi_optimizers:
        print(f"unrecognized optimizer name: {opt_name}")
        print("expected usage: python run.py <opt_name>")
        print(f"possible optimizer choices:")
        for k in vi_optimizers.keys():
            print(f"- {k}")
        exit(2)

    # other configs
    random_seed = 42
    epochs = 200
    warmup_epochs = 5
    batch_size = 50
    train_ratio = 0.9
    # hess_init = 0.5  # DEBUG test increase hess_init
    hess_init = 1.0
    momentum = 0.9
    weight_decay = 2e-4
    mc_samples = 1
    ess = int(50000 * train_ratio)
    test_mc_samples = 64
    save_folder = f"train_{opt_name}"

    os.makedirs(save_folder, exist_ok=True)
    ckpt_path = opjoin(save_folder, f"{opt_name}_trained.tar")

    # start plot visualization process
    with run_with_plot(save_folder, block=True):

        # init logger and monitors
        logger = Logger(save_to=opjoin(save_folder, f"log_{opt_name}.txt"))
        logger.info(f"Optimizer: {opt_name}, init lr: {init_lr}")
        train_mon = MultiMonitor(
            csv=HostCSVMonitor(
                opjoin(save_folder, f"results_train_{opt_name}.csv"),
                append=False,
            ).jaxify(),
            plot=HostPlotMonitor(
                f"plot_train_{opt_name}", step_name="epoch"
            ).jaxify(),
        )
        eval_mon = MultiMonitor(
            csv=HostCSVMonitor(
                opjoin(save_folder, f"results_eval_{opt_name}.csv"),
                append=False,
            ).jaxify(),
            plot=HostPlotMonitor(
                f"plot_eval_{opt_name}", step_name="epoch"
            ).jaxify(),
        )

        trainset, testset = get_cifar10()
        logger.info(f"num samples: train {len(trainset)}, test {len(testset)}")

        trainset = trainset.batchify(batch_size)
        testset = testset.batchify(batch_size)
        n_train = int(len(trainset) * train_ratio)
        trainset, evalset = trainset.split(n_train)
        logger.info(
            f"num batches after batching {batch_size} and split {train_ratio}:"
            f" train {len(trainset)}, eval {len(evalset)}, test {len(testset)}"
        )

        modelkey, optkey, trainkey = jrsplit(jrkey(random_seed), 3)

        model, out_shape = resnet20().reset((-1, 3, 32, 32), modelkey)
        logger.info(
            f"Using model resnet20 with {get_param_count(model) / 1e3:.1f}K "
            f"parameters."
        )

        optimizer, scheduler = get_optimizer_scheduler(
            opt_name,
            model,
            optkey,
            init_lr,
            ess=ess,
            momentum=momentum,
            momentum_hess=momentum_hess,
            prior_prec=ess * weight_decay,
            init_std=((hess_init + weight_decay) * ess) ** -0.5,
            mc_samples=mc_samples,
            warmup_epochs=warmup_epochs,
            total_epochs=epochs,
        )

        train_mon = train_mon.start("epoch", "loss", "acc", "brier", "ece")
        eval_mon = eval_mon.start("epoch", "loss", "acc", "brier", "ece")
        # # run the training process
        # model, optimizer, logger = train(
        #     trainset, evalset, epochs, trainkey, model, optimizer, scheduler,
        #     logger, train_mon, eval_mon
        # )
        # can be globally jitted
        model, optimizer, scheduler, logger, train_mon, eval_mon = jit(
            partial(train, trainset, evalset, epochs)
        )(
            trainkey,
            model,
            optimizer,
            scheduler,
            logger,
            train_mon,
            eval_mon,
        )

        train_mon.stop()
        eval_mon.stop()

        # save trained model
        save((model, optimizer), ckpt_path, overwrites=True)
        logger = logger.info(f"checkpoint saved to {ckpt_path}")

        # run final test on saved model
        _, _, logger = jit(
            lambda m, o, lg: test(
                testset, test_mc_samples, m, o, lg, save_folder=save_folder
            )
        )(*load(ckpt_path), logger)
        logger.close()
