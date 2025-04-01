"""run CIFAR-10 classification with ResNet, monitor training with matplotlib"""

from functools import partial
from time import perf_counter
import sys
import os
# disable gpu memory pre-allocation
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# deterministic ops, slower. cf.: https://github.com/google/jax/issues/13672
# os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true"
# os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
# os.environ["TF_DETERMINISTIC_ops"] = "1"
from jax import jit, Array, ShapeDtypeStruct
from jax.dtypes import canonicalize_dtype
from jax.numpy import argmax, take_along_axis, expand_dims
from jax.nn import log_softmax
from jax.lax import fori_loop, cond
from jax.random import split as jrsplit, key as jrkey
from optax.schedules import cosine_decay_schedule
import datasets
from ojnn import Module
from ojnn.optim import SGD, make_supervised_objective
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
from vgg import vgg11, vgg13, vgg16, vgg19
from resnet import resnet20, resnet164, resnet110, resnet1001
from densenet import (
    densenet40k12,
    densenet100k12,
    densenet100k24,
    densenetbc100k12,
    densenetbc250k24,
    densenetbc190k40,
)


def get_time() -> Array:
    return host_callback(
        perf_counter,
        debug_impl=False,
        result_shape_dtypes=ShapeDtypeStruct(
            shape=(), dtype=canonicalize_dtype(float)
        ),
    )()


models = {
    "vgg11": vgg11,
    "vgg13": vgg13,
    "vgg16": vgg16,
    "vgg19": vgg19,
    "resnet20": resnet20,
    "resnet110": resnet110,
    "resnet164": resnet164,
    "resnet1001": resnet1001,
    "densenet40k12": densenet40k12,
    "densenet100k12": densenet100k12,
    "densenet100k24": densenet100k24,
    "densenetbc100k12": densenetbc100k12,
    "densenetbc250k24": densenetbc250k24,
    "densenetbc190k40": densenetbc190k40,
}


def acc(output: Array, target: Array) -> Array:
    preds = argmax(output, axis=-1)
    return (preds == target).mean()


def cross_entropy(logits: Array, labels: Array) -> Array:
    return -take_along_axis(
        log_softmax(logits, axis=-1), expand_dims(labels, axis=-1), axis=-1
    ).mean()


# get SGD optimizer with warmup cosine learning rate scheduling
def get_optimizer(
    model: Module,
    lr: float,
    momentum: float,
    weight_decay: float,
    epochs: int,
    batches_per_epoch: int,
) -> SGD:
    total_steps = epochs * batches_per_epoch
    # define SGD optimizer
    sgd = SGD(
        model.trainable_parameters(),
        learning_rate=cosine_decay_schedule(lr, total_steps),
        momentum=momentum,
        weight_decay=weight_decay,
    )
    return sgd


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
    optimizer, model, loss, logits = optimizer.step(
        model, objective, inputs, targets, rngkey=ffkey
    )
    return optimizer, model, loss, logits, key


def _deterministic_ff(objective, optimizer, model, inputs, targets, key):
    optimizer, model, loss, logits = optimizer.step(
        model, objective, inputs, targets
    )
    return optimizer, model, loss, logits, key


def train_step(model: Module, optimizer: SGD, key, data_batch):
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
    optimizer: SGD,
    logger,
    train_mon,
    eval_mon,
    epoch,
):
    # train epoch
    model = model.update_mode("train")
    key, skey = jrsplit(trainkey)
    tracker = MetricTracker().start("loss", "acc")

    def _train_step(data_batch, state):
        model, optimizer, logger, tracker, key, step = state
        model, optimizer, key, loss, logits, targets = train_step(
            model, optimizer, key, data_batch.values()
        )
        train_acc = acc(logits, targets)
        tracker = tracker.log(loss=loss, acc=train_acc)
        cond(
            step % 100 == 0,
            partial(
                logger.info,
                "- step %s: loss=%.4f, acc=%.4f",
                step,
                tracker["loss"],
                tracker["acc"],
            ),
            lambda: logger,
        )

        return model, optimizer, logger, tracker, key, step + 1

    start_time = get_time()
    model, optimizer, logger, tracker, key, _ = foreach_loop(
        trainset,
        _train_step,
        (model, optimizer, logger, tracker, key, 0),
        shuffle_key=skey,
    )
    logger = logger.info(
        "Train: loss=%.4f, acc=%.4f, time: %.1f sec",
        tracker["loss"],
        tracker["acc"],
        get_time() - start_time,
    )
    train_mon = train_mon.log(
        epoch=epoch, loss=tracker["loss"], acc=tracker["acc"]
    )

    # eval epoch

    model = model.update_mode("eval")
    tracker = MetricTracker().start("loss", "acc")

    def _eval_step(data_batch, state):
        model, tracker = state
        model, loss, logits, targets = eval_step(model, data_batch.values())
        eval_acc = acc(logits, targets)
        tracker = tracker.log(loss=loss, acc=eval_acc)
        return model, tracker

    start_time = get_time()
    model, tracker = foreach_loop(evalset, _eval_step, (model, tracker))
    logger = logger.info(
        "Eval : loss=%.4f, acc=%.4f, time: %.1f sec",
        tracker["loss"],
        tracker["acc"],
        get_time() - start_time,
    )
    eval_mon = eval_mon.log(
        epoch=epoch, loss=tracker["loss"], acc=tracker["acc"]
    )

    model = model.update_mode("train")
    return key, model, optimizer, logger, train_mon, eval_mon


def train(
    trainset,
    evalset,
    trainkey,
    epochs: int,
    model: Module,
    optimizer: SGD,
    logger,
    train_mon,
    eval_mon,
):

    def _epoch_fn(epoch, state):
        trainkey, model, optimizer, logger, train_mon, eval_mon = state
        logger = logger.info("### epoch %d ###", epoch)
        trainkey, model, optimizer, logger, train_mon, eval_mon = train_epoch(
            trainset,
            evalset,
            trainkey,
            model,
            optimizer,
            logger,
            train_mon,
            eval_mon,
            epoch,
        )
        return trainkey, model, optimizer, logger, train_mon, eval_mon

    logger = logger.info("*** training starts ***")

    _, model, optimizer, logger, train_mon, eval_mon = fori_loop(
        0,
        epochs,
        _epoch_fn,
        (trainkey, model, optimizer, logger, train_mon, eval_mon),
    )

    logger = logger.info("*** training complete ***")

    return model, optimizer, logger, train_mon, eval_mon


def test(testset, model: Module, logger):
    model = model.update_mode("eval")
    tracker = MetricTracker().start("loss", "acc")

    logger = logger.info("### test started ###")

    def _process_batch(data_batch, state):
        model, tracker = state
        model, loss, logits, targets = eval_step(model, data_batch.values())
        tracker = tracker.log(loss=loss, acc=acc(logits, targets))
        return model, tracker

    start_time = get_time()
    model, tracker = foreach_loop(testset, _process_batch, (model, tracker))
    logger.info(
        "Test : loss=%.4f, acc=%.4f, time: %.1f sec",
        tracker["loss"],
        tracker["acc"],
        get_time() - start_time,
    )

    model = model.update_mode("train")
    logger = logger.info("### test complete ###")

    return model, logger


def get_param_count(model: Module) -> int:
    from jax.tree import flatten as tree_flatten

    return sum(p.size for p in tree_flatten(model.trainable_parameters())[0])


if __name__ == "__main__":
    # allow 1G max mem usage to use CIFAR-10 dataset in memory for speed up
    datasets.config.IN_MEMORY_MAX_SIZE = 1 * (10**9)
    # parse modelname from terminal
    if len(sys.argv) != 3:
        print("expected usage: python run.py <model_name> <init_lr>")
        exit(1)
    model_name = sys.argv[1]
    if model_name not in models:
        print(f"unrecognized model name: {model_name}")
        print("expected usage: python run.py <model_name> <init_lr>")
        print(f"possible model choices:")
        for k in models.keys():
            print(f"- {k}")
        exit(2)
    model_path = f"./{model_name}_trained.tar"

    # other configs
    random_seed = 42
    epochs = 200
    batch_size = 100
    train_ratio = 0.9
    init_lr = float(sys.argv[2])
    momentum = 0.9
    weight_decay = 5e-4

    # start plot visualization process
    with run_with_plot(".", block=True):

        # init logger and monitors
        logger = Logger(save_to=f"./log_{model_name}.txt")
        logger.info(f"Model: {model_name}, init lr: {init_lr}")
        train_mon = MultiMonitor(
            csv=HostCSVMonitor(
                f"results_train_{model_name}.csv", append=False
            ).jaxify(),
            plot=HostPlotMonitor(
                f"plot_train_{model_name}", step_name="epoch"
            ).jaxify(),
        )
        eval_mon = MultiMonitor(
            csv=HostCSVMonitor(
                f"results_eval_{model_name}.csv", append=False
            ).jaxify(),
            plot=HostPlotMonitor(
                f"plot_eval_{model_name}", step_name="epoch"
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

        modelkey, trainkey = jrsplit(jrkey(random_seed))

        model, out_shape = models[model_name]().reset(
            (-1, 3, 32, 32), modelkey
        )
        logger.info(
            f"Using model {model_name} with "
            f"{get_param_count(model) / 1e6:.1f}M parameters."
        )

        optimizer = get_optimizer(
            model,
            init_lr,
            momentum=momentum,
            weight_decay=weight_decay,
            epochs=epochs,
            batches_per_epoch=len(trainset),
        )

        train_mon = train_mon.start("epoch", "loss", "acc")
        eval_mon = eval_mon.start("epoch", "loss", "acc")
        # # run the training process
        # model, optimizer, logger = train(
        #     trainset, evalset, trainkey, epochs, model, optimizer, logger,
        #     train_mon, eval_mon
        # )
        # can be globally jitted
        model, optimizer, logger, train_mon, eval_mon = jit(
            partial(train, trainset, evalset)
        )(trainkey, epochs, model, optimizer, logger, train_mon, eval_mon)

        # save trained model
        save(model, model_path, overwrites=True)
        logger = logger.info(f"model saved to {model_path}")

        # run final test on saved model
        model, logger = test(testset, load(model_path), logger)

        train_mon.stop()
        eval_mon.stop()
        logger.close()
