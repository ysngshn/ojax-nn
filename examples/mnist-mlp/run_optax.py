"""MNIST classification with multi-layer perceptron"""

# noinspection PyUnresolvedReferences
from functools import partial
from jax import jit, Array
from jax.lax import fori_loop, cond
from jax.random import split as jrsplit, key as jrkey
from optax import (
    softmax_cross_entropy_with_integer_labels,
    warmup_cosine_decay_schedule,
    linear_schedule,
)
from ojnn import Module, Sequential
from ojnn import Flatten2d, Dense, ReLU
from ojnn.optim import (
    Optimizer,
    SGD,
    inject_hyperparams,
    make_supervised_objective,
)
from ojnn.io import foreach_loop, Logger, MetricTracker, load, save
from data_utils import MNIST, ComposeTransform, ImageToArray, Normalize
from optim_utils import acc


def cross_entropy(logits, labels):
    return softmax_cross_entropy_with_integer_labels(logits, labels).mean()


# get a multi-layer perceptron with 2 dense hidden layers
def get_mlp(depth: int = 3, width: int = 1000) -> Module:
    assert depth >= 2
    assert width > 0
    layers = (
        [Flatten2d()]
        + sum(([Dense(width), ReLU()] for _ in range(depth - 1)), start=[])
        + [Dense(10)]
    )
    return Sequential(*layers)


# get SGD optimizer with lr and momentum schedules


def get_optimizer(
    model: Module,
    lr: float,
    momentum: float,
    weight_decay: float,
    epochs: int,
    batchs_per_epoch: int,
    warmup_epoch: int = 1,
) -> SGD:
    assert epochs >= warmup_epoch
    # define SGD optimizer
    lr_schedule = warmup_cosine_decay_schedule(
        init_value=1e-2 * lr,
        peak_value=lr,
        warmup_steps=warmup_epoch * batchs_per_epoch,
        decay_steps=epochs * batchs_per_epoch,
    )
    momentum_schedule = linear_schedule(
        momentum,
        0.1 * momentum,
        transition_steps=(epochs - warmup_epoch) * batchs_per_epoch,
        transition_begin=warmup_epoch * batchs_per_epoch,
    )
    sgd = inject_hyperparams(SGD)(
        model.trainable_parameters(),
        learning_rate=lr_schedule,
        momentum=momentum_schedule,
        weight_decay=weight_decay,
    )
    return sgd


def standardize_inputs(inputs: Array) -> Array:
    return ComposeTransform(
        [ImageToArray(), Normalize(MNIST.MEAN, MNIST.STD)]
    )(inputs, None)


def train_step(model: Module, optimizer: Optimizer, data_batch):
    inputs, targets = data_batch
    inputs = standardize_inputs(inputs)
    objective = make_supervised_objective(cross_entropy)
    optimizer, model, loss, logits = optimizer.step(
        model, objective, inputs, targets
    )
    return model, optimizer, loss, logits, targets


def eval_step(model: Module, data_batch):
    inputs, targets = data_batch
    inputs = standardize_inputs(inputs)
    model, logits = model(inputs)
    loss = cross_entropy(logits, targets)
    return model, loss, logits, targets


def train_epoch(
    trainset,
    evalset,
    trainkey,
    model: Module,
    optimizer: Optimizer,
    tracker,
    logger,
):
    # train epoch
    newkey, skey = jrsplit(trainkey)
    tracker = tracker.start("loss", "acc", "nll")

    def _train_step(data_batch, state):
        model, optimizer, logger, tracker, step = state
        model, optimizer, loss, logits, targets = train_step(
            model, optimizer, data_batch.values()
        )
        tracker = tracker.log(
            loss=loss,
            acc=acc(logits, targets),
            nll=cross_entropy(logits, targets),
        )
        cond(
            step % 100 == 0,
            partial(
                logger.info,
                "- step %s: loss=%.4f, acc=%.4f, nll=%.4f",
                step,
                tracker["loss"],
                tracker["acc"],
                tracker["nll"],
            ),
            lambda: logger,
        )

        return model, optimizer, logger, tracker, step + 1

    model, optimizer, logger, tracker, _ = foreach_loop(
        trainset,
        _train_step,
        (model, optimizer, logger, tracker, 0),
        shuffle_key=skey,
    )
    logger = logger.info(
        "Train: loss=%.4f, acc=%.4f, nll=%.4f",
        tracker["loss"],
        tracker["acc"],
        tracker["nll"],
    )
    tracker = tracker.stop()

    # eval epoch

    model = model.update_mode("eval")
    tracker = tracker.start("loss", "acc", "nll")

    def _eval_step(data_batch, state):
        model, tracker = state
        model, loss, logits, targets = eval_step(model, data_batch.values())
        tracker = tracker.log(
            loss=loss,
            acc=acc(logits, targets),
            nll=cross_entropy(logits, targets),
        )
        return model, tracker

    model, tracker = foreach_loop(evalset, _eval_step, (model, tracker))
    logger = logger.info(
        "Eval : loss=%.4f, acc=%.4f, nll=%.4f",
        tracker["loss"],
        tracker["acc"],
        tracker["nll"],
    )
    tracker = tracker.stop()
    model = model.update_mode("train")

    return newkey, model, optimizer, tracker, logger


def train(
    trainset,
    evalset,
    trainkey,
    epochs: int,
    model: Module,
    optimizer: Optimizer,
    tracker,
    logger,
):

    def _epoch_fn(epoch, state):
        trainkey, model, optimizer, tracker, logger = state
        logger = logger.info("### epoch %d ###", epoch)
        trainkey, model, optimizer, tracker, logger = train_epoch(
            trainset,
            evalset,
            trainkey,
            model,
            optimizer,
            tracker,
            logger,
        )
        return trainkey, model, optimizer, tracker, logger

    logger = logger.info("*** training starts ***")

    _, model, optimizer, tracker, logger = fori_loop(
        0,
        epochs,
        _epoch_fn,
        (trainkey, model, optimizer, tracker, logger),
    )

    logger = logger.info("*** training complete ***")

    return model, optimizer, tracker, logger


def test(testset, model: Module, tracker, logger):
    model = model.update_mode("eval")
    tracker = tracker.start("loss", "acc", "nll")

    logger = logger.info("### test started ###")

    def _process_batch(data_batch, state):
        model, tracker = state
        model, loss, logits, targets = eval_step(model, data_batch.values())
        tracker = tracker.log(
            loss=loss,
            acc=acc(logits, targets),
            nll=cross_entropy(logits, targets),
        )
        return model, tracker

    model, tracker = foreach_loop(testset, _process_batch, (model, tracker))
    logger.info(
        "Eval : loss=%.4f, acc=%.4f, nll=%.4f",
        tracker["loss"],
        tracker["acc"],
        tracker["nll"],
    )
    tracker = tracker.stop()

    model = model.update_mode("train")
    logger = logger.info("### test complete ###")

    return model, tracker, logger


if __name__ == "__main__":
    # configs
    seed = 42
    batch_size = 100
    epochs = 20
    init_lr = 0.1
    momentum = 0.9
    weight_decay = 0.001
    train_ratio = 5 / 6
    model_path = "./model_optax.tar"

    modelkey, trainkey = jrsplit(jrkey(seed))
    tracker = MetricTracker()
    logger = Logger()

    # define and initialize the MLPmodel
    model = get_mlp()
    model = model.init((1, 28, 28), rngkey=modelkey)

    # get and split tran/eval/test datasets
    dataset = MNIST("../data", train=True, download=True)
    n_train = int(train_ratio * len(dataset))
    trainset, evalset = dataset.split(n_train)
    testset = MNIST("../data", train=False, download=True)
    # inspect element
    img = trainset[0]["image"]
    logger = logger.info(
        "Input shape %s, max %s, min %s.",
        [int(s) for s in img.shape],
        img.max(),
        img.min(),
    )
    # batchify datasets
    trainset = trainset.batchify(batch_size)
    evalset = evalset.batchify(batch_size)
    testset = testset.batchify(batch_size)

    # get SGD optimizer with scheduling
    optimizer = get_optimizer(
        model,
        init_lr,
        momentum,
        weight_decay,
        epochs,
        len(trainset),
        warmup_epoch=1,
    )

    # jitted-version of the whole training process
    model, optimizer, tracker, logger = jit(partial(train, trainset, evalset))(
        trainkey, epochs, model, optimizer, tracker, logger
    )

    # save trained model
    save(model, model_path, overwrites=True)
    logger = logger.info(f"model saved to {model_path}")

    # run final test on saved model
    model, tracker, logger = test(testset, load(model_path), tracker, logger)
    logger.close()
