from functools import partial
from time import perf_counter
import sys
import os

# disable gpu memory pre-allocation
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# deterministic ops, slower. cf.: https://github.com/google/jax/issues/13672
os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
os.environ["TF_DETERMINISTIC_ops"] = "1"
from jax import Array, jit, ShapeDtypeStruct
from jax.numpy import argmax, take_along_axis, expand_dims, isscalar
from jax.dtypes import canonicalize_dtype
from jax.random import split as jrsplit, key as jrkey
from jax.lax import cond, fori_loop
from jax.nn import log_sigmoid, log_softmax
import datasets
from ojnn import Module
from ojnn.optim import AdamW, make_supervised_objective
from ojnn.io import (
    save,
    load,
    host_callback,
    foreach_loop,
    run_with_plot,
    Logger,
    MultiMonitor,
    HostCSVMonitor,
    HostPlotMonitor,
    MetricTracker,
)
from models import (
    TextConvNet,
    RNNNet,
    LSTMNet,
    GRUNet,
    BiRNNNet,
    BiLSTMNet,
    BiGRUNet,
    IndRNNNet,
    TransformerNet,
)
from data_utils import get_imdb


def get_time() -> Array:
    return host_callback(
        perf_counter,
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


def binary_acc(logits: Array, target: Array) -> Array:
    preds = logits >= 0
    return (preds == target).mean()


def binary_cross_entropy(logits: Array, target: Array) -> Array:
    return -(
        target * log_sigmoid(logits) + (1 - target) * log_sigmoid(-logits)
    ).mean()


def get_optimizer(
    model: Module,
    lr: float,
    weight_decay: float,
) -> AdamW:
    return AdamW(
        model.trainable_parameters(),
        learning_rate=lr,
        weight_decay=weight_decay,
    )


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


def train_step(model: Module, optimizer: AdamW, key, inputs, targets):
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


def eval_step(model: Module, inputs, targets):
    model, logits = model(inputs)
    loss = cross_entropy(logits, targets)
    return model, loss, logits, targets


def train_epoch(
    trainset,
    evalset,
    trainkey,
    model: Module,
    optimizer: AdamW,
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
            model, optimizer, key, data_batch["tokens"], data_batch["label"]
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
        model, loss, logits, targets = eval_step(
            model, data_batch["tokens"], data_batch["label"]
        )
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
    optimizer: AdamW,
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
        model, loss, logits, targets = eval_step(
            model, data_batch["tokens"], data_batch["label"]
        )
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


def list_params(model: Module):
    from jax.tree import flatten as tree_flatten, unflatten as tree_unflatten
    from pprint import pp as pprint

    ps, mdef = tree_flatten(model.trainable_parameters())
    pprint(tree_unflatten(mdef, [() if isscalar(p) else p.shape for p in ps]))


def get_param_count(model: Module) -> int:
    from jax.tree import flatten as tree_flatten

    return sum(
        1 if isscalar(p) else p.size
        for p in tree_flatten(model.trainable_parameters())[0]
    )


if __name__ == "__main__":
    # allow 1GB max mem usage to use dataset in memory for speed up
    datasets.config.IN_MEMORY_MAX_SIZE = 1 * (10**9)

    # configs
    random_seed = 42
    epochs = 100
    train_ratio = 0.9
    lr = float(sys.argv[2])
    pretrained_glove = True
    vocab_size = 400001
    max_length = 500  # RNN's Acc goes down 4% when changed from 500 to 200

    # available models
    model_map = {
        "cnn-4": partial(TextConvNet, depth=4),
        "cnn-6": partial(TextConvNet, depth=6),
        "cnn-8": partial(TextConvNet, depth=8),
        "rnn-4": partial(RNNNet, depth=4),
        "rnn-6": partial(RNNNet, depth=6),
        "rnn-8": partial(RNNNet, depth=8),
        "lstm-4": partial(LSTMNet, depth=4),
        "lstm-6": partial(LSTMNet, depth=6),
        "gru-4": partial(GRUNet, depth=4),
        "gru-6": partial(GRUNet, depth=6),
        "birnn-4": partial(BiRNNNet, depth=4),
        "birnn-6": partial(BiRNNNet, depth=6),
        "bilstm-4": partial(BiLSTMNet, depth=4),
        "bigru-4": partial(BiGRUNet, depth=4),
        "indrnn-6": partial(IndRNNNet, depth=6),
        "transformer": partial(TransformerNet, n_block=4, max_len=max_length),
    }

    # parse modelname from terminal
    if len(sys.argv) != 3:
        print("expected usage: python run.py <model_name> <lr>")
        exit(1)
    model_name = sys.argv[1]
    if model_name not in model_map:
        print(f"unrecognized model name: {model_name}")
        print("expected usage: python run.py <model_name> <lr>")
        print(f"possible model choices:")
        for k in model_map.keys():
            print(f"- {k}")
        exit(2)
    model_path = f"./{model_name}_trained.tar"
    # weight_decay = 1.0
    weight_decay = 1e-8
    # weight_decay = 1e-8 if model_name == "transformer" else 0.1  # 0.1
    batch_size = 50 if model_name == "transformer" else 100

    # start plot visualization process
    with run_with_plot(".", block=True):

        # init logger and monitors
        logger = Logger(save_to=f"./log_{model_name}.txt")
        logger.info(f"Model: {model_name}, lr: {lr}, w.decay: {weight_decay}")
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

        trainset, testset, _ = get_imdb(
            vocab_size=vocab_size,
            length=max_length,
            pretrained_glove=pretrained_glove,
            shuffle_train_seed=0,
        )
        logger.info(
            f"num samples: train {len(trainset)}, test {len(testset)}; "
            f"length {max_length}"
        )

        trainset = trainset.batchify(batch_size)
        testset = testset.batchify(batch_size)
        n_train = int(len(trainset) * train_ratio)
        trainset, evalset = trainset.split(n_train)
        logger.info(
            f"num batches after batching {batch_size} and split {train_ratio}:"
            f" train {len(trainset)}, eval {len(evalset)}, test {len(testset)}"
        )

        modelkey, trainkey = jrsplit(jrkey(random_seed))

        model = model_map[model_name](
            vocab_size=vocab_size, pretrained_glove=pretrained_glove
        ).init((-1, max_length), modelkey)
        logger.info(
            f"Using model {model_name} with "
            f"{get_param_count(model) / 1e3:.1f}K parameters."
        )

        optimizer = get_optimizer(model, lr, weight_decay)

        train_mon = train_mon.start("epoch", "loss", "acc")
        eval_mon = eval_mon.start("epoch", "loss", "acc")
        # run the training process
        model, optimizer, logger, train_mon, eval_mon = train(
            trainset,
            evalset,
            trainkey,
            epochs,
            model,
            optimizer,
            logger,
            train_mon,
            eval_mon,
        )
        # # can be globally jitted
        # model, optimizer, logger, train_mon, eval_mon = jit(
        #     partial(train, trainset, evalset)
        # )(trainkey, epochs, model, optimizer, logger, train_mon, eval_mon)

        # save trained model
        save(model, model_path, overwrites=True)
        logger = logger.info(f"model saved to {model_path}")

        # run final test on saved model
        model, logger = test(testset, load(model_path), logger)

        train_mon.stop()
        eval_mon.stop()
        logger.close()
