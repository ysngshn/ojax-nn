from typing_extensions import Self
from numpy import ndarray
from matplotlib import pyplot as plt
from jax import Array
from jax.numpy import (
    zeros,
    mean,
    square,
    sum,
    max,
    argmax,
    linspace,
    histogram,
    abs,
)
from jax.nn import one_hot, softmax
from ojax import OTree
from ojnn.io import host_callback


def brier_score(logits: Array, targets: Array, axis: int = -1) -> Array:
    return mean(
        sum(
            square(
                softmax(logits, axis=axis)
                - one_hot(targets, num_classes=logits.shape[-1], axis=axis)
            ),
            axis=axis,
        )
    )


class ECETracker(OTree):
    num_bin: int
    bin_counts: Array
    bin_confs: Array
    bin_corrects: Array

    def __init__(self, num_bin: int = 20):
        self.assign_(
            num_bin=num_bin,
            bin_counts=zeros([num_bin]),
            bin_confs=zeros([num_bin]),
            bin_corrects=zeros([num_bin]),
        )

    @property
    def ece(self) -> Array:
        return sum(abs(self.bin_confs - self.bin_corrects)) / sum(
            self.bin_counts
        )

    def log(self: Self, logits: Array, targets: Array, axis: int = -1) -> Self:
        confs = max(softmax(logits, axis=axis), axis=axis)
        corrects = (argmax(logits, axis=axis) == targets).astype(confs.dtype)
        bin_edges = linspace(0, 1, self.num_bin + 1, endpoint=True)
        new_counts = histogram(confs, bins=bin_edges)[0]
        new_confs = histogram(confs, bins=bin_edges, weights=confs)[0]
        new_corrects = histogram(confs, bins=bin_edges, weights=corrects)[0]
        return self.update(
            bin_counts=self.bin_counts + new_counts,
            bin_confs=self.bin_confs + new_confs,
            bin_corrects=self.bin_corrects + new_corrects,
        )

    def reset(self: Self) -> Self:
        num_bin = self.num_bin
        return self.update(
            bin_counts=zeros([num_bin]),
            bin_confs=zeros([num_bin]),
            bin_corrects=zeros([num_bin]),
        )

    def plot_reliability_diagram(
        self, displays: bool = False, saveas: str | None = None
    ) -> None:
        host_callback(
            lambda bins: host_plot_reliability_diagram(bins, displays, saveas),
        )((self.bin_counts, self.bin_corrects, self.bin_confs))


def host_plot_reliability_diagram(
    bins: tuple[ndarray, ndarray, ndarray],
    displays: bool = False,
    saveas: str | None = None,
) -> None:
    nbin = len(bins[0])
    binvals = [float(i) / nbin for i in range(nbin + 1)]
    accconfs = [
        (float(corr) / bc, cconf / bc) if bc > 0 else (0.0, 0.0)
        for bc, corr, cconf in zip(*bins)
    ]
    weights = (
        [acc for acc, _ in accconfs],
        [conf - acc for acc, conf in accconfs],
    )
    fig = plt.figure(figsize=(5, 5))
    a1 = fig.add_subplot(111)
    a2 = a1.twinx()
    a1.set_xlim(0, 1)
    a1.set_ylim(0, 1)
    a1.set_xlabel("Confidence")
    a1.set_ylabel("Accuracy")
    _, _, ps = a1.hist(
        [binvals[:-1], binvals[:-1]],
        binvals,
        weights=weights,
        color=[(0, 0, 1, 1), (1, 0, 0, 0.5)],
        label=("Empirical", "Gap"),
        stacked=True,
    )
    total = sum(bins[0])
    freqs = [float(c) / total for c in bins[0]]
    a2.set_ylim(0, 1)
    a2.set_ylabel("Frequency")
    a2.hist(
        [binvals[:-1]],
        binvals,
        weights=[freqs],
        rwidth=0.3,
        color=[(0.5, 0.5, 0.5, 1)],
        label=("Frequency",),
    )
    fig.set_layout_engine("constrained")
    fig.legend(loc="upper left", bbox_to_anchor=a1.get_position())

    hatches = ["", "/"]
    linewidths = [1.0, 1.0]
    edgecolors = [(0, 0, 0.5, 1), (1, 0, 0, 1)]
    for pset, h, lw, ec in zip(ps, hatches, linewidths, edgecolors):
        for patch in pset.patches:
            patch.set_hatch(h)
            patch.set_lw(lw)
            patch.set_edgecolor(ec)
    if saveas:
        fig.savefig(saveas)
    if displays:
        fig.show()
    plt.close(fig)
