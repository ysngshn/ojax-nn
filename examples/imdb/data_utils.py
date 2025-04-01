from collections.abc import Iterable
from pathlib import Path
import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers
from ojnn.io.from_datasets import Dataset


def _get_glove_tokenizer(vocab_size: int, save_to=None) -> Tokenizer:
    import gensim.downloader as api

    print("loading Gensim model ... ", end="")
    model = api.load("glove-wiki-gigaword-100")
    print("done.")
    # shift by one to add [unk]
    vocab_dict = {
        **{"[UNK]": 0},
        **{
            k: i + 1
            for i, k in enumerate(model.index_to_key[: vocab_size - 1])
        },
    }
    tokenizer = Tokenizer(
        models.WordLevel(vocab=vocab_dict, unk_token="[UNK]")
    )
    tokenizer.normalizer = normalizers.Sequence(
        [
            normalizers.Replace("<br />", ""),
            normalizers.NFD(),
            normalizers.Lowercase(),
            normalizers.StripAccents(),
        ]
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    if save_to:
        tokenizer.save(save_to, pretty=True)
    return tokenizer


def _build_tokenizer(
    corpus: Iterable[str], vocab_size, save_to=None
) -> Tokenizer:

    tokenizer = Tokenizer(models.WordLevel(vocab=None, unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.Sequence(
        [
            normalizers.Replace("<br />", ""),
            normalizers.NFD(),
            normalizers.Lowercase(),
            normalizers.StripAccents(),
        ]
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.WordLevelTrainer(
        vocab_size=vocab_size, min_frequency=1, special_tokens=["[UNK]"]
    )
    tokenizer.train_from_iterator(corpus, trainer)
    if save_to:
        tokenizer.save(save_to, pretty=True)
    return tokenizer


def get_imdb(
    vocab_size=400001,
    length=80,
    pretrained_glove: bool = False,
    shuffle_train_seed: int | None = None,
) -> tuple[Dataset, Dataset, Tokenizer]:
    assert vocab_size > 0
    # load data
    data_dir = str(Path(__file__).parent.parent.joinpath("data").resolve())
    if pretrained_glove:
        if vocab_size > 400001:
            raise ValueError("pretrained gloVe have max vocab size 400001")
        tokenizer_name = f"pretrain-glove-{vocab_size}.tokenizer.json"
    else:
        tokenizer_name = f"word-level-{vocab_size}.tokenizer.json"
    tokenizer_dir = Path(__file__).parent.joinpath(tokenizer_name).resolve()
    dsdict = load_dataset("imdb", cache_dir=data_dir)
    trainset, testset = dsdict["train"], dsdict["test"]
    # get tokenizer
    if tokenizer_dir.is_file():
        tokenizer = Tokenizer.from_file(str(tokenizer_dir))
    else:
        if pretrained_glove:
            tokenizer = _get_glove_tokenizer(vocab_size, str(tokenizer_dir))
        else:
            tokenizer = _build_tokenizer(
                trainset["text"], vocab_size, str(tokenizer_dir)
            )

    # tokenize dataset
    def _tokenize(text: str, tokenizer: Tokenizer):
        enc = tokenizer.encode(text)
        enc.truncate(length)
        enc.pad(length, direction="left")
        return enc.ids

    def _transform(x):
        return {
            "tokens": _tokenize(x["text"], tokenizer),
        }

    # tokenize train, shuffle once to remove bad surprise for split
    if shuffle_train_seed is None:
        trainset = trainset.map(
            _transform, remove_columns=["text"]
        ).with_format("numpy", dtype=np.int32)
    else:
        trainset = (
            trainset.map(_transform, remove_columns=["text"])
            .shuffle(seed=shuffle_train_seed)
            .flatten_indices()
            .with_format("numpy", dtype=np.int32)
        )
    testset = testset.map(_transform, remove_columns=["text"]).with_format(
        "numpy", dtype=np.int32
    )
    return Dataset(trainset), Dataset(testset), tokenizer


if __name__ == "__main__":
    trainset, _, tokenizer = get_imdb(pretrained_glove=True)
    # examples
    for i in range(10):
        lbl = trainset[i]["label"]
        tokens = trainset[i]["tokens"]
        print(f'"{tokenizer.decode(tokens)}" [{lbl}]\n')
