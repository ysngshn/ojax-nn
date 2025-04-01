# IMDB textual sentiment analysis

> Note: 
> - The Huggingface `tokenizers` package is used to prepare the IMDB dataset, thus one should install this package before running this example.
> - The pretrained GloVe word embedding (40k vocab, 300 features) from Gensim is also used. Thus `gensim` should also be installed to reproduce the results with pretrained embedding.

Here we have a classical NLP classification task which can be tackled with different architectures: temporal CNNs, RNNs and transformer. This setup uses the word tokenization from Huggingface and pretrained GloVe word embedding, which helps the training with small NLP datasets like IMDB.

We also showcase a Pallas-based implementation of a custom RNN called [Independently RNN](http://arxiv.org/abs/1803.04831) and how it can easily work with OJAX-NN.

Experiments can be run with command
```python
python run.py <model-name> <adam_lr>
```
e.g., `python run.py indrnn-6 1e-3`. The list of available models can be found in `run.py` and their definitions in `models.py`.
