# Assignment 1

Please follow the below installation instructions before beginning the assignment.


## Note: This repo only works with torchtext 0.9 or above which requires PyTorch 1.8 or above. If you are using torchtext 0.8 then please use [this](https://github.com/bentrevett/pytorch-seq2seq/tree/torchtext08) branch

This repo contains tutorials covering understanding and implementing sequence-to-sequence (seq2seq) models using [PyTorch](https://github.com/pytorch/pytorch) 1.8, [torchtext](https://github.com/pytorch/text) 0.9 and [spaCy](https://spacy.io/) 3.0,  using Python 3.8.

**If you find any mistakes or disagree with any of the explanations, please do not hesitate to [submit an issue](https://github.com/bentrevett/pytorch-seq2seq/issues/new). I welcome any feedback, positive or negative!**

## Getting Started

To install PyTorch, see installation instructions on the [PyTorch website](pytorch.org).

To install torchtext:

``` bash
pip install torchtext
```

We'll also make use of spaCy to tokenize our data. To install spaCy, follow the instructions [here](https://spacy.io/usage/) making sure to install both the English and German models with:

``` bash
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm
```

## Tutorials

* 4 - [Packed Padded Sequences, Masking, Inference and BLEU](https://github.com/bentrevett/pytorch-seq2seq/blob/master/4%20-%20Packed%20Padded%20Sequences%2C%20Masking%2C%20Inference%20and%20BLEU.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bentrevett/pytorch-seq2seq/blob/master/4%20-%20Packed%20Padded%20Sequences%2C%20Masking%2C%20Inference%20and%20BLEU.ipynb)

    In this notebook, we will improve the previous model architecture by adding *packed padded sequences* and *masking*. These are two methods commonly used in NLP. Packed padded sequences allow us to only process the non-padded elements of our input sentence with our RNN. Masking is used to force the model to ignore certain elements we do not want it to look at, such as attention over padded elements. Together, these give us a small performance boost. We also cover a very basic way of using the model for inference, allowing us to get translations for any sentence we want to give to the model and how we can view the attention values over the source sequence for those translations. Finally, we show how to calculate the BLEU metric from our translations.


## Acknowledgements

Code comes from the pytorch-seq2seq repository [here](https://github.com/bentrevett/pytorch-seq2seq)
    
