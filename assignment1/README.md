# Assignment 1

Please follow the below installation instructions before beginning the assignment.


## Note: This repo only works with torchtext 0.9 or above which requires PyTorch 1.8 or above. If you are using torchtext 0.8 then please use [this](https://github.com/bentrevett/pytorch-seq2seq/tree/torchtext08) branch

This repo contains tutorials covering understanding and implementing sequence-to-sequence (seq2seq) models using [PyTorch](https://github.com/pytorch/pytorch) 1.8, [torchtext](https://github.com/pytorch/text) 0.9 and [spaCy](https://spacy.io/) 3.0,  using Python 3.8.


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

Or, if you're using CoLab:

``` python
import spacy.cli
spacy.cli.download("de_core_news_sm")
spacy.cli.download("en_core_web_sm")
```


## Acknowledgements

Code comes from the pytorch-seq2seq repository [here](https://github.com/bentrevett/pytorch-seq2seq)
    
