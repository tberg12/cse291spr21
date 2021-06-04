### Assignment 3

In this assignment, you will only need to implement the inside algorithm of the CRF model. A python package `opt_einsum` is used to speed up the bi-affine function.


### Getting start
First, pull the assignment repo, and under the `assignment3` folder, run:

```sh
pip install opt_einsum graphviz stanza
python main.py
```


### Drawing a Tree

```python
from node import from_string, draw_tree
t = from_string(l)
draw_tree(t)