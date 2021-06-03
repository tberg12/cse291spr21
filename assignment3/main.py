import torch.nn as nn
import torch
import os
from opt_einsum import contract
from datetime import datetime, timedelta
from collections import Counter
import torch.autograd as autograd
from torch.optim import Adam
from data import Dataset, Tree, Field, RawField, ChartField
import argparse
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

class Metric(object):
  
    def __lt__(self, other):
        return self.score < other

    def __le__(self, other):
        return self.score <= other

    def __ge__(self, other):
        return self.score >= other

    def __gt__(self, other):
        return self.score > other

    @property
    def score(self):
        return 0.


class SpanMetric(Metric):
  
    def __init__(self, eps=1e-12):
        super().__init__()

        self.n = 0.0
        self.n_ucm = 0.0
        self.n_lcm = 0.0
        self.utp = 0.0
        self.ltp = 0.0
        self.pred = 0.0
        self.gold = 0.0
        self.eps = eps

    def __call__(self, preds, golds):
        for pred, gold in zip(preds, golds):
            upred = Counter([(i, j) for i, j, label in pred])
            ugold = Counter([(i, j) for i, j, label in gold])
            utp = list((upred & ugold).elements())
            lpred = Counter(pred)
            lgold = Counter(gold)
            ltp = list((lpred & lgold).elements())
            self.n += 1
            self.n_ucm += len(utp) == len(pred) == len(gold)
            self.n_lcm += len(ltp) == len(pred) == len(gold)
            self.utp += len(utp)
            self.ltp += len(ltp)
            self.pred += len(pred)
            self.gold += len(gold)
        return self

    def __repr__(self):
        s = f"UCM: {self.ucm:6.2%} LCM: {self.lcm:6.2%} "
        s += f"UP: {self.up:6.2%} UR: {self.ur:6.2%} UF: {self.uf:6.2%} "
        s += f"LP: {self.lp:6.2%} LR: {self.lr:6.2%} LF: {self.lf:6.2%}"

        return s

    @property
    def score(self):
        return self.lf

    @property
    def ucm(self):
        return self.n_ucm / (self.n + self.eps)

    @property
    def lcm(self):
        return self.n_lcm / (self.n + self.eps)

    @property
    def up(self):
        return self.utp / (self.pred + self.eps)

    @property
    def ur(self):
        return self.utp / (self.gold + self.eps)

    @property
    def uf(self):
        return 2 * self.utp / (self.pred + self.gold + self.eps)

    @property
    def lp(self):
        return self.ltp / (self.pred + self.eps)

    @property
    def lr(self):
        return self.ltp / (self.gold + self.eps)

    @property
    def lf(self):
        return 2 * self.ltp / (self.pred + self.gold + self.eps)


def stripe(x, n, w, offset=(0, 0), dim=1):
    r"""
    Returns a diagonal stripe of the tensor.

    Args:
        x (~torch.Tensor): the input tensor with 2 or more dims.
        n (int): the length of the stripe.
        w (int): the width of the stripe.
        offset (tuple): the offset of the first two dims.
        dim (int): 1 if returns a horizontal stripe; 0 otherwise.

    Returns:
        a diagonal stripe of the tensor.

    Examples:
        >>> x = torch.arange(25).view(5, 5)
        >>> x
        tensor([[ 0,  1,  2,  3,  4],
                [ 5,  6,  7,  8,  9],
                [10, 11, 12, 13, 14],
                [15, 16, 17, 18, 19],
                [20, 21, 22, 23, 24]])
        >>> stripe(x, 2, 3)
        tensor([[0, 1, 2],
                [6, 7, 8]])
        >>> stripe(x, 2, 3, (1, 1))
        tensor([[ 6,  7,  8],
                [12, 13, 14]])
        >>> stripe(x, 2, 3, (1, 1), 0)
        tensor([[ 6, 11, 16],
                [12, 17, 22]])
    """

    x, seq_len = x.contiguous(), x.size(1)
    stride, numel = list(x.stride()), x[0, 0].numel()
    stride[0] = (seq_len + 1) * numel
    stride[1] = (1 if dim == 1 else seq_len) * numel
    return x.as_strided(size=(n, w, *x.shape[2:]),
                        stride=stride,
                        storage_offset=(offset[0]*seq_len+offset[1])*numel)

def cky(scores, mask):
    lens = mask[:, 0].sum(-1)
    batch_size, seq_len, seq_len, n_labels = scores.shape
    scores = scores.permute(1, 2, 3, 0)

    s = scores.new_zeros(seq_len, seq_len, batch_size)
    p_s = scores.new_zeros(seq_len, seq_len, batch_size).long()
    p_l = scores.new_zeros(seq_len, seq_len, batch_size).long()

    for d in range(2, seq_len + 1): # d = 2, 3, ..., seq_len
        # define the offset variable for convenience
        offset = d - 1
        n = seq_len - offset
        starts = p_s.new_tensor(range(n)).unsqueeze(0)
        # [batch_size, n]
        s_label, p_label = scores.diagonal(offset).max(0)
        p_l.diagonal(offset).copy_(p_label)

        if d == 2:
            s.diagonal(offset).copy_(s_label)
            continue
        # [n, offset, batch_size]
        s_span = stripe(s, n, offset-1, (0, 1)) + stripe(s, n, offset-1, (1, offset), 0)
        # [batch_size, n, offset]
        s_span = s_span.permute(2, 0, 1)
        # [batch_size, n]
        s_span, p_span = s_span.max(-1)
        s.diagonal(offset).copy_(s_span + s_label)
        p_s.diagonal(offset).copy_(p_span + starts + 1)

    def backtrack(p_s, p_l, i, j):
        if j == i + 1:
            return [(i, j, p_l[i][j])]
        split, label = p_s[i][j], p_l[i][j]
        ltree = backtrack(p_s, p_l, i, split)
        rtree = backtrack(p_s, p_l, split, j)
        return [(i, j, label)] + ltree + rtree

    p_s = p_s.permute(2, 0, 1).tolist()
    p_l = p_l.permute(2, 0, 1).tolist()
    trees = [backtrack(p_s[i], p_l[i], 0, length)
             for i, length in enumerate(lens.tolist())]

    return trees


class CRFConstituency(nn.Module):
    r"""
    TreeCRF for calculating partitions and marginals of constituency trees in :math:`O(n^3)` :cite:`zhang-etal-2020-fast`.
    """

    @torch.enable_grad()
    def forward(self, scores, mask, target=None, require_marginals=False):
        lens = mask[:, 0].sum(-1)
        total = lens.sum()

        batch_size, seq_len, seq_len, n_labels = scores.shape
        # always enable the gradient computation of scores
        # in order for the computation of marginal probs

        s = self.inside(scores.requires_grad_(), mask)
        logZ = s[0].gather(0, lens.unsqueeze(0)).sum()

        # marginal probs are used for decoding, and can be computed by
        # combining the inside algorithm and autograd mechanism
        # instead of the entire inside-outside process
        
        probs = scores
        if require_marginals:
            probs, = autograd.grad(logZ, scores, retain_graph=scores.requires_grad)
        if target is None:
            return probs

        target_mask = mask & (target > -1)

        score_tmp = scores.reshape(-1, n_labels)[target_mask.reshape(-1)]
        labels = target[target_mask]
        score = score_tmp[torch.arange(labels.size(0)), labels]

        loss = (logZ - score.sum(-1)) / total
        return loss, probs


    def inside(self, scores, mask):
        batch_size, seq_len, seq_len, n_labels = scores.shape
        # [seq_len, seq_len, n_labels, batch_size]
        scores = scores.permute(1, 2, 3, 0)
        # [seq_len, seq_len, batch_size]
        mask = mask.permute(1, 2, 0)

        # working in the log space, initial s with log(0) == -inf
        s = torch.full_like(scores[:, :, 0], float('-inf'))

        for d in range(2, seq_len + 1): # d = 2, 3, ..., seq_len
            # define the offset variable for convenience
            offset = d - 1
            # n denotes the number of spans to iterate,
            # from span (0, offset) to span (n, n+offset) given the offset
            n = seq_len - offset
            # diag_mask is used for ignoring the excess of each sentence
            # [batch_size, n]
            diag_mask = mask.diagonal(offset)

            ##### TODO   
            # if d == 2:
            #    DO something 
            # else:
            #    DO something

        return s


class Biaffine(nn.Module):
    r"""
    Biaffine layer for first-order scoring :cite:`dozat-etal-2017-biaffine`.

    This function has a tensor of weights :math:`W` and bias terms if needed.
    The score :math:`s(x, y)` of the vector pair :math:`(x, y)` is computed as :math:`x^T W y / d^s`,
    where `d` and `s` are vector dimension and scaling factor respectively.
    :math:`x` and :math:`y` can be concatenated with bias terms.

    Args:
        n_in (int):
            The size of the input feature.
        n_out (int):
            The number of output channels.
        scale (float):
            Factor to scale the scores. Default: 0.
        bias_x (bool):
            If ``True``, adds a bias term for tensor :math:`x`. Default: ``True``.
        bias_y (bool):
            If ``True``, adds a bias term for tensor :math:`y`. Default: ``True``.
    """

    def __init__(self, n_in, n_out=1, scale=0, bias_x=True, bias_y=True):
        super().__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.scale = scale
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.weight = nn.Parameter(torch.Tensor(n_out, n_in+bias_x, n_in+bias_y))

        self.reset_parameters()

    def __repr__(self):
        s = f"n_in={self.n_in}"
        if self.n_out > 1:
            s += f", n_out={self.n_out}"
        if self.scale != 0:
            s += f", scale={self.scale}"
        if self.bias_x:
            s += f", bias_x={self.bias_x}"
        if self.bias_y:
            s += f", bias_y={self.bias_y}"

        return f"{self.__class__.__name__}({s})"

    def reset_parameters(self):
        nn.init.zeros_(self.weight)

    def forward(self, x, y):
        r"""
        Args:
            x (torch.Tensor): ``[batch_size, seq_len, n_in]``.
            y (torch.Tensor): ``[batch_size, seq_len, n_in]``.

        Returns:
            ~torch.Tensor:
                A scoring tensor of shape ``[batch_size, n_out, seq_len, seq_len]``.
                If ``n_out=1``, the dimension for ``n_out`` will be squeezed automatically.
        """

        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)
        # [batch_size, n_out, seq_len, seq_len]
        s = contract('bxi,oij,byj->boxy', x, self.weight, y) / self.n_in ** self.scale
        # remove dim 1 if n_out == 1
        s = s.squeeze(1)

        return s


class MLP(nn.Module):
    r"""
    Applies a linear transformation together with a non-linear activation to the incoming tensor:
    :math:`y = \mathrm{Activation}(x A^T + b)`

    Args:
        n_in (~torch.Tensor):
            The size of each input feature.
        n_out (~torch.Tensor):
            The size of each output feature.
        dropout (float):
            If non-zero, introduces a :class:`SharedDropout` layer on the output with this dropout ratio. Default: 0.
        activation (bool):
            Whether to use activations. Default: True.
    """

    def __init__(self, n_in, n_out, dropout=0, activation=True):
        super().__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.linear = nn.Linear(n_in, n_out)
        self.activation = nn.LeakyReLU(negative_slope=0.1) if activation else nn.Identity()
        self.dropout = nn.Dropout(p=dropout)

        self.reset_parameters()

    def __repr__(self):
        s = f"n_in={self.n_in}, n_out={self.n_out}"
        if self.dropout.p > 0:
            s += f", dropout={self.dropout.p}"

        return f"{self.__class__.__name__}({s})"

    def reset_parameters(self):
        nn.init.orthogonal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        r"""
        Args:
            x (~torch.Tensor):
                The size of each input feature is `n_in`.

        Returns:
            A tensor with the size of each output feature `n_out`.
        """

        x = self.linear(x)
        x = self.activation(x)
        x = self.dropout(x)

        return x


class Model(nn.Module):
    def __init__(self, n_words, n_labels, n_tags, n_embed=100, n_feat_embed=100, embed_dropout=.33, 
                    n_lstm_hidden=400, n_lstm_layers=3, encoder_dropout=.33,
                    n_label_mlp=100, mlp_dropout=.33):

        super().__init__()
        self.word_embed = nn.Embedding(num_embeddings=n_words, embedding_dim=n_embed)
        n_input = n_embed
        self.tag_embed = nn.Embedding(num_embeddings=n_tags, embedding_dim=n_feat_embed)
        n_input += n_feat_embed
        self.embed_dropout = nn.Dropout(p=embed_dropout)
        self.encoder = nn.LSTM(input_size=n_input, hidden_size=n_lstm_hidden, num_layers=n_lstm_layers,
                        bidirectional=True, dropout=encoder_dropout)
        self.encoder_dropout = nn.Dropout(p=encoder_dropout)

        args.n_hidden = n_lstm_hidden * 2

        self.mlp_l = MLP(n_in=args.n_hidden, n_out=n_label_mlp, dropout=mlp_dropout)
        self.mlp_r = MLP(n_in=args.n_hidden, n_out=n_label_mlp, dropout=mlp_dropout)


        self.feat_biaffine = Biaffine(n_in=n_label_mlp, n_out=n_labels, bias_x=True, bias_y=True)
        self.crf = CRFConstituency()
        self.criterion = nn.CrossEntropyLoss()

    def embed(self, words, feats):
        ext_words = words

        # get outputs from embedding layers
        word_embed = self.word_embed(ext_words)
        tag_embed = self.tag_embed(feats.pop())
        # concatenate the word and tag representations
        embed = torch.cat((word_embed, tag_embed), -1)
        
        return self.embed_dropout(embed)


    def forward(self, words, feats=None):
        r"""
        Args:
            words (~torch.LongTensor): ``[batch_size, seq_len]``.
                Word indices.
            feats (list[~torch.LongTensor]):
                A list of feat indices.
                The size is either ``[batch_size, seq_len, fix_len]`` if ``feat`` is ``'char'`` or ``'bert'``,
                or ``[batch_size, seq_len]`` otherwise.
                Default: ``None``.

        Returns:
            ~torch.Tensor:
                The second of shape ``[batch_size, seq_len, seq_len, n_labels]`` stores
                scores of all possible labels on each constituent.
        """

        x = self.encode(words, feats)

        x_f, x_b = x.chunk(2, -1)
        x = torch.cat((x_f[:, :-1], x_b[:, 1:]), -1)

        # apply MLPs to the BiLSTM output states
        feat_r_l = self.mlp_l(x)
        feat_r_r = self.mlp_r(x)

        # [batch_size, seq_len, seq_len, n_labels]
        feat_r = self.feat_biaffine(feat_r_l, feat_r_r).permute(0, 2, 3, 1)

        return feat_r


    def loss(self, scores, charts, mask, require_marginals=True):

        loss, scores = self.crf(scores, mask, charts, require_marginals=require_marginals)

        return loss, scores

    def encode(self, words, feats=None):
        x = pack_padded_sequence(self.embed(words, feats), words.ne(args.pad_index).sum(1).tolist(), True, False)
        x, _ = self.encoder(x)
        x, _ = pad_packed_sequence(x, True, total_length=words.shape[1])
        return self.encoder_dropout(x)

    def decode(self, s_feat, mask):
        scores = cky(s_feat, mask)
        return scores

def train(model, traindata, devdata, optimizer):
    elapsed = timedelta()
    best_e, best_metric = 1, Metric()

    for epoch in range(1, args.epochs + 1):
        start = datetime.now()

        print(f"Epoch {epoch} / {args.epochs}:")
        model.train()

        for i, (words, *feats, trees, charts) in enumerate(traindata.loader):
            # mask out the lower left triangle     
            word_mask = words.ne(args.pad_index)[:, 1:]
            mask = word_mask if len(words.shape) < 3 else word_mask.any(-1)
            mask = (mask.unsqueeze(1) & mask.unsqueeze(2)).triu_(1)

            s_feat = model(words, feats)
            loss, _ = model.loss(s_feat, charts, mask, require_marginals=False)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            optimizer.zero_grad()
            if i % 50 == 0:
                print(f"{i} iter of epoch {epoch}, loss: {loss:.4f}")
        
        loss, dev_metric = evaluate(model, devdata.loader)
        print(f"{'dev:':5} loss: {loss:.4f} - {dev_metric}")

        t = datetime.now() - start
        if dev_metric > best_metric:
            best_e, best_metric = epoch, dev_metric
        elapsed += t

        print(f"Epoch {best_e} saved")
        print(f"{'dev:':5} {best_metric}")
        print(f"{elapsed}s elapsed, {elapsed / epoch}s/epoch")

@torch.no_grad()
def evaluate(model, loader):

    model.eval()

    total_loss, metric = 0, SpanMetric()

    for words, *feats, trees, charts in loader:
        # mask out the lower left triangle     
        word_mask = words.ne(args.pad_index)[:, 1:]
        mask = word_mask if len(words.shape) < 3 else word_mask.any(-1)
        mask = (mask.unsqueeze(1) & mask.unsqueeze(2)).triu_(1)

        s_feat = model(words, feats)
        loss, s_feat = model.loss(s_feat, charts, mask, require_marginals=True)
        chart_preds = model.decode(s_feat, mask)
        # since the evaluation relies on terminals,
        # the tree should be first built and then factorized
        preds = [Tree.build(tree, [(i, j, CHART.vocab[label]) for i, j, label in chart])
                    for tree, chart in zip(trees, chart_preds)]
        total_loss += loss.item()
        metric([Tree.factorize(tree, args.delete, args.equal) for tree in preds],
                [Tree.factorize(tree, args.delete, args.equal) for tree in trees])
    total_loss /= len(loader)

    return total_loss, metric

@torch.no_grad()
def predict(model, loader):
    model.eval()

    preds = {'trees': [], 'probs': []}
    for words, *feats, trees in loader:
        word_mask = words.ne(args.pad_index)[:, 1:]
        mask = word_mask if len(words.shape) < 3 else word_mask.any(-1)
        mask = (mask.unsqueeze(1) & mask.unsqueeze(2)).triu_(1)
        lens = mask[:, 0].sum(-1)
        s_feat = model(words, feats)
        s_span = model.crf(s_feat, mask, require_marginals=True)
        chart_preds = model.decode(s_span, mask)
        preds['trees'].extend([Tree.build(tree, [(i, j, CHART.vocab[label]) for i, j, label in chart])
                                for tree, chart in zip(trees, chart_preds)])
        if args.prob:
            preds['probs'].extend([prob[:i-1, 1:i].cpu() for i, prob in zip(lens, s_span)])

    return preds


if __name__ == "__main__":
    global args

    p = argparse.ArgumentParser(description='Create CRF Constituency Parser.')
    p.add_argument('--feat', '-f', choices=['tag', 'char'], nargs='+', help='features to use')
    p.add_argument('--encoder', choices=['lstm', 'bert'], default='lstm', help='encoder to use')
    p.add_argument('--max-len', type=int, help='max length of the sentences')
    p.add_argument('--buckets', default=32, type=int, help='max num of buckets to use')
    p.add_argument('--batch-size', default=256, type=int, help='training batch size')
    p.add_argument('--data', default='data/ptb/', help='path to data file')
    p.add_argument('--embed', default=None, help='path to pretrained embeddings')
    p.add_argument('--unk', default='unk', help='unk token in pretrained embeddings')
    p.add_argument('--n-embed', default=100, type=int, help='dimension of embeddings')
    p.add_argument('--n-lstm-hidden', default=200, type=int, help='dimension of lstm hidden')
    p.add_argument('--n-lstm-layers', default=2, type=int, help='number of lstm layers')
    p.add_argument('--epochs', default=10, type=int, help='the number of training epochs')
    p.add_argument('--lr', default=2e-3, type=float, help='learning rate')
    p.add_argument('--mu', default=.9, type=float, help='learning rate, mu')
    p.add_argument('--nu', default=.9, type=float, help='learning rate, nu')
    p.add_argument('--eps', default=1e-12, type=float, help='learning rate, eps')
    p.add_argument('--weight-decay', default=1e-5, type=float, help='learning rate, weight decay')
    p.add_argument('--clip', default=5., type=float, help='gradient clipping')


    args = p.parse_args()

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.mbr = True

    args.delete = {'TOP', 'S1', '-NONE-', ',', ':', '``', "''", '.', '?', '!', ''}
    args.equal = {'ADVP': 'PRT'}

    args.train = os.path.join(args.data, 'train.pid')
    args.dev = os.path.join(args.data, 'dev.pid')
    args.test = os.path.join(args.data, 'test.pid')

    print("Building the fields")
    WORD = Field('words', pad='<pad>', unk='<unk>', bos='<bos>', eos='<eos>', lower=True)
    TAG = Field('tags', bos="<bos>", eos='<eos>')
    TREE = RawField('trees')
    global CHART
    CHART = ChartField('charts')
    transform = Tree(WORD=(WORD), POS=TAG, TREE=TREE, CHART=CHART)
    traindata = Dataset(transform, args.train)
    WORD.build(traindata, 2, None)
    CHART.build(traindata)
    TAG.build(traindata)

    args.encoder = 'lstm'
    args.n_words = WORD.vocab.n_init
    args.n_labels = len(CHART.vocab)
    args.n_tags = len(TAG.vocab) if TAG is not None else None
    args.pad_index = WORD.pad_index
    args.unk_index = WORD.unk_index
    args.bos_index = WORD.bos_index
    args.eos_index = WORD.eos_index
    print(args)
    print("Building the model")
    model = Model(n_words=args.n_words, n_labels=args.n_labels, n_tags=args.n_tags, 
                    n_lstm_hidden=args.n_lstm_hidden, n_lstm_layers=args.n_lstm_layers).to(args.device)
    print(model)

    transform.train()
    traindata = Dataset(transform, args.train)
    devdata = Dataset(transform, args.dev)
    testdata = Dataset(transform, args.test)
    traindata.build(args.batch_size, args.buckets, True)
    devdata.build(args.batch_size, args.buckets)
    testdata.build(args.batch_size, args.buckets)
    print(f"\n{'train:':6} {traindata}\n{'dev:':6} {devdata}\n{'test:':6} {testdata}\n")
    optimizer = Adam(model.parameters(), args.lr, (args.mu, args.nu), args.eps, args.weight_decay)
    train(model, traindata, devdata, optimizer)
    evaluate(model, testdata.loader)