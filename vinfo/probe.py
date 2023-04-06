"""Classes for specifying probe pytorch modules.
Draws from https://github.com/john-hewitt/structural-probes"""

import numpy
import torch
import torch.nn as nn
from tqdm import tqdm
from utils import InitYAMLObject
from yaml import YAMLObject


class Probe(nn.Module, InitYAMLObject):
    def print_param_count(self):
        total_params = 0
        for param in self.parameters():
            total_params += numpy.prod(param.size())
        tqdm.write("Probe has {} parameters".format(total_params))


class OneWordLinearLabelProbe(Probe):
    """Computes a linear function of each word vector.

    For a batch of sentences, computes all n scores
    for each sentence in the batch.
    """

    yaml_tag = "!OneWordLinearLabelProbe"

    def __init__(self, args, model_dim, label_space_size, zero_features=False):
        super(OneWordLinearLabelProbe, self).__init__()
        self.args = args
        self.model_dim = model_dim
        self.label_space_size = label_space_size
        self.linear = nn.Linear(self.model_dim, self.label_space_size)
        # self.linear2 = nn.Linear(self.label_space_size, self.label_space_size)
        self.print_param_count()
        dropout = 0.0
        self.dropout = nn.Dropout(p=dropout)
        print("Applying dropout {}".format(dropout))
        self.zero_features = zero_features
        self.to(args["device"])

    def forward(self, batch):
        """Computes all n label logits for each sentence in a batch.

        Args:
          batch: a batch of word representations of the shape
            (batch_size, max_seq_len, representation_dim)
        Returns:
          A tensor of logits of shape (batch_size, max_seq_len)
        """
        # batchlen, seqlen, dimension = batch.size()
        # batch = self.dropout(batch)
        # batch = self.linear1(batch)
        # logits = self.linear2(batch)
        if self.zero_features:
            batch = torch.zeros_like(batch)
        batch = self.linear(batch)
        # batch = self.linear2(torch.nn.functional.gelu(batch))
        return batch


class OneWordMLPLabelProbe(Probe):
    """Computes a linear function of each word vector.

    For a batch of sentences, computes all n scores
    for each sentence in the batch.
    """

    yaml_tag = "!OneWordMLPLabelProbe"

    def __init__(self, args, model_dim, label_space_size, zero_features=False):
        super(OneWordMLPLabelProbe, self).__init__()
        self.args = args
        self.model_dim = model_dim
        self.label_space_size = label_space_size
        self.linear = nn.Linear(self.model_dim, 100)
        self.linear2 = nn.Linear(100, self.label_space_size)
        self.print_param_count()
        dropout = 0.0
        self.dropout = nn.Dropout(p=dropout)
        print("Applying dropout {}".format(dropout))
        self.zero_features = zero_features
        self.to(args["device"])

    def forward(self, batch):
        """Computes all n label logits for each sentence in a batch.

        Args:
          batch: a batch of word representations of the shape
            (batch_size, max_seq_len, representation_dim)
        Returns:
          A tensor of logits of shape (batch_size, max_seq_len)
        """
        # batchlen, seqlen, dimension = batch.size()
        # batch = self.dropout(batch)
        # batch = self.linear1(batch)
        # logits = self.linear2(batch)
        if self.zero_features:
            batch = torch.zeros_like(batch)
        batch = self.linear(batch)
        batch = self.linear2(torch.nn.functional.gelu(batch))
        return batch


class SentenceLinearLabelProbe(Probe):
    """Computes a linear function of pairs of vectors.

    For a batch of sentences, computes all n scores
    for each sentence in the batch.
    """

    yaml_tag = "!SentenceLinearLabelProbe"

    def __init__(self, args, model_dim, label_space_size, zero_features=False):
        super(SentenceLinearLabelProbe, self).__init__()
        self.args = args
        self.model_dim = model_dim
        self.label_space_size = label_space_size
        self.linear = nn.Linear(self.model_dim, self.label_space_size)
        self.linear2 = nn.Linear(self.label_space_size, self.label_space_size)
        self.print_param_count()
        dropout = 0
        self.dropout = nn.Dropout(p=0)
        print("Applying dropout {}".format(dropout))
        self.zero_features = zero_features
        self.to(args["device"])

    def forward(self, batch):
        """Computes all n label logits for each sentence in a batch.

        Args:
          batch: a batch of word representations of the shape
            (batch_size, max_seq_len, representation_dim)
        Returns:
          A tensor of logits of shape (batch_size, max_seq_len)
        """
        # batch = torch.max(batch, dim=1).values
        batch = torch.mean(batch, dim=1)
        if self.zero_features:
            batch = torch.zeros_like(batch)
        # batch = self.linear2(torch.nn.functional.gelu(self.linear(batch)))
        batch = self.linear(batch)
        return batch


class SentenceMLPLabelProbe(Probe):
    """Computes a linear function of pairs of vectors.

    For a batch of sentences, computes all n scores
    for each sentence in the batch.
    """

    yaml_tag = "!SentenceMLPLabelProbe"

    def __init__(self, args, model_dim, label_space_size, zero_features=False):
        super(SentenceMLPLabelProbe, self).__init__()
        self.args = args
        self.model_dim = model_dim
        self.label_space_size = label_space_size
        self.linear = nn.Linear(self.model_dim, 100)
        self.linear2 = nn.Linear(100, self.label_space_size)
        self.print_param_count()
        dropout = 0
        self.dropout = nn.Dropout(p=0)
        print("Applying dropout {}".format(dropout))
        self.zero_features = zero_features
        self.to(args["device"])

    def forward(self, batch):
        """Computes all n label logits for each sentence in a batch.

        Args:
          batch: a batch of word representations of the shape
            (batch_size, max_seq_len, representation_dim)
        Returns:
          A tensor of logits of shape (batch_size, max_seq_len)
        """
        # batch = torch.max(batch, dim=1).values
        batch = torch.mean(batch, dim=1)
        if self.zero_features:
            batch = torch.zeros_like(batch)
        batch = self.linear2(torch.nn.functional.gelu(self.linear(batch)))
        return batch


class TwoWordPSDProbe(Probe):
    """Computes squared L2 distance after projection by a matrix.
    For a batch of sentences, computes all n^2 pairs of distances
    for each sentence in the batch.
    """

    yaml_tag = "!TwoWordPSDProbe"

    def __init__(self, args, model_dim, label_space_size):
        super(TwoWordPSDProbe, self).__init__()
        self.args = args
        self.label_space_size = label_space_size
        self.model_dim = model_dim
        self.proj = nn.Parameter(
            data=torch.zeros(self.model_dim, self.label_space_size)
        )
        nn.init.uniform_(self.proj, -0.05, 0.05)
        self.to(args["device"])

    def forward(self, batch):
        """Computes all n^2 pairs of distances after projection
        for each sentence in a batch.
        Note that due to padding, some distances will be non-zero for pads.
        Computes (B(h_i-h_j))^T(B(h_i-h_j)) for all i,j
        Args:
          batch: a batch of word representations of the shape
            (batch_size, max_seq_len, representation_dim)
        Returns:
          A tensor of distances of shape (batch_size, max_seq_len, max_seq_len)
        """
        transformed = torch.matmul(batch, self.proj)
        batchlen, seqlen, rank = transformed.size()
        transformed = transformed.unsqueeze(2)
        transformed = transformed.expand(-1, -1, seqlen, -1)
        transposed = transformed.transpose(1, 2)
        diffs = transformed - transposed
        squared_diffs = diffs.pow(2)
        squared_distances = torch.sum(squared_diffs, -1)
        return squared_distances


class OneWordPSDProbe(Probe):
    """Computes squared L2 norm of words after projection by a matrix."""

    yaml_tag = "!OneWordPSDProbe"

    def __init__(self, args, model_dim, label_space_size):
        super(OneWordPSDProbe, self).__init__()
        self.args = args
        self.label_space_size = label_space_size
        self.model_dim = model_dim
        self.proj = nn.Parameter(
            data=torch.zeros(self.model_dim, self.label_space_size)
        )
        nn.init.uniform_(self.proj, -0.05, 0.05)
        self.to(args["device"])

    def forward(self, batch):
        """Computes all n depths after projection
        for each sentence in a batch.
        Computes (Bh_i)^T(Bh_i) for all i
        Args:
          batch: a batch of word representations of the shape
            (batch_size, max_seq_len, representation_dim)
        Returns:
          A tensor of depths of shape (batch_size, max_seq_len)
        """
        transformed = torch.matmul(batch, self.proj)
        batchlen, seqlen, rank = transformed.size()
        norms = torch.bmm(
            transformed.view(batchlen * seqlen, 1, rank),
            transformed.view(batchlen * seqlen, rank, 1),
        )
        norms = norms.view(batchlen, seqlen)
        return norms
