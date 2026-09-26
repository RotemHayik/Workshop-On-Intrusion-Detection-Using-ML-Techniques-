"""The two neural architectures; graph operations use native PyTorch."""

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence


class BiLSTM(nn.Module):
    def __init__(self, hidden=32, layers=1, dropout=0.2, **unused):
        super().__init__()
        self.q = nn.Embedding(17, 4, padding_idx=0)
        self.r = nn.Embedding(9, 3, padding_idx=0)
        self.lstm = nn.LSTM(
            22,
            hidden,
            layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if layers > 1 else 0,
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden * 2, 1))

    def forward(self, x, q, r, length):
        joined = torch.cat([x, self.q(q), self.r(r)], -1)
        # Bucket by exact valid length to retain optimized CPU LSTM kernels.
        # Every pass ends at the true sequence end, including the backward pass.
        encoded = joined.new_zeros((len(joined), self.lstm.hidden_size * 2))
        for size in torch.unique(length):
            indices = torch.where(length == size)[0]
            _, (h, _) = self.lstm(joined[indices, : int(size)])
            encoded = encoded.index_copy(0, indices, torch.cat([h[-2], h[-1]], -1))
        return self.head(encoded).squeeze(-1)


class SageLayer(nn.Module):
    """Separate self and mean-neighbor maps implement mean GraphSAGE."""

    def __init__(self, inp, out):
        super().__init__()
        self.self_map = nn.Linear(inp, out)
        self.neighbor_map = nn.Linear(inp, out, bias=False)

    def forward(self, x, edge):
        src, dst = edge
        total = torch.zeros_like(x).index_add_(0, dst, x[src])
        degree = (
            torch.bincount(dst, minlength=len(x)).clamp_min(1).to(x.dtype).unsqueeze(1)
        )
        return F.normalize(
            F.relu(self.self_map(x) + self.neighbor_map(total / degree)), p=2, dim=1
        )


class GraphSAGE(nn.Module):
    def __init__(self, hidden=32, layers=2, dropout=0.2, pooling="mean", **unused):
        super().__init__()
        self.pooling = pooling
        self.dropout = dropout
        self.layers = nn.ModuleList(
            [SageLayer(13 if i == 0 else hidden, hidden) for i in range(layers)]
        )
        self.head = nn.Linear(hidden * (2 if pooling == "mean_plus_max" else 1), 1)

    def forward(self, x, edge, batch):
        for layer in self.layers:
            x = F.dropout(layer(x, edge), p=self.dropout, training=self.training)
        n = int(batch.max()) + 1
        sums = x.new_zeros((n, x.shape[1])).index_add_(0, batch, x)
        mean = sums / torch.bincount(batch, minlength=n).unsqueeze(1)
        if self.pooling == "mean_plus_max":
            mx = x.new_full((n, x.shape[1]), -float("inf"))
            mx.scatter_reduce_(
                0, batch[:, None].expand_as(x), x, reduce="amax", include_self=True
            )
            mean = torch.cat([mean, mx], 1)
        return self.head(mean).squeeze(-1)
