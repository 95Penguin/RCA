"""Author neural specialists, with declared historical-task input adapters."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from baselines.author_imports import author_package
from .frameworks import dgl_batch


class AnoFusionAuthor(nn.Module):
    def __init__(self, data):
        super().__init__()
        from sklearn.metrics import mutual_info_score
        api = author_package("anofusion", "model", "AnoFusion", ("GTblock", "GTlayer", "GATGRU", "GAT"))
        d = data.dataset
        # Author's six relation types: three within-modality and three between.
        f, n = len(d.feature_columns), len(d.services)
        order = [i for m in ("metric", "log", "trace") for i in data.groups[m]]
        self.order = order
        modalities = [m for m in range(3) for _ in data.groups[("metric", "log", "trace")[m]]]
        relation = {(0, 0): 0, (1, 1): 1, (2, 2): 2, (0, 1): 3, (0, 2): 4, (1, 2): 5}
        matrices = np.zeros((n, f, f, 6), dtype=np.float32)
        for service in range(n):
            values = d.features[data.input_fit_indices, service][:, order]
            # Explicit training-only discretization avoids treating each continuous
            # reading as its own category in sklearn's discrete MI estimator.
            discrete = np.zeros_like(values, dtype=int)
            for j in range(f):
                bins = np.unique(np.quantile(values[:, j], np.linspace(0, 1, 9)))
                discrete[:, j] = np.searchsorted(bins[1:-1], values[:, j])
            for i in range(f):
                for j in range(f):
                    k = relation[tuple(sorted((modalities[i], modalities[j])))]
                    matrices[service, i, j, k] = mutual_info_score(discrete[:, i], discrete[:, j])
        self.register_buffer("relations", torch.from_numpy(matrices))
        self.model = api.Net(f, 6, d.history-1, 0.1)

    def forward(self, x):
        b, t, n, f = x.shape
        values = x[..., self.order].permute(0, 2, 3, 1).reshape(b*n, f, t)
        graph = self.relations[None].expand(b, -1, -1, -1, -1).reshape(b*n, f, f, 6)
        predicted = self.model(values[..., :-1].contiguous(), graph)
        loss = (predicted-values[..., -1]).square().mean(dim=-1).reshape(b, n)
        return {"loss": loss.mean(), "risk": loss.max(dim=1).values}


class HadesAuthor(nn.Module):
    def __init__(self, metric_num, dimension=32, hidden=32):
        super().__init__()
        api = author_package("hades", "hades/codes", "models.fuse", ("models",))
        self.model = api.MultiModel([metric_num], "cpu", feature_type="word2vec",
                                   hidden_size=hidden, word_embedding_dim=dimension,
                                   window_size=5, log_window_size=16, self_attention=True,
                                   temporal_hidden_sizes=[hidden], temporal_kernel_sizes=[2],
                                   temporal_dropout=0.1, pooling=False,
                                   inner_hidden_sizes=[hidden], inner_kernel_sizes=[2], inner_dropout=0.1,
                                   decoder_layer_num=2, linear_size=hidden,
                                   log_layer_num=2, transformer_hidden=hidden*2)

    def forward(self, metrics, logs):
        fused, _ = self.model.encoder([metrics.transpose(1, 2)], logs)
        return self.model.decoder(fused)


class DiagFusionAuthor(nn.Module):
    def __init__(self, dimension, nodes, types, hidden=32):
        super().__init__()
        api = author_package("diagfusion", "", "models.layers", ("models",))
        self.root = api.TAGClassifier(dimension, hidden, nodes)
        self.type_classifier = api.TAGClassifier(dimension, hidden, types)

    def forward(self, graph):
        features = torch.stack([graph.ndata[m] for m in ("metric", "log", "trace")]).mean(dim=0)
        return self.root(graph, features), self.type_classifier(graph, features)


class TVDiagAuthor(nn.Module):
    def __init__(self, dimension, types, hidden=32):
        super().__init__()
        roots = ("config", "core", "helper")
        api = author_package("tvdiag", "", "core.model.MainModel", roots)
        config = SimpleNamespace(modalities=["metric", "log", "trace"],
                                 alert_embedding_dim=dimension, graph_hidden_dim=hidden,
                                 graph_out=hidden, graph_layers=2, aggregator="mean",
                                 feat_drop=0.1, linear_hidden=[hidden], ft_num=types)
        self.model = api.MainModel(config)
        aw = author_package("tvdiag", "", "core.loss.AutomaticWeightedLoss", roots)
        sup = author_package("tvdiag", "", "core.loss.SupervisedContrastiveLoss", roots)
        unsup = author_package("tvdiag", "", "core.loss.UnsupervisedContrastiveLoss", roots)
        self.awl = aw.AutomaticWeightedLoss(4)
        self.sup, self.unsup = sup.SupConLoss(device="cpu"), unsup.UspConLoss(device="cpu")
        self.augment = author_package("tvdiag", "", "core.aug", roots)

    def forward(self, graph):
        _, _, root, fault_type = self.model(graph)
        return root, fault_type

    def objective(self, graphs, root_labels, type_labels):
        import dgl
        augmented = [self.augment.aug_drop_node(g, int(root), 0.2) for g, root in zip(graphs, root_labels)]
        batch = dgl.batch(graphs+augmented)
        identities = torch.cat([root_labels, root_labels])
        types = torch.cat([type_labels, type_labels])
        features, _, root, fault_type = self.model(batch)
        offset, losses = 0, []
        for count in batch.batch_num_nodes().tolist():
            target = batch.ndata["root"][offset:offset+count].argmax().reshape(1)
            losses.append(F.cross_entropy(root[offset:offset+count].reshape(1, -1), target))
            offset += count
        rcl = torch.stack(losses).mean()
        fti = F.cross_entropy(fault_type, types)
        task = self.sup(features["metric"], identities)+self.sup(features["trace"], identities)+self.sup(features["log"], types)
        cross = self.unsup(features["metric"], features["log"])+self.unsup(features["metric"], features["trace"])
        return self.awl(rcl, fti, 0.1*task, 0.1*cross)


class MaatAuthor(nn.Module):
    def __init__(self, target_dim, diff_steps=100, samples=16, hidden=32):
        super().__init__()
        api = author_package("maat", "src", "model.network")
        config = dict(target_dim=target_dim, input_size=target_dim*3+2,
                      history_length=5, context_length=4, prediction_length=1,
                      lags_seq=[1], cell_hidden_size=hidden, cell_num_layers=2,
                      embedding_dimension=2, conditioning_hidden=hidden,
                      diff_steps=diff_steps, scaling=False)
        self.train_net = api.TrainingNetwork(**config)
        self.predict_net = api.PredictionNetwork(num_parallel_samples=samples, **config)
        self.config = config

    def arguments(self, x, observed=None):
        b, t, c = x.shape
        # Deterministic relative position features; future positions are known,
        # not future telemetry or labels.
        positions = torch.arange(6, dtype=x.dtype)/6*2*torch.pi
        time = torch.stack([positions.sin(), positions.cos()], dim=-1)
        return dict(target_dimension_indicator=torch.arange(c)[None].expand(b, -1),
                    past_time_feat=time[None, :5].expand(b, -1, -1), past_target_cdf=x,
                    past_observed_values=torch.ones_like(x) if observed is None else observed,
                    past_is_pad=torch.zeros(b, 5),
                    future_time_feat=time[None, 5:].expand(b, -1, -1))

    def loss(self, x, future, observed, past_observed=None):
        args = self.arguments(x, past_observed)
        result = self.train_net(**args, future_target_cdf=future[:, None],
                                 future_observed_values=observed[:, None])
        return result[0]

    @torch.no_grad()
    def forecast(self, x, seed, observed=None):
        self.predict_net.load_state_dict(self.train_net.state_dict())
        self.predict_net.eval()
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            return self.predict_net(**self.arguments(x, observed))[:, :, 0]

    @staticmethod
    def forecast_features(samples, latest):
        # Repository extract_feat.py explicitly supports customized features.
        # Include level, change and forecast uncertainty; no realized future value.
        mean = samples.mean(axis=1)
        return np.concatenate([mean, mean-latest, samples.std(axis=1)], axis=-1)
