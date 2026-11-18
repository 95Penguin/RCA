"""Explicitly named author-model / RE2-panel adaptations, not paper-score replicas."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from baselines.source import load_author_module

METHODS = ("ltfp", "instantops", "eadro", "art")
IMPLEMENTATIONS = {
    "ltfp": "ltfp-upstream-panel-state-v1",
    "instantops": "instantops-paper-panel-permutation-v1",
    "eadro": "eadro-upstream-panel-state-v1",
    "art": "art-upstream-panel-score-lag-v1",
}


def dgl_batch(adjacency):
    import dgl
    graphs = []
    if adjacency.ndim == 4:
        adjacency = adjacency[:, -1]
    for a in adjacency:
        a = a > 0
        a = a | torch.eye(a.shape[0], device=a.device, dtype=torch.bool)
        i, j = a.nonzero(as_tuple=True)
        # Existing common graph is symmetric. No inferred causal direction.
        graphs.append(dgl.graph((i, j), num_nodes=a.shape[0], device=a.device))
    return dgl.batch(graphs)


class LTFPAuthor(nn.Module):
    def __init__(self, feature_dim, groups, hidden, dropout, correlation):
        super().__init__()
        upstream = load_author_module("ltfp", "src/ltfp/torch_models.py")
        self.order = [i for name in ("metric", "log", "trace") for i in groups.get(name, [])]
        self.model = upstream.LTFPModel(upstream.LTFPConfig(
            feature_dim, hidden, 0, 1, n_tcn_layers=3, n_graph_layers=2,
            dropout=dropout, modality_dims=tuple(len(groups[n]) for n in ("metric", "log", "trace") if n in groups)))
        self.register_buffer("correlation", torch.as_tensor(correlation, dtype=torch.float32))

    def forward(self, x, graph):
        graph = graph[:, -1] if graph.ndim == 4 else graph
        output = self.model(x[..., self.order].permute(0, 2, 1, 3),
                            torch.maximum(graph, self.correlation))
        return {"failure_logit": output["fault_logits"][:, 0], "root_logits": output["root_logits"]}


class EadroAuthor(nn.Module):
    def __init__(self, feature_dim, groups, nodes, history, hidden):
        super().__init__()
        upstream = load_author_module("eadro", "codes/model.py", patch=True)
        self.metric_indices = groups["metric"]
        self.log_indices = groups["log"]
        # The original TraceModel expects one latency channel. The other input
        # features remain available to methods whose author encoders accept them.
        self.trace_index = None
        self.model = upstream.MainModel(
            len(self.log_indices), len(self.metric_indices), nodes, "cpu",
            log_dim=16, fuse_dim=128, metric_hiddens=[hidden], metric_kernel_sizes=[2],
            trace_hiddens=[hidden], trace_kernel_sizes=[2], graph_hiddens=[hidden],
            attn_head=4, detect_hiddens=[hidden], locate_hiddens=[hidden],
            self_attn=True, chunk_lenth=history)

    def forward(self, x, graph):
        b, t, n, f = x.shape
        g = dgl_batch(graph)
        per_node = x.permute(0, 2, 1, 3).reshape(b*n, t, f)
        g.ndata["metrics"] = per_node[..., self.metric_indices]
        g.ndata["logs"] = per_node[..., self.log_indices].mean(dim=1)
        g.ndata["traces"] = per_node[..., self.trace_index:self.trace_index+1]
        z = self.model.encoder(g)
        # Author MainModel.forward has misspelled attributes and constructs
        # detection labels from root labels. Call its actual registered heads:
        # the project has independent business and root-valid labels.
        detect = self.model.detecter(z)
        return {"failure_logit": detect[:, 1] - detect[:, 0],
                "root_logits": self.model.localizer(z)}


class ARTAuthor(nn.Module):
    def __init__(self, feature_dim, nodes, hidden, dropout):
        super().__init__()
        upstream = load_author_module("art", "models/unified_representation/layers.py", patch=True)
        heads = 2 if nodes % 2 == 0 else 1
        self.model = upstream.AutoRegressor(nodes, heads, feature_dim, hidden, hidden,
                                           hidden, dropout, 1, 2, 1)

    def forward(self, x, graph):
        # All five observed minutes are used: first four predict observed t.
        # Its residual at t is a historical score for y(t+1), not a future residual.
        g = dgl_batch(graph)
        _, prediction = self.model(g, x[:, :-1])
        deviation = (prediction - x[:, -1]).square()
        severity = deviation.sum(dim=-1)
        top = severity.topk(min(3, severity.shape[1]), dim=1).indices
        system = deviation.gather(1, top[..., None].expand(-1, -1, deviation.shape[-1])).sum(dim=1)
        root = F.cosine_similarity(deviation, system[:, None], dim=-1)
        return {"risk_score": system.sum(dim=-1), "root_logits": root,
                "reconstruction_loss": deviation.mean(), "telemetry_prediction": prediction}


def build_framework(method, data, hidden=64, dropout=0.1, top_k=2):
    d = data.dataset
    if method == "ltfp":
        protocol = load_author_module("ltfp", "src/ltfp/protocol.py")
        signals = abs(d.features[data.input_fit_indices]).max(axis=-1)
        _, correlation = protocol.combine_topology_and_correlation(
            torch.eye(len(d.services)).numpy(), signals, top_k)
        return LTFPAuthor(len(d.feature_columns), data.groups, hidden, dropout, correlation)
    if method == "eadro":
        model = EadroAuthor(len(d.feature_columns), data.groups, len(d.services), d.history, hidden)
        model.trace_index = d.feature_columns.index("trace_duration_mean_ms")
        return model
    if method == "art":
        return ARTAuthor(len(d.feature_columns), len(d.services), hidden, dropout)
    if method == "instantops":
        from models.instantops import InstantOpsGNNGRU
        return InstantOpsAdapter(InstantOpsGNNGRU(len(d.feature_columns), hidden, hidden,
                                                 dropout=dropout))
    raise ValueError(f"Unsupported framework: {method}")


class InstantOpsAdapter(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, graph):
        out = self.model(x, graph)
        logit = out["failure_logits"]
        return {"failure_logit": logit[:, 1] - logit[:, 0]}


@torch.no_grad()
def permutation_scores(model, x, graph, *, repeats=16, seed=42):
    """Single-window temporal block permutation; signed risk drop, graph fixed.

    This is a declared paper-based variant: the paper's unreleased exact
    permutation implementation is not asserted. Repeats do not depend on batch
    composition, so one sample's root rank is invariant to evaluation batching.
    """
    if repeats < 1 or x.shape[1] < 2:
        raise ValueError("Permutation attribution needs repeats>=1 and history>=2")
    was_training = model.training
    model.eval()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    permutations = [torch.randperm(x.shape[1], generator=generator).to(x.device)
                    for _ in range(repeats)]
    try:
        original = model(x, graph)["failure_logit"].sigmoid()
        scores = []
        for service in range(x.shape[2]):
            perturbed = []
            for perm in permutations:
                xp = x.clone()
                xp[:, :, service] = x[:, perm, service]
                perturbed.append(model(xp, graph)["failure_logit"].sigmoid())
            scores.append(original - torch.stack(perturbed).mean(dim=0))
        return torch.stack(scores, dim=1)
    finally:
        model.train(was_training)
