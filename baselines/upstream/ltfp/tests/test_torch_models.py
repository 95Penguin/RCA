from __future__ import annotations

import sys
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ltfp.torch_models import (  # noqa: E402
    GraphAttentionLayer,
    LTFPConfig,
    LTFPModel,
    TemporalEncoder,
    lead_time_multitask_loss,
)


def _config(**overrides) -> LTFPConfig:
    values = {
        "input_dim": 4,
        "hidden_dim": 6,
        "n_fault_types": 3,
        "n_horizons": 3,
        "n_tcn_layers": 1,
        "n_graph_layers": 1,
        "dropout": 0.0,
        "modality_dims": (2, 2),
    }
    values.update(overrides)
    return LTFPConfig(**values)


def test_modality_mask_makes_unavailable_fill_values_irrelevant():
    torch.manual_seed(3)
    encoder = TemporalEncoder(_config()).eval()
    x = torch.randn(2, 2, 5, 4)
    modality_mask = torch.tensor(
        [
            [[True, False], [True, True]],
            [[False, True], [True, False]],
        ]
    )
    changed = x.clone()
    for batch_idx in range(x.size(0)):
        for node_idx in range(x.size(1)):
            if not modality_mask[batch_idx, node_idx, 0]:
                changed[batch_idx, node_idx, :, :2] = 1000.0 * torch.randn(5, 2)
            if not modality_mask[batch_idx, node_idx, 1]:
                changed[batch_idx, node_idx, :, 2:] = 1000.0 * torch.randn(5, 2)

    with torch.no_grad():
        original_output = encoder(x, modality_mask=modality_mask)
        changed_output = encoder(changed, modality_mask=modality_mask)

    torch.testing.assert_close(original_output, changed_output, rtol=0.0, atol=1e-6)


def test_single_modality_still_uses_shared_fusion_projection():
    torch.manual_seed(5)
    encoder = TemporalEncoder(
        _config(input_dim=2, modality_dims=(2,), n_tcn_layers=0, n_graph_layers=0)
    ).eval()
    fusion_calls = []
    hook = encoder.fusion.register_forward_hook(lambda _module, _inputs, output: fusion_calls.append(output))
    try:
        output = encoder(torch.randn(2, 3, 4, 2), modality_mask=torch.ones(2, 3, 1, dtype=torch.bool))
    finally:
        hook.remove()

    assert output.shape == (2, 3, 6)
    assert len(fusion_calls) == 1


def test_service_with_no_available_modality_has_finite_zero_state():
    encoder = TemporalEncoder(_config(n_tcn_layers=0)).eval()
    mask = torch.tensor([[[False, False], [True, False]]])
    with torch.no_grad():
        output = encoder(torch.randn(1, 2, 4, 4), modality_mask=mask)

    assert torch.isfinite(output).all()
    torch.testing.assert_close(output[0, 0], torch.zeros_like(output[0, 0]))


def test_weighted_gat_uses_edge_strength_and_inserts_self_loops():
    layer = GraphAttentionLayer(2, dropout=0.0, edge_weight_gamma=1.0).eval()
    with torch.no_grad():
        layer.linear.weight.copy_(torch.eye(2))
        layer.attn_src.zero_()
        layer.attn_dst.zero_()

    node_state = torch.tensor([[[1.0, 0.0], [0.0, 2.0]]])
    weak = torch.tensor([[0.0, 0.01], [0.0, 0.0]])
    strong = torch.tensor([[0.0, 1.0], [0.0, 0.0]])

    with torch.no_grad():
        weak_output, weak_attention = layer(node_state, weak, return_attention=True)
        strong_output, strong_attention = layer(node_state, strong, return_attention=True)

    assert weak_attention[0, 0, 1] < strong_attention[0, 0, 1]
    assert weak_attention[0, 1, 0] == 0
    torch.testing.assert_close(weak_attention[0, 1, 1], torch.tensor(1.0))
    assert not torch.allclose(weak_output, strong_output)


def test_monotonic_decoder_depends_on_graph_input():
    model = LTFPModel(_config(n_graph_layers=0, n_tcn_layers=0))
    with torch.no_grad():
        model.fault_base_head.weight.fill_(0.25)
        model.fault_base_head.bias.zero_()
        model.fault_delta_head.weight.fill_(0.1)
        model.fault_delta_head.bias.fill_(-0.5)

    graph_states = torch.stack([torch.zeros(6), torch.ones(6)])
    logits = model._monotonic_horizon_logits(
        model.fault_base_head, model.fault_delta_head, graph_states
    )

    assert logits.shape == (2, 3)
    assert not torch.allclose(logits[0], logits[1])
    assert torch.all(logits[:, 1:] > logits[:, :-1])


def test_default_model_trains_only_failure_and_all_node_root_heads():
    torch.manual_seed(7)
    model = LTFPModel(_config())
    assert model.type_head is None
    assert model.lead_base_head is None
    assert model.lead_delta_head is None
    assert model.lead_time_head is None
    assert model.root_prior_weight == 0.0
    assert model.enable_root_prior is False

    x = torch.randn(2, 3, 5, 4)
    adjacency = torch.zeros(3, 3)
    outputs = model(x, adjacency)

    assert outputs["fault_logits"].shape == (2, 3)
    assert outputs["root_logits"].shape == (2, 3)
    assert torch.isfinite(outputs["root_logits"]).all()
    assert outputs["type_logits"].shape == (2, 3)
    assert outputs["lead_logits"].shape == (2, 3)
    assert outputs["_type_head_enabled"] is False
    assert outputs["_auxiliary_heads_enabled"] is False

    labels = torch.tensor([[0.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    roots = torch.tensor([0, 2])
    loss = lead_time_multitask_loss(
        outputs,
        labels,
        type_labels=torch.tensor([1, 2]),
        root_labels=roots,
        horizons=(60, 300, 600),
        type_weight=10.0,
        lead_weight=10.0,
    )
    reference = lead_time_multitask_loss(
        outputs,
        labels,
        root_labels=roots,
        horizons=(60, 300, 600),
    )
    torch.testing.assert_close(loss, reference)
    loss.backward()
    assert model.fault_base_head.weight.grad is not None
    assert model.root_head[-1].weight.grad is not None


def test_optional_legacy_heads_can_be_enabled_explicitly():
    model = LTFPModel(_config(enable_type_head=True, enable_auxiliary_heads=True)).eval()
    with torch.no_grad():
        outputs = model(torch.randn(2, 3, 5, 4), torch.zeros(3, 3))

    assert model.type_head is not None
    assert model.lead_base_head is not None
    assert outputs["_type_head_enabled"] is True
    assert outputs["_auxiliary_heads_enabled"] is True
    assert outputs["type_logits"].shape == (2, 3)
    assert outputs["lead_logits"].shape == (2, 3)


def test_full_model_mask_and_opt_in_root_prior_ignore_masked_fill_values():
    torch.manual_seed(11)
    model = LTFPModel(_config(root_prior_weight=0.5, enable_root_prior=True)).eval()
    x = torch.randn(1, 3, 5, 4)
    mask = torch.tensor([[[True, False], [True, True], [False, True]]])
    changed = x.clone()
    changed[0, 0, :, 2:] = 999.0
    changed[0, 2, :, :2] = -999.0
    adjacency = torch.zeros(3, 3)

    with torch.no_grad():
        first = model(x, adjacency, modality_mask=mask)
        second = model(changed, adjacency, service_modality_mask=mask)

    torch.testing.assert_close(first["fault_logits"], second["fault_logits"], rtol=0.0, atol=1e-6)
    torch.testing.assert_close(first["root_logits"], second["root_logits"], rtol=0.0, atol=1e-6)


def test_direct_concatenation_ablation_respects_missing_modality_mask():
    torch.manual_seed(19)
    model = LTFPModel(_config(fusion_mode="concat", n_graph_layers=0)).eval()
    adjacency = torch.eye(3)
    base = torch.randn(2, 3, 5, 4)
    changed = base.clone()
    changed[:, :, :, 2:] = 99999.0
    mask = torch.tensor([[[True, False], [True, False], [True, False]]] * 2)
    with torch.no_grad():
        left = model(base, adjacency, modality_mask=mask)["fault_logits"]
        right = model(changed, adjacency, modality_mask=mask)["fault_logits"]
    torch.testing.assert_close(left, right)


def test_independent_horizon_ablation_has_an_unconstrained_head():
    model = LTFPModel(_config(monotonic_horizons=False))
    assert model.fault_independent_head is not None
    assert model.fault_delta_head is None
    output = model(torch.randn(2, 3, 5, 4), torch.eye(3))["fault_logits"]
    assert output.shape == (2, 3)
