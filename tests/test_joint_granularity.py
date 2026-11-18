import sys
import unittest
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from models.joint_granularity import JointGranularityModel


class JointGranularityTests(unittest.TestCase):
    def make(self):
        model = JointGranularityModel(6, 3, 5, (0,1), (2,3), [[0,1,2,3],[4],[5]], hidden_dim=8, dropout=0).eval()
        model.configure_normal_reference(torch.zeros(5,3,2), torch.ones(5,3,2))
        return model

    def test_both_final_losses_update_the_shared_model(self):
        x = torch.randn(2,5,3,6)
        x[...,2:4] = 1
        graph = torch.ones(2,5,3,3)
        for task in ("failure", "root"):
            model = self.make()
            self.assertTrue(all(p.requires_grad for p in model.parameters()))
            output = model(x, graph)
            loss = (output["failure_logits"].log_softmax(-1)[:,1] if task=="failure"
                    else output["node_logits"].log_softmax(-1)[:,0]).neg().mean()
            loss.backward()
            for parameter in (model.branches[0].project.weight, model.graph_layers[0].project.weight,
                              model.service_evidence[0].weight, model.pool[0].weight):
                self.assertIsNotNone(parameter.grad)
                self.assertGreater(float(parameter.grad.abs().sum()), 0)
            if task == "failure":
                self.assertGreater(float(model.context_query.weight.grad.abs().sum()), 0)
            torch.testing.assert_close(output["cross_granularity_attention"].sum(-1), torch.ones(2))
            self.assertEqual(output["failure_logits"].shape, (2,2))
            self.assertEqual(output["node_scores"].shape, (2,3))

    def test_encoded_local_only_removes_the_direct_failure_input(self):
        original = self.make()
        encoded = JointGranularityModel(
            6, 3, 5, (0,1), (2,3), [[0,1,2,3],[4],[5]],
            hidden_dim=8, dropout=0, direct_failure_deviation=False,
        ).eval()
        encoded.load_state_dict(original.state_dict(), strict=True)
        self.assertEqual(sum(p.numel() for p in original.parameters()),
                         sum(p.numel() for p in encoded.parameters()))
        x = torch.randn(2,5,3,6)
        x[...,2:4] = 1
        graph = torch.ones(2,5,3,3)
        captured = {}
        def capture(name):
            def hook(module, args):
                captured[name] = args[0].detach().clone()
            return hook
        first = original.failure_head.register_forward_pre_hook(capture("original"))
        second = encoded.failure_head.register_forward_pre_hook(capture("encoded"))
        try:
            before, after = original(x, graph), encoded(x, graph)
        finally:
            first.remove()
            second.remove()
        torch.testing.assert_close(captured["original"][:, :-6], captured["encoded"][:, :-6])
        self.assertGreater(float(captured["original"][:, -6:].abs().sum()), 0)
        self.assertEqual(float(captured["encoded"][:, -6:].abs().sum()), 0)
        for name in ("node_logits", "service_evidence_embeddings", "cross_granularity_attention"):
            torch.testing.assert_close(before[name], after[name])
        loss = -after["failure_logits"].log_softmax(-1)[:,1].mean()
        loss.backward()
        self.assertGreater(float(encoded.service_evidence[0].weight.grad[:, -6:].abs().sum()), 0)
        self.assertGreater(float(encoded.context_query.weight.grad.abs().sum()), 0)
        self.assertEqual(float(encoded.failure_head[0].weight.grad[:, -6:].abs().sum()), 0)

    def test_encoded_local_factory_and_checkpoint_reconstruction(self):
        from models.factory import build_model, training_policy
        columns = ("metric_cpu", "observed_cpu", "log_count", "trace_span_count")
        original = build_model("joint-granularity", feature_columns=columns, num_services=3)
        built = build_model("joint-granularity-encoded-local", feature_columns=columns, num_services=3)
        self.assertTrue(original.model.direct_failure_deviation)
        self.assertFalse(built.model.direct_failure_deviation)
        self.assertFalse(built.checkpoint_config["direct_failure_deviation"])
        restored = build_model(built.checkpoint_config["architecture"],
                               feature_columns=columns, num_services=3)
        restored.model.load_state_dict(built.model.state_dict(), strict=True)
        self.assertFalse(restored.model.direct_failure_deviation)
        policy = training_policy("joint-granularity-encoded-local", node_loss_weight=1,
                                 rank_loss_weight=0.5, rca_method="masking")
        self.assertEqual((policy.node_loss_weight, policy.rank_loss_weight, policy.rca_method),
                         (0, 0.5, "node-head"))

    def test_null_selection_can_reject_all_local_evidence(self):
        model = JointGranularityModel(
            6, 3, 5, (0,1), (2,3), [[0,1,2,3],[4],[5]],
            hidden_dim=8, dropout=0, direct_failure_deviation=False, allow_null_evidence=True,
        ).eval()
        with torch.no_grad():
            model.context_query.weight.copy_(torch.eye(8))
            model.service_key.weight.zero_()
            model.null_evidence_key.fill_(100)
        evidence = torch.ones(2,3,8)
        attention, null = model._service_attention(torch.ones(2,8), evidence)
        torch.testing.assert_close(attention.sum(-1) + null, torch.ones(2))
        self.assertTrue((null > 0.999).all())
        selected = (attention.unsqueeze(-1) * evidence).sum(1)
        self.assertLess(float(selected.detach().abs().max()), 1e-6)
        with torch.no_grad():
            model.null_evidence_key.fill_(-100)
        attention, null = model._service_attention(torch.ones(2,8), evidence)
        torch.testing.assert_close(attention, torch.full((2,3), 1/3))
        self.assertTrue((null < 1e-6).all())

    def test_null_selection_preserves_shared_initialization_and_root_output(self):
        def make(enabled):
            torch.manual_seed(42)
            model = JointGranularityModel(
                6, 3, 5, (0,1), (2,3), [[0,1,2,3],[4],[5]], hidden_dim=8,
                dropout=0, direct_failure_deviation=False, allow_null_evidence=enabled,
            ).eval()
            model.configure_normal_reference(torch.zeros(5,3,2), torch.ones(5,3,2))
            return model
        baseline, model = make(False), make(True)
        for name, value in baseline.state_dict().items():
            torch.testing.assert_close(value, model.state_dict()[name])
        self.assertEqual(sum(p.numel() for p in model.parameters()) -
                         sum(p.numel() for p in baseline.parameters()), 8)
        x = torch.randn(2,5,3,6)
        x[...,2:4] = 1
        graph = torch.ones(2,5,3,3)
        before, output = baseline(x,graph), model(x,graph)
        torch.testing.assert_close(before['node_logits'], output['node_logits'])
        torch.testing.assert_close(output['cross_granularity_attention'].sum(-1) +
                                   output['null_evidence_attention'], torch.ones(2))
        loss = -output['failure_logits'].log_softmax(-1)[:,1].mean()
        loss = loss - 0.5 * output['node_logits'].log_softmax(-1)[:,0].mean()
        loss.backward()
        self.assertGreater(float(model.null_evidence_key.grad.abs().sum()), 0)
        self.assertEqual([name for name,p in model.named_parameters() if p.grad is None], [])

    def test_null_selection_factory_checkpoint_roundtrip(self):
        from models.factory import build_model, training_policy
        columns = ('metric_cpu','observed_cpu','log_count','trace_span_count')
        built = build_model('joint-granularity-null-selection', feature_columns=columns, num_services=3)
        config = built.checkpoint_config
        self.assertTrue(config['allow_null_evidence'])
        self.assertFalse(config['direct_failure_deviation'])
        copy = build_model(config['architecture'], feature_columns=columns, num_services=3)
        copy.model.load_state_dict(built.model.state_dict(), strict=True)
        self.assertTrue(copy.model.allow_null_evidence)
        x = torch.randn(2,5,3,4)
        x[...,1] = 1
        graph = torch.ones(2,5,3,3)
        for model in (built.model,copy.model):
            model.configure_normal_reference(torch.zeros(5,3,1),torch.ones(5,3,1))
            model.eval()
        torch.testing.assert_close(built.model(x,graph)['failure_logits'],copy.model(x,graph)['failure_logits'])
        policy = training_policy('joint-granularity-null-selection',node_loss_weight=1,
                                 rank_loss_weight=0.5,rca_method='masking')
        self.assertEqual((policy.node_loss_weight,policy.rank_loss_weight,policy.rca_method),
                         (0,0.5,'node-head'))

    def test_missing_metric_deviations_and_checkpoint_roundtrip(self):
        model = self.make()
        x = torch.randn(2,5,3,6)
        x[...,2:4] = 0
        self.assertEqual(float(model._metric_deviation(x).abs().sum()), 0)
        copy = self.make()
        copy.load_state_dict(model.state_dict())
        graph = torch.ones(2,5,3,3)
        torch.testing.assert_close(model(x,graph)["node_scores"], copy(x,graph)["node_scores"])

    def test_combined_objective_reaches_every_trainable_parameter(self):
        model = self.make()
        x = torch.randn(2,5,3,6)
        x[...,2:4] = 1
        output = model(x, torch.ones(2,5,3,3))
        loss = -output["failure_logits"].log_softmax(-1)[:,1].mean()
        loss = loss - 0.5 * output["node_logits"].log_softmax(-1)[:,0].mean()
        loss.backward()
        unused = [name for name,p in model.named_parameters() if p.grad is None]
        self.assertEqual(unused, [])


if __name__ == "__main__":
    unittest.main()
