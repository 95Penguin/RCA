"""Central model construction and model-specific training policy."""

from __future__ import annotations

from dataclasses import dataclass

from torch import nn

from .baselines import LastValueMLP, ServiceTCN
from .coarse_fine import (
    LTFPContextEvidenceCollaboration,
    LTFPIsolatedLocalAdapterEvidence,
    LTFPLocalAdapterEvidence,
    LTFPNullAnchoredEvidence,
    LTFPProtectedRootEvidence,
    LTFPProtectedRootSidecar,
    LTFPResidualEvidence,
    LTFPStateRelationEvidence,
    LTFPMetricBypass,
)
from .eadro import EadroAdapted
from .instantops import InstantOpsGNNGRU
from .joint_granularity import JointGranularityModel
from .ltfp import (
    LTFPBaseline,
    LTFPDualRouter,
    LTFPFaultTypeAux,
    LTFPForecastAux,
    LTFPNoGraph,
    LTFPRootCauseConsistent,
    LTFPSharedGraphExperts,
    LTFPTaskIndependentGraph,
    LTFPTransitionResidual,
)


MODEL_NAMES = (
    "joint-granularity",
    "joint-granularity-encoded-local",
    "joint-granularity-null-selection",
    "ltfp-raw-metric-bypass-evidence",
    "ltfp-metric-deviation-bypass-evidence",
    "last-mlp",
    "tcn",
    "gru",
    "gcn-gru",
    "instantops-adapted",
    "ltfp",
    "ltfp-type-aux",
    "ltfp-dual-router",
    "ltfp-task-graph",
    "ltfp-shared-graph-experts",
    "ltfp-no-graph",
    "ltfp-null-joint",
    "ltfp-null-marginal",
    "ltfp-residual-joint",
    "ltfp-residual-protected-joint",
    "ltfp-residual-marginal",
    "ltfp-local-adapter-joint",
    "ltfp-local-adapter-isolated-joint",
    "ltfp-protected-root-sidecar",
    "ltfp-context-evidence-collab",
    "ltfp-local-normal-evidence",
    "ltfp-state-relation-evidence",
    "ltfp-forecast",
    "ltfp-rc2",
    "ltfp-transition",
    "eadro-adapted",
)

FEATURE_MODALITIES = ("metric", "log", "trace")


@dataclass(frozen=True)
class ModelBuild:
    model: nn.Module
    checkpoint_config: dict[str, object]


@dataclass(frozen=True)
class TrainingPolicy:
    node_loss_weight: float
    rank_loss_weight: float
    rca_method: str


def named_feature_groups(feature_columns: tuple[str, ...]) -> dict[str, list[int]]:
    """Group telemetry columns using the shared panel naming convention.

    Entry-operation features are trace-derived business observations, so they
    remain in the trace group. Resource observation masks remain with metrics.
    """
    groups = {name: [] for name in FEATURE_MODALITIES}
    for index, name in enumerate(feature_columns):
        if name.startswith("log_"):
            groups["log"].append(index)
        elif name.startswith(("trace_", "entry_", "observed_entry")):
            groups["trace"].append(index)
        else:
            groups["metric"].append(index)
    return {name: indices for name, indices in groups.items() if indices}


def feature_groups(feature_columns: tuple[str, ...]) -> list[list[int]]:
    """Return non-empty metric, log and trace groups in a stable order."""
    named = named_feature_groups(feature_columns)
    return [named[name] for name in FEATURE_MODALITIES if name in named]


def forecast_targets(
    feature_columns: tuple[str, ...],
) -> list[tuple[int, int | None]]:
    """Return forecasted value columns and their optional observation masks."""
    names = {name: index for index, name in enumerate(feature_columns)}
    targets: list[tuple[int, int | None]] = []
    for index, name in enumerate(feature_columns):
        if name.startswith("observed_") or name == "log_observed":
            continue
        if name.startswith("metric_"):
            mask = names.get("observed_" + name.removeprefix("metric_"))
        elif name.startswith("log_"):
            mask = names.get("log_observed")
        elif name == "entry_p95_ms":
            mask = names.get("observed_entry_p95")
        elif name == "entry_degradation_score_rolling":
            mask = names.get("observed_entry_baseline")
        else:
            mask = None
        targets.append((index, mask))
    if not targets:
        raise ValueError("telemetry forecast requires at least one value feature")
    return targets


def build_model(
    name: str,
    *,
    feature_columns: tuple[str, ...],
    num_services: int,
    history: int = 5,
    num_fault_types: int = 0,
    joint_service_pooling: bool = False,
) -> ModelBuild:
    """Build one model under the shared input/output contract."""
    if name not in MODEL_NAMES:
        raise ValueError(f"unknown model: {name}")
    feature_dim = len(feature_columns)
    groups = feature_groups(feature_columns)
    if name in {"joint-granularity", "joint-granularity-encoded-local",
                "joint-granularity-null-selection"}:
        names = {column: i for i, column in enumerate(feature_columns)}
        values = [i for i, column in enumerate(feature_columns) if column.startswith("metric_")]
        masks = [names.get("observed_" + feature_columns[i].removeprefix("metric_")) for i in values]
        direct_deviation = name == "joint-granularity"
        model = JointGranularityModel(
            feature_dim, num_services, history, values, masks, groups,
            direct_failure_deviation=direct_deviation,
            allow_null_evidence=name == "joint-granularity-null-selection",
        )
        return ModelBuild(model, {"architecture": name, "feature_dim": feature_dim,
                                 "feature_groups": groups, "history": history,
                                 "num_services": num_services, "training": "from-scratch end-to-end",
                                 "metric_value_indices": values, "metric_mask_indices": masks,
                                 "normal_reference": "training-only system-negative windows",
                                 "direct_failure_deviation": direct_deviation,
                                 "allow_null_evidence": model.allow_null_evidence,
                                 "service_selection": ("N+1 attention with zero-valued null option"
                                                       if model.allow_null_evidence else "N service attention"),
                                 "capacity_control": ("64 additional zero-initialized key parameters; same shared initialization"
                                                      if model.allow_null_evidence else "same parameter shapes and initialization"),
                                 "checkpoint": "one full model selected by validation AP"})
    if name in {"ltfp-raw-metric-bypass-evidence", "ltfp-metric-deviation-bypass-evidence"}:
        names = {column: index for index, column in enumerate(feature_columns)}
        values = [i for i, column in enumerate(feature_columns) if column.startswith("metric_")]
        masks = [names.get("observed_" + feature_columns[i].removeprefix("metric_")) for i in values]
        model = LTFPMetricBypass(
            feature_dim=feature_dim, num_services=num_services, history=history,
            metric_value_indices=values, metric_mask_indices=masks,
            feature_groups=groups,
            use_normal_deviation=name == "ltfp-metric-deviation-bypass-evidence",
        )
        return ModelBuild(model, {"architecture": name, "feature_dim": feature_dim,
                                 "feature_groups": groups, "normal_deviation": model.use_normal_deviation})
    if name == "last-mlp":
        model = LastValueMLP(feature_dim=feature_dim)
        config: dict[str, object] = {
            "feature_dim": feature_dim,
            "architecture": name,
        }
    elif name == "tcn":
        model = ServiceTCN(feature_dim=feature_dim)
        config = {"feature_dim": feature_dim, "architecture": name}
    elif name in {
        "ltfp",
        "ltfp-type-aux",
        "ltfp-dual-router",
        "ltfp-task-graph",
        "ltfp-shared-graph-experts",
        "ltfp-no-graph",
        "ltfp-forecast",
        "ltfp-null-joint",
        "ltfp-null-marginal",
        "ltfp-rc2",
        "ltfp-transition",
        "ltfp-residual-joint",
        "ltfp-residual-protected-joint",
        "ltfp-residual-marginal",
        "ltfp-local-adapter-joint",
        "ltfp-local-adapter-isolated-joint",
        "ltfp-protected-root-sidecar",
        "ltfp-context-evidence-collab",
        "ltfp-local-normal-evidence",
        "ltfp-state-relation-evidence",
    }:
        if name == "ltfp":
            model = LTFPBaseline(feature_dim=feature_dim, feature_groups=groups)
        elif name == "ltfp-type-aux":
            model = LTFPFaultTypeAux(
                feature_dim=feature_dim,
                num_fault_types=num_fault_types,
                feature_groups=groups,
            )
        elif name == "ltfp-dual-router":
            model = LTFPDualRouter(feature_dim=feature_dim, feature_groups=groups)
        elif name == "ltfp-task-graph":
            model = LTFPTaskIndependentGraph(
                feature_dim=feature_dim, feature_groups=groups
            )
        elif name == "ltfp-shared-graph-experts":
            model = LTFPSharedGraphExperts(
                feature_dim=feature_dim, feature_groups=groups
            )
        elif name == "ltfp-no-graph":
            model = LTFPNoGraph(feature_dim=feature_dim, feature_groups=groups)
        elif name in {"ltfp-null-joint", "ltfp-null-marginal"}:
            model = LTFPNullAnchoredEvidence(
                feature_dim=feature_dim, feature_groups=groups
            )
        elif name == "ltfp-protected-root-sidecar":
            model = LTFPProtectedRootSidecar(
                feature_dim=feature_dim, feature_groups=groups
            )
        elif name == "ltfp-context-evidence-collab":
            model = LTFPContextEvidenceCollaboration(
                feature_dim=feature_dim, feature_groups=groups
            )
        elif name in {
            "ltfp-local-normal-evidence",
            "ltfp-state-relation-evidence",
        }:
            column_index = {
                column: index for index, column in enumerate(feature_columns)
            }
            metric_values = [
                index
                for index, column in enumerate(feature_columns)
                if column.startswith("metric_")
            ]
            metric_masks = [
                column_index.get(
                    "observed_"
                    + feature_columns[index].removeprefix("metric_")
                )
                for index in metric_values
            ]
            model = LTFPStateRelationEvidence(
                feature_dim=feature_dim,
                num_services=num_services,
                history=history,
                metric_value_indices=metric_values,
                metric_mask_indices=metric_masks,
                use_relation_graph=name == "ltfp-state-relation-evidence",
                feature_groups=groups,
            )
        elif name in {
            "ltfp-local-adapter-joint",
            "ltfp-local-adapter-isolated-joint",
        }:
            adapter_model = (
                LTFPIsolatedLocalAdapterEvidence
                if name == "ltfp-local-adapter-isolated-joint"
                else LTFPLocalAdapterEvidence
            )
            model = adapter_model(
                feature_dim=feature_dim, feature_groups=groups
            )
        elif name in {
            "ltfp-residual-joint",
            "ltfp-residual-protected-joint",
            "ltfp-residual-marginal",
        }:
            evidence_model = (
                LTFPProtectedRootEvidence
                if name == "ltfp-residual-protected-joint"
                else LTFPResidualEvidence
            )
            model = evidence_model(
                feature_dim=feature_dim, feature_groups=groups
            )
        elif name == "ltfp-transition":
            model = LTFPTransitionResidual(
                feature_dim=feature_dim, feature_groups=groups
            )
        elif name == "ltfp-rc2":
            model = LTFPRootCauseConsistent(
                feature_dim=feature_dim, feature_groups=groups
            )
        else:
            targets = forecast_targets(feature_columns)
            model = LTFPForecastAux(
                feature_dim=feature_dim,
                forecast_dim=len(targets),
                feature_groups=groups,
            )
            model.forecast_target_indices = tuple(index for index, _ in targets)
            model.forecast_observation_indices = tuple(mask for _, mask in targets)
        task_relations = {
            "ltfp": "task-aligned single-horizon architecture",
            "ltfp-type-aux": (
                "LTFP with positive-window fault-type auxiliary supervision"
            ),
            "ltfp-dual-router": (
                "LTFP with incident-adaptive task-specific modality routing"
            ),
            "ltfp-task-graph": (
                "LTFP with task-independent graph interaction experts"
            ),
            "ltfp-shared-graph-experts": (
                "capacity-matched LTFP with shared graph experts"
            ),
            "ltfp-no-graph": "temporal-only LTFP without graph interaction",
            "ltfp-null-joint": (
                "null-anchored service evidence with fine-grained root supervision"
            ),
            "ltfp-null-marginal": (
                "null-anchored service evidence with marginal failure supervision only"
            ),
            "ltfp-residual-joint": (
                "LTFP with zero-initialized root-supervised service-evidence residual"
            ),
            "ltfp-residual-protected-joint": (
                "root-supervised residual evidence with protected shared backbone"
            ),
            "ltfp-residual-marginal": (
                "LTFP with zero-initialized marginal service-evidence residual"
            ),
            "ltfp-local-adapter-joint": (
                "protected global prediction with a root-adaptable local evidence space"
            ),
            "ltfp-local-adapter-isolated-joint": (
                "fully protected global backbone with an isolated local evidence space"
            ),
            "ltfp-protected-root-sidecar": (
                "unchanged LTFP prediction path with a detached task-specific RCA sidecar"
            ),
            "ltfp-context-evidence-collab": (
                "protected global-context-guided collaboration with local root evidence"
            ),
            "ltfp-local-normal-evidence": (
                "protected metric normal-deviation RCA with an identity relation control"
            ),
            "ltfp-state-relation-evidence": (
                "protected metric normal-deviation RCA with separate graph relation context"
            ),
            "ltfp-forecast": "LTFP with dense next-minute telemetry forecasting",
            "ltfp-rc2": (
                "LTFP with root-cause-consistent service-risk aggregation"
            ),
            "ltfp-transition": (
                "LTFP with latest-state prior plus causal temporal residual"
            ),
        }
        config = {
            "feature_dim": feature_dim,
            "architecture": name,
            "paper_task_relation": task_relations[name],
            "feature_groups": groups,
        }
        if name == "ltfp-type-aux":
            config.update({
                "fault_type_classes": num_fault_types,
                "fault_type_supervision": (
                    "cross-entropy on future user-visible failure windows only"
                ),
            })
        elif name == "ltfp-dual-router":
            config.update({
                "modality_names": list(named_feature_groups(feature_columns)),
                "routing": (
                    "separate incident-adaptive soft routes for failure prediction and RCA"
                ),
                "shared_components": (
                    "modality encoders, fusion projection, graph layers, and graph pooling"
                ),
            })
        elif name in {"ltfp-task-graph", "ltfp-shared-graph-experts"}:
            config.update({
                "graph_experts": 2,
                "graph_assignment": (
                    "failure expert 1; RCA expert 2"
                    if name == "ltfp-task-graph"
                    else "same mean of experts 1 and 2 for both tasks"
                ),
                "capacity_control": (
                    "the two variants have identical graph-expert parameter counts"
                ),
            })
        elif name == "ltfp-no-graph":
            config["graph_layers"] = 0
        elif name in {"ltfp-null-joint", "ltfp-null-marginal"}:
            config.update({
                "joint_evidence_space": "no-failure anchor plus per-service evidence",
                "service_aggregation": "temperature-scaled log-mean-exp",
                "root_supervision": name == "ltfp-null-joint",
            })
        elif name == "ltfp-protected-root-sidecar":
            config.update({
                "modality_names": list(named_feature_groups(feature_columns)),
                "root_supervision": True,
                "root_gradient_to_shared_backbone": False,
                "root_sidecar": ["modality_router", "fusion", "graph_layers"],
                "coarse_prediction_path": "unchanged LTFP",
            })
        elif name == "ltfp-context-evidence-collab":
            config.update({
                "root_supervision": True,
                "root_gradient_to_shared_backbone": False,
                "failure_gradient_to_root_evidence": False,
                "coarse_prediction_path": "LTFP plus zero-initialized residual",
                "interaction": "global-context-guided local evidence selection",
                "root_checkpoint_selection": "validation-event-mrr",
            })
        elif name in {
            "ltfp-local-normal-evidence",
            "ltfp-state-relation-evidence",
        }:
            config.update({
                "metric_value_columns": [
                    feature_columns[index] for index in model.metric_value_indices
                ],
                "normal_reference": "train-only fully healthy history and target",
                "root_gradient_to_shared_backbone": False,
                "coarse_prediction_path": "unchanged LTFP",
                "relation_graph": (
                    "trace-derived observed adjacency"
                    if model.use_relation_graph
                    else "identity capacity control"
                ),
                "root_checkpoint_selection": "validation-event-mrr",
            })
        elif name in {
            "ltfp-local-adapter-joint",
            "ltfp-local-adapter-isolated-joint",
        }:
            config.update({
                "service_evidence_auxiliary": "normal/failure discrimination",
                "residual_initialization": 0.0,
                "root_supervision": True,
                "root_gradient_to_shared_backbone": False,
                "fine_grained_heads": ["predictive_anomaly", "root_responsibility"],
                "local_adapter_shared_between_fine_heads": True,
                "fine_gradient_to_shared_backbone": (
                    name != "ltfp-local-adapter-isolated-joint"
                ),
            })
        elif name in {
            "ltfp-residual-joint",
            "ltfp-residual-protected-joint",
            "ltfp-residual-marginal",
        }:
            config.update({
                "service_evidence_auxiliary": "normal/failure discrimination",
                "residual_initialization": 0.0,
                "root_supervision": name != "ltfp-residual-marginal",
                "root_gradient_to_shared_backbone": (
                    name != "ltfp-residual-protected-joint"
                ),
            })
        elif name == "ltfp-transition":
            config["failure_logit_decomposition"] = (
                "latest-minute state prior + full-history temporal residual"
            )
        elif name == "ltfp-rc2":
            config.update({
                "prediction_localization_link": (
                    "normalized log-sum-exp over the same service risk logits used for RCA"
                ),
                "service_risk_evidence": (
                    "local state + graph-propagated state + propagation difference"
                ),
                "risk_pooling_temperature": model.temperature,
            })
        elif name == "ltfp-forecast":
            config.update({
                "telemetry_forecast_horizon_minutes": 1,
                "telemetry_forecast_columns": [
                    feature_columns[index] for index, _ in targets
                ],
                "forecast_decision_path": (
                    "predicted service telemetry is fused into failure and RCA states"
                ),
            })
    elif name == "eadro-adapted":
        model = EadroAdapted(
            feature_dim=feature_dim,
            num_services=num_services,
            feature_groups=groups,
        )
        config = {
            "feature_dim": feature_dim,
            "architecture": name,
            "paper_task_relation": (
                "reactive detector/localizer architecture adapted to future labels"
            ),
            "feature_groups": groups,
        }
    else:
        gcn_layers = 0 if name == "gru" else 2
        model = InstantOpsGNNGRU(
            feature_dim=feature_dim,
            gcn_layers=gcn_layers,
            joint_service_pooling=joint_service_pooling,
        )
        config = {
            "feature_dim": feature_dim,
            "architecture": name,
            "gcn_layers": gcn_layers,
            "joint_service_pooling": joint_service_pooling,
        }
        if name == "instantops-adapted":
            config.update(
                {
                    "paper_task_relation": (
                        "two-layer GCN plus GRU future predictor; service masking "
                        "approximates the paper's permutation-based RCA"
                    ),
                    "locator_training": "none; failure-label training only",
                }
            )
    return ModelBuild(model=model, checkpoint_config=config)


def training_policy(
    name: str,
    *,
    node_loss_weight: float,
    rank_loss_weight: float,
    rca_method: str,
) -> TrainingPolicy:
    """Apply architecture-specific constraints in one auditable place."""
    if name == "instantops-adapted":
        return TrainingPolicy(0.0, 0.0, "masking")
    if name in {"joint-granularity", "joint-granularity-encoded-local",
                "joint-granularity-null-selection"}:
        return TrainingPolicy(0.0, rank_loss_weight, "node-head")
    if name == "ltfp-null-marginal":
        return TrainingPolicy(0.0, 0.0, rca_method)
    if name == "ltfp-null-joint":
        return TrainingPolicy(0.0, rank_loss_weight, rca_method)
    if name == "ltfp-residual-marginal":
        return TrainingPolicy(0.0, 0.0, rca_method)
    if name in {
        "ltfp-residual-joint",
        "ltfp-residual-protected-joint",
        "ltfp-local-adapter-joint",
        "ltfp-local-adapter-isolated-joint",
        "ltfp-protected-root-sidecar",
        "ltfp-context-evidence-collab",
        "ltfp-local-normal-evidence",
        "ltfp-state-relation-evidence",
    }:
        return TrainingPolicy(0.0, rank_loss_weight, rca_method)
    return TrainingPolicy(node_loss_weight, rank_loss_weight, rca_method)


__all__ = [
    "FEATURE_MODALITIES",
    "MODEL_NAMES",
    "ModelBuild",
    "TrainingPolicy",
    "build_model",
    "feature_groups",
    "named_feature_groups",
    "forecast_targets",
    "training_policy",
]
