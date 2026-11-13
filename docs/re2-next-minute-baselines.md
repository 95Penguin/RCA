# RE2 next-minute baselines (seed 42)

All values are unweighted mean±sample SD over five deployment-held-out folds.
Detection uses threshold 0.5; RCA is retrospective event-level ranking.
LTFP is task-aligned. Eadro-adapted ports Eadro's reactive architecture to the future label and is not an exact reproduction of Eadro's original task.

| Dataset | Model | Precision | Recall | F1 | AP | Hit@1 | Hit@3 | Hit@5 | MRR |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| RE2-OB | Last-MLP | 0.756±0.219 | 0.789±0.189 | 0.745±0.140 | 0.870±0.075 | 0.813±0.090 | 0.901±0.102 | 0.967±0.075 | 0.872±0.064 |
| RE2-OB | TCN | 0.765±0.185 | 0.771±0.170 | 0.750±0.125 | 0.859±0.094 | 0.726±0.126 | 0.967±0.075 | 0.967±0.075 | 0.841±0.074 |
| RE2-OB | GRU | 0.819±0.254 | 0.702±0.268 | 0.687±0.157 | 0.879±0.070 | 0.768±0.107 | 0.920±0.061 | 0.967±0.075 | 0.846±0.051 |
| RE2-OB | MTG (GCN+GRU) | 0.886±0.106 | 0.734±0.265 | 0.766±0.160 | 0.910±0.060 | 0.390±0.099 | 0.893±0.098 | 0.913±0.084 | 0.631±0.066 |
| RE2-OB | InstantOps-adapted | 0.862±0.125 | 0.852±0.133 | 0.846±0.083 | 0.929±0.036 | 0.502±0.246 | 0.694±0.214 | 0.831±0.100 | 0.644±0.191 |
| RE2-OB | LTFP | 0.872±0.119 | 0.827±0.138 | 0.837±0.083 | 0.919±0.026 | 0.807±0.187 | 0.936±0.071 | 0.947±0.077 | 0.878±0.119 |
| RE2-OB | Eadro-adapted | 0.840±0.280 | 0.836±0.136 | 0.801±0.186 | 0.918±0.072 | 0.738±0.179 | 0.927±0.101 | 0.927±0.101 | 0.831±0.117 |
| RE2-TT | Last-MLP | 0.577±0.191 | 0.683±0.151 | 0.613±0.153 | 0.676±0.168 | 0.621±0.161 | 0.845±0.192 | 0.973±0.060 | 0.750±0.117 |
| RE2-TT | TCN | 0.598±0.165 | 0.648±0.241 | 0.615±0.189 | 0.683±0.190 | 0.682±0.165 | 0.957±0.060 | 1.000±0.000 | 0.814±0.093 |
| RE2-TT | GRU | 0.614±0.310 | 0.634±0.179 | 0.574±0.164 | 0.708±0.198 | 0.663±0.240 | 0.938±0.074 | 0.968±0.044 | 0.797±0.148 |
| RE2-TT | MTG (GCN+GRU) | 0.668±0.164 | 0.525±0.214 | 0.560±0.163 | 0.684±0.186 | 0.374±0.170 | 0.789±0.108 | 0.930±0.091 | 0.593±0.134 |
| RE2-TT | InstantOps-adapted | 0.596±0.278 | 0.612±0.244 | 0.562±0.200 | 0.671±0.172 | 0.298±0.088 | 0.456±0.219 | 0.620±0.281 | 0.434±0.139 |
| RE2-TT | LTFP | 0.765±0.135 | 0.784±0.158 | 0.771±0.130 | 0.846±0.127 | 0.894±0.204 | 0.987±0.030 | 0.987±0.030 | 0.932±0.131 |
| RE2-TT | Eadro-adapted | 0.650±0.164 | 0.637±0.121 | 0.624±0.068 | 0.727±0.143 | 0.552±0.139 | 0.956±0.042 | 1.000±0.000 | 0.750±0.086 |

## Baseline scope

- **LTFP** is a task-aligned architecture baseline: modality-specific temporal encoders, gated fusion, weighted graph attention, and graph/node multitask heads. It uses the common causal latest-minute dependency graph; it does not reproduce LTFP's separately fitted hybrid correlation graph.
- **InstantOps-adapted** uses the paper's two-layer GCN plus GRU next-step predictor and trains only on failure labels. Service masking is the deterministic service-level approximation of its permutation-importance RCA; the paper does not provide enough released preprocessing/code detail for a bit-exact reproduction.
- **Eadro-adapted** transfers Eadro's multimodal encoders, GLU fusion, GATv2-style propagation, attention pooling, and detector/localizer. Eadro originally performs current-state troubleshooting, so these numbers are not an exact Eadro task reproduction.
- BARO is excluded from this prediction table because its Bayesian change-point detector and RobustScorer are reactive detection/RCA components. Relabeling them as a next-minute predictor would no longer be a faithful BARO baseline.

## Immediate reading

- InstantOps-adapted has the best RE2-OB detection F1/AP, but its perturbation-based localization is much weaker than LTFP's supervised service ranker.
- LTFP is the strongest RE2-TT model for both detection and localization and gives the best localization balance on RE2-OB.
- Eadro-adapted improves over MTG detection on both datasets, but its localization is inconsistent: strong Hit@3/5, weaker Hit@1.
- The results use one seed. They establish runnable baselines, not final statistical significance; the paper table still needs multiple seeds or scenario-level confidence intervals.
