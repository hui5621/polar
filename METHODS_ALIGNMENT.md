# Manuscript-to-code alignment

This note maps the manuscript equations and Table 3 settings to the runnable package. Numerical values reported in Table 3 are centralized in `src/polar_route/config.py` as `PAPER`; implementation choices needed where the manuscript is qualitative are kept separately as `UNSPECIFIED`.

## Equation map

| Manuscript | Implementation |
| --- | --- |
| Eqs. (1)–(2), environmental and hard-feasibility fields | `src/polar_route/scene.py` loads the twelve fixed projected scenes and composes land, chart, ice, and concentration constraints. |
| Eqs. (3)–(4), segment direction, along-track current, and route cost | `src/polar_route/metrics.py` samples each segment at 0.35-cell spacing, length-weights its environmental values, projects current onto the route, and integrates the cost. |
| Eqs. (5)–(9), SDA resistance, power, SOG, travel time, and fuel | `src/polar_route/vessel.py` implements the Lindqvist-type resistance and 32-step speed solve. Resistance, power, and fuel rate use STW; travel time and total fuel use current-corrected SOG. |
| Eqs. (18)–(31), S-IACO search, repair, archive, and route selection | `src/polar_route/siaco.py` implements scene descriptors, the two pheromone branches, risk-aware evaporation, bounded repair, tolerance dominance, and representative-route ranking. |
| Eqs. (32)–(36), candidate preview, Mask/Bias, and Shadow gate | `src/polar_route/sag_ppo.py` builds the six semantic operators, applies the published preview and gate coefficients, and verifies the proposed action before execution. |
| Eqs. (37)–(40), proposed/executed PPO update | `src/polar_route/ppo.py` computes GAE, updates the Actor only for unchanged proposals, and trains the Critic on every verified transition. |
| Table 3 and experiment entry points | `src/polar_route/config.py` contains the published constants; `src/polar_route/train.py` provides the command-line runner and defaults to five seeds, 500 episodes, 420 decision steps, 30 ants, and 225 S-IACO iterations. |

The vessel evaluator adds the projected ocean-current component to STW to obtain SOG. It rejects samples with non-positive SOG and returns route travel time and fuel only for feasible routes.

## Settings whose numerical rule is not stated

The manuscript gives equations or qualitative rules for these items but does not provide every numerical mapping or loss definition. They are explicit and grouped under `UNSPECIFIED` rather than presented as Table 3 values:

- S-IACO's descriptor-to-profile mapping, route-cost weights, soft-risk threshold and scene-dependent budget, repair-window radii, pheromone exchange ratios, deposit constant, and Pareto tolerance/archive cap.
- SAG-PPO's local candidate-search weights, decision stride, bias-history increments, and the operator-specific gate adjustment vector `ζ` (initialized to zero because no values are reported).
- The 45-degree dynamic-turn violation count and the closed forms used for the auxiliary behavior-cloning, pairwise preview-ranking, route-boundary, action-margin, and KL losses. The action-margin threshold is separately configurable; it is not conflated with its Table 3 loss coefficient.

These defaults make the implementation deterministic and reproducible. They should not be interpreted as additional numerical values reported by the manuscript. The paper-result CSVs in `results/paper/` remain the supplied final data; the runner writes new runs to `results/runs/`.

## Commands

```bash
polar-train --algorithm sagppo --scene 2025-08-05 --seeds 0 1 2 3 4
polar-train --algorithm all --scene all --seeds 0 1 2 3 4
```

The first command runs S-IACO followed by SAG-PPO for one scene and five seeds. The second runs the full algorithm set across all twelve scenes.
