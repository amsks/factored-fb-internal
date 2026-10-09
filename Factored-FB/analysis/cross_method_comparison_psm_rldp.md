# Cross-method comparison — PSM + RLDP folded in (2026-06-10)

Tuned 1M×10 state runs (`psm_state_tuned ortho1 lr_phi1e-5`, `rldp_state_tuned ortho0.001 lr_b1e-5`)
probed with the full suite and placed alongside the existing paper methods.
**[fresh]** = computed this session (10 seeds); **[paper]** = from `PAPER/rlbrew/content/results.tex`
(prior sessions / figures). All on cube-single-play state.

## 1. Phase funnel — reach → secure(control) → success, + S0 failure attribution

| method | reached | secured | success | fail@transport | fail@grasp | src |
| :-- | :-: | :-: | :-: | :-: | :-: | :-- |
| PSM  | 100% | 87.9% | 28.6% | 59.3% | 12.1% | [fresh] |
| RLDP | 100% | 85.1% | 26.2% | 58.9% | 14.9% | [fresh] |
| GCIQL / FB / CRL | ~100% | high | — | dominant | negligible | [paper] fig_funnel; FB−GCIQL deficit = +0.2 approach / +2.4 grasp / −14.6 maintain / −7.8 transport |

Both new FB-family methods match the paper thesis: approach solved, failures land **after control acquisition** (transport/maintain), not at grasp.

## 2. Counterfactual value rank-AUC (does the agent's value rank its own grasps by success?)

| method | AUC | src |
| :-- | :-: | :-- |
| GCIQL | 0.89 | [paper] |
| CRL   | 0.68 | [paper] |
| FB    | ~0.60 | [paper] |
| **PSM**  | **0.516 ± 0.021** | [fresh] |
| **RLDP** | **0.489 ± 0.025** (paper quoted 0.52) | [fresh] |

The FB-family methods (incl. PSM) sit at chance: value is nearly blind to whether a grasp will be kept.

## 3. Factorization — is transport distance d recoverable from the backward rep B? (verdict B2)

| method | R²(B→d) linear | R²(B→d) mlp | raw-state ceiling | verdict | src |
| :-- | :-: | :-: | :-: | :-- | :-- |
| PSM  | 0.91 | 0.94 | 0.96 | B2: info in B, failure downstream | [fresh] |
| RLDP | 0.69 | 0.82 | 0.96 | B2: info in B (lossier), failure downstream | [fresh] |
| FB   | fwd-map readout 0.70 > raw ceiling; bilinear F·z reads only 0.61 | | | readout failure | [paper] |

The basis encodes controllability/transport distance; the **bilinear value readout throws it away**.

## 4. Matched-observation aliasing — value ranks controllable vs about-to-fail look-alikes (0.5 = chance)

| method | approach | pickup | carry | src |
| :-- | :-: | :-: | :-: | :-- |
| PSM  | 0.62 | 0.61 | 0.61 | [fresh] |
| RLDP | 0.60 | 0.58 | 0.60 | [fresh] |
| FB (bilinear value) | — | — | ~0.61 | [paper] |

Even in the carry regime, value barely separates look-alike controllable vs failing states.

## Takeaway
PSM and RLDP behave like the rest of the FB family: a backward/forward basis that **linearly encodes** the
interaction/transport variable (factorization B2, R² up to 0.91 for PSM), but a bilinear value readout that
**aliases** controllable vs failing states (~0.61) and is **chance-level** at ranking counterfactual grasps
(AUC ~0.5). Behaviorally this surfaces as transport-dominated failure. This corroborates the paper's
"readout, not basis" framing and the readout-ablation line ([[psm-readout-ablation]]).

Artifacts: `analysis/profiles/{psm,rldp}_phase/aggregate/`, `analysis/factorization/{psm,rldp}/aggregate/`,
`analysis/value/repsep/cf_value_multiseed{,_table}.md`, `analysis/value/aliasing_psm_rldp/aggregate/`.
