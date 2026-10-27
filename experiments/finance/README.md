# Finance: rare-drop ranking

This is a retained study with its own recorded protocol and source version.
See the [experiment index](../README.md) for how it relates to the latest runs.

Predict whether an S&P 500 stock falls at least 5% over the next five observed
trading dates, at Friday's close. Data comes from the public
[S&P 500 snapshot](https://www.kaggle.com/datasets/andrewmvd/sp-500-stocks).
A temporal split reserves a final period and purges overlapping forward-label
windows at both boundaries. Preparation and splits match the previous study.

Local Qwen3.8-27B in BF16 with `xhigh` reasoning selected experiments within a
three-trial cap, after one simulated model approval. Every trial used the same
Random Forest / LightGBM families and development split.

| Trial | RF change | RF validation AP | Unchanged LightGBM AP |
|---|---|---:|---:|
| 1 | Baseline: 32 trees, depth 8 | 0.139892 | 0.127045 |
| 2 | Explicit balanced class weighting | 0.138458 | 0.127045 |
| 3 | 400 trees, depth 8, no class weighting | 0.138822 | 0.127045 |

The baseline remained selected: reserved-period AP 0.107625 and ROC-AUC 0.618164.
Independent checkpoint replay confirmed all predictions and requested parameter
changes. Both RF interventions changed all validation probabilities; unchanged
control scores were reproducible.

The LLM nevertheless inferred a feature ceiling and treated failed experiments
as evidence for temporal shift. Those interpretations are unsupported. The
measurements establish only that these two configurations did not improve AP on
this split. This motivated prospective experimental questions and code-generated
findings in the current harness.

This is one dataset and seed, with constituent survivorship and historical
final-period reuse. It does not establish trading value, calibrated event
probabilities or statistical significance.

[Complete public conversation](conversation.md) · [Native messages](messages.json)
· [Scores, configurations and original notes](record.json)
