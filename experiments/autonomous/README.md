# CSV agent evaluations

Completed 2026-10-20 with local Qwen3.8-27B, BF16 and xhigh reasoning; one simulated model-family approval
and up to five comparisons per task. Every case starts from its original CSVs
and objective in a new workspace. Sources remain fixed throughout each search.

Selection uses three frozen development folds. The reserved evaluation uses the
saved fit from the last development fold, without refitting on final data.

| Task | Metric | Development: initial → selected | Reserved: initial → selected | Reserved constant |
|---|---|---:|---:|---:|
| [Building energy](energy/conversation.md) | RMSE ↓ | 98.63 → 93.27 | 94.93 → 105.49 | 91.06 |
| [Bike sharing](bikes/conversation.md) | RMSE ↓ | 78.54 → 78.33 | 70.02 → 65.89 | 241.07 |
| [Wearable activity](activity/conversation.md) | Macro F1 ↑ | 0.94833 → 0.94983 | 0.86884 → 0.86369 | 0.05331 |
| [Finance drop risk](finance/conversation.md) | Average precision ↑ | 0.16223 → 0.16223 | 0.09406 → 0.09406 | 0.06466 |

Initial means the best candidate from trial 1; selected means the candidate chosen
using development scores. Bikes improved on reserved data. Finance retained its
initial linear model. The energy and activity development gains did not transfer;
the selected energy model also lost to the constant reference on reserved data.

All four searches completed without operator intervention, with multiple predictive
families in later trials and up to three concurrent fits. Energy stopped after four
trials; the other tasks used five. The [action and explanation review](explanation_review.md)
checks the conversations separately from predictive gains. It identifies a proposed
energy interaction test whose generated columns were excluded from the model inputs,
and unsupported claims in conversational prose. Execution checks passing does not
certify that a trial tests the LLM's stated hypothesis.

These are single-seed retrospective experiments on previously inspected datasets.
Discovery sees whole-upload profiles before splitting. Repeated validation queries
can be optimistic; these runs do not establish significance, calibrated probabilities,
deployment value or monotonic gains from larger trial budgets.

The supplied objectives specify the targets and allowed inputs. Tasks are virtual
metering in one house from same-interval room sensors; hourly bike
demand from calendar inputs and exactly previous-hour weather; activity prediction
for held-out people; Friday stock-drop ranking with five-observation forward labels.
Temporal label boundaries are purged, and transformations fit on training rows.

Each linked conversation has accompanying native messages, preparation code,
trial settings, notes, final scores and independent replay checks. Compressed
request/execution traces retain tool errors and automatic recovery. Raw data and
fitted checkpoints stay in the local workspace. See [summary](summary.json),
[environment](environment.json), [datasets](datasets.json), [protocol](run_protocol.json)
and [validation](validation.json).

The independent checks reproduce predictions, all fold scores, custom diagnostics,
model parameters, preprocessing, final selection, split boundaries and model approval.
The [action checks](action_review.json) additionally compare 15 proposed next-trial
configurations with execution across 19 trials and verify 46 answered native calls.
These checks do not certify the correctness of the LLM prose. Recorded source hashes
identify the evaluated code; the current interface also normalizes missing `custom_`
metric prefixes automatically.
