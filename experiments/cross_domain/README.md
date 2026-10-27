# Three complementary CSV tasks

This is a retained study with its own recorded protocol and source version.
See the [experiment index](../README.md) for how it relates to the latest runs.

The agent receives each CSV's bounded pandas profiles and an operational goal,
proposes a configuration, writes preparation code, and requests model approval.
One simulated user approves CPU Random Forest, LightGBM and XGBoost. The agent
then has up to five benchmark trials, including a three-model baseline, with
fixed development splits and a separate final partition. A trial can contain
several fits; diagnostic tool calls do not consume a benchmark trial.

## Tasks

- **Building energy:** estimate a missing appliance meter reading from room
  temperature, humidity and calendar information in the same ten-minute interval.
  This is a virtual meter, not a future-consumption forecast. The 19,735 readings
  come from one house. Lights, airport weather and random control columns are
  excluded; evaluation is chronological.
- **Bike sharing:** predict an hour's rental count at its start. Calendar inputs
  and exactly the previous hour's weather are available. A validated timestamp
  join preserves missing histories; current weather and target components are
  excluded. Labels become observable one hour later, with purging at both
  temporal boundaries. There are 17,379 observed hours; absent target hours are
  not invented or imputed.
- **Wearable activity:** classify six activities from 561 author-engineered
  smartphone sensor features. People, rather than windows, are separated across
  splits: 19 training, five validation and six final participants. This tests
  multiclass classification and transfer to held-out
  participants, complementing the two temporal regression tasks.

Dataset descriptions: [UCI energy](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction),
[UCI bike sharing](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset),
[UCI activity](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones).
Exact downloaded snapshots and conversion are recorded in [datasets.json](datasets.json)
and [prepare_data.py](prepare_data.py). UCI downloads were unavailable through the
VM proxy; verified public mirrors supplied the files.

## Measurements

Scores below compare the best model in the initial three-model trial with the
candidate selected on validation. Final scores were unavailable to the agent;
models are the saved inner-training fits, without a final-driven refit.

| Task | Metric | Validation: initial → selected | Reserved final: initial → selected |
|---|---|---|---|
| Energy | RMSE, Wh ↓ | 104.65 → 90.48 | 155.89 → 109.68 |
| Bike sharing | RMSE, rentals ↓ | 67.86 → 65.69 | 70.40 → 68.32 |
| Wearable activity | Macro F1 ↑ | 0.9505 → 0.9523 | 0.8607 → 0.8617 |

Energy improved against its initial LightGBM, but a constant estimated from
training labels scored **91.06 Wh RMSE on the final period**, better than the
selected model's 109.68. This run does not establish a useful energy predictor.
Bike sharing improved about 3% on the final period; its constant reference scored
241.08. Neither comparison establishes statistical significance.
Activity improved only slightly on the final participants. The validation winner
was already found in trial two; three further trials did not help. Its final
macro F1 was much lower than validation, illustrating uncertainty across people
rather than establishing a cause. A 44-feature subset worsened validation macro
F1 to 0.8916 and was discarded; subsequent trials restored all 561 features.

The energy agent added evening mean error (prediction minus observation); the
bike agent added RMSE during working-day commuter hours. Each explicit diagnostic
was applied to every saved trial and kept separate from the fixed selection
metric. Validation evening bias was overestimation, contrary to the user's
initial concern about underestimation. Bike commuter-hour validation RMSE fell
from 97.59 to 91.50; final commuter RMSE was 107.10.
The activity agent added sitting and standing recall. The selected model's
validation recalls were 0.9251 and 0.9905, but final recalls were 0.6865 and
0.8093. A candidate with higher sitting recall was not selected when
its primary macro F1 was lower; an operational preference would need review
before a new search.

## What the conversations show

The revised harness supplies code-generated measured findings, evidence IDs,
actual configuration differences and concise prospective questions. A
[finance replay](../reasoning/README.md) illustrates why unsuccessful tuning
does not identify a feature ceiling or distribution shift. More reasoning alone
is insufficient.

The bike conversation preserved two unsuccessful trials and retained the
incumbent. Most notes state bounded comparisons, but the second note still
suggests a “practical limit” without evidence, and the fourth overinterprets a
possible negative outcome. These original notes remain in the logs. The next
prompt revision explicitly limits alternatives to possible measured outcomes;
it is not retrospectively credited for the energy or bike responses.
Activity also speculated about “intrinsic” sensor overlap in its first note,
despite the stronger instruction. Its later notes were more restrained and its
stop note left causes unresolved. Unsupported hypotheses remain a limitation;
code-generated findings do not validate the surrounding LLM prose.

The discovery stage also asked unnecessary questions, including whether missing
bike targets should be imputed. The reviewed configuration and generated script
instead preserved observed targets and rows. Script review remains necessary.
Activity script generation exhausted its 16,384-token budget before producing
visible code. The failed request is preserved; the retry explicitly requests
minimal identity preparation without listing 561 feature names in source.
After its baseline, repeated wide profiles and evidence exhausted GPU memory.
The continuation compacts repeated inspection details in provider requests while
preserving the full original messages, scores, class labels, notes and tool-call
order. It resumes the existing search without a second model approval. Both
failures and [the context change](context_source_update.json) are logged.
An operator supplied the preparation retry instruction and patched context
compaction. The activity case is an assisted run, not an unattended success.
All three agents also corrected invalid diagnostic names from tool feedback;
those rejected calls did not consume benchmark trials.

## Records and reproduction

| Task | Full native conversation | Discovery and script generation | Scores, notes and audit |
|---|---|---|---|
| Energy | [conversation](energy/conversation.md), [messages](energy/messages.json) | [requests/responses](energy/discovery.json), [script](energy/preparation.py) | [scores](energy/result.json), [notes](energy/notes.json), [independent checks](energy/independent_checks.json) |
| Bike sharing | [conversation](bikes/conversation.md), [messages](bikes/messages.json) | [requests/responses](bikes/discovery.json), [script](bikes/preparation.py) | [scores](bikes/result.json), [notes](bikes/notes.json), [independent checks](bikes/independent_checks.json) |
| Wearable activity | [conversation](activity/conversation.md), [messages](activity/messages.json) | [requests/responses](activity/discovery.json), [script](activity/preparation.py) | [scores](activity/result.json), [notes](activity/notes.json), [independent checks](activity/independent_checks.json) |

Each task directory also contains native request/execution events, usage and
failure metadata, exact reviewed/trial configurations and partition indices.
Private model thinking is not recorded. Local workspaces retain uploaded CSVs,
full repeated request contexts, preprocessors and fitted checkpoints.

```bash
PYTHONPATH=. uv run python experiments/cross_domain/prepare_data.py
PYTHONPATH=. uv run python -m ml_pipeline.agent_domain_experiment \
  --data ml_pipeline/data/agent_domains --output results/agent_domains_replay
```

The live backend was pinned local `Qwen/Qwen3.8-27B`, BF16, `xhigh` reasoning,
greedy generation, on a Jean Zay H100. Models used CPU fits, seed 42, standard
train-fitted preprocessing, and an initial 100 trees with four threads.
[Provenance](provenance.json) records dependency versions and source hashes.
[Runtime update](runtime_update.json) records when constant-reference feedback
was added after energy, before bike fitting. [Activity changes](activity_source_update.json)
record the later prompt and diagnostic fixes, preserving the earlier versions
as diffs. This is a descriptive development study, not a controlled harness ablation.
The energy and bike plots were regenerated after search with task-appropriate
diagnostics; [the record](post_run_diagnostics.json) preserves both image hashes.
No original messages, predictions or selections were rewritten. The
[audit script](audit.py) replays saved checkpoints and predictions; it requires
the original local workspace artifacts and accepts `--run` for their run directory.

Single-seed, adaptively queried validation scores can be optimistic. Prefixes of
one five-trial conversation are not independent 1/3/5-budget experiments.
Discovery profiles describe the uploaded CSV before partitioning, including
whole-table aggregates. The loop withholds final scores and predictions; this
is not a fully sealed dataset-discovery evaluation.
Energy covers one house; bike observations span two calendar years; activity uses
a custom subject split of the combined published files, with inherited feature
engineering and normalization. No leaderboard or free-living deployment claim
is supported. Hosted providers and interactive browser behavior were not tested.

## Validation

The targeted agent, metrics, trace, proposal and forecasting checks passed:
**78 passed, one skipped** because skrub is not installed. The exact invocation
is saved in [cpu_validation.json](cpu_validation.json). Independent checkpoint
replays verified all scores, predictions, preprocessing, temporal purging,
subject separation, custom diagnostics and the single simulated model approval.
All three saved workspaces passed [Streamlit AppTest](streamlit_smoke.json),
including notes, diagnostic tables and final scores. This does not test a browser
session or hosted providers.

On the research VM, open `cross_domain_energy`, `cross_domain_bikes` or
`cross_domain_activity` in the app's saved-workspace sidebar. The aliases point
to the original workspaces under `results/agent_domains_20261018/live_v1/`.
