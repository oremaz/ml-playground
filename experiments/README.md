# Logged evaluations

The [latest CSV assistant evaluation](autonomous/README.md) uses local
Qwen3.8-27B, BF16 and xhigh reasoning, with one simulated model-family approval
per task and a budget of up to five trials.

| Conversation | Task | Completed trials |
|---|---|---|
| [Building energy](autonomous/energy/conversation.md) | Estimate appliance consumption from room sensors | 4 |
| [Bike sharing](autonomous/bikes/conversation.md) | Predict hourly demand from calendar and previous-hour weather | 5 |
| [Wearable activity](autonomous/activity/conversation.md) | Classify activity for held-out participants | 5 |
| [Finance drop risk](autonomous/finance/conversation.md) | Rank Friday decisions by five-observation drop risk | 5 |

Each record includes native messages, preparation code, trial configurations,
scores, execution traces and independent replay checks. Raw data and fitted
checkpoints remain in local workspaces.

See the [results and protocol](autonomous/README.md) and the
[action and explanation review](autonomous/explanation_review.md). Predictive
gains are mixed, and the review identifies mistakes even where execution checks
passed. These single-seed retrospective runs do not establish general agent
superiority or consistent gains from larger trial budgets.
