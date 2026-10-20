# Forecasting: foundation-model integration

Forecast the S&P 500 index five observed trading dates ahead from 128 past values.
All methods used the same three rolling validation origins and frozen checkpoint
revisions. Earlier observed validation values can enter later contexts; each
forecast excludes its own future outcomes. Point predictions use medians.

| Method | Validation MAE |
|---|---:|
| Last value | 54.69 |
| Seasonal naive, period 5 | 61.63 |
| Timestral-209M | 53.59 |
| Toto 2.0 313M | 54.57 |
| TimesFM 3.0 | 64.23 |
| Chronos-2 | 46.61 |

Chronos was selected and obtained MAE 28.24 on another three final windows.
Only the selected model was evaluated there. All four real adapters passed H100
checks for missing history, forecast shapes, finite/noncrossing quantiles and
batch independence. Scores were independently replayed from saved predictions.

This tests working integration, not general predictive superiority: each split
scores only 15 observations of one retrospective series, and pretraining overlap
is unknown. Other Toto sizes are registered but were not live-tested. Forecast
MAE does not assess drop-risk ranking or trading performance.

The local Qwen `xhigh` conversation has one simulated family approval and one
six-method trial. Its warning about sample size is appropriate; its claim that
all models' error rises with horizon is too broad and not endorsed.

[Complete public conversation](conversation.md) · [Native messages](messages.json)
· [Scores, configurations and original notes](record.json)
