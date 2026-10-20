# Action and explanation review

Review of all four completed searches, 2026-10-20. The scope includes the observable
assistant messages, native arguments and results, structured notes and actual trial
configurations. Private model thinking is excluded. Predictive scores are assessed
separately in the [results table](README.md).

## Overall assessment

The searches contain useful, task-relevant experiments, but conversational coherence
is incomplete. All 15 next-trial configurations match the corresponding execution;
that agreement does not establish that the arguments implement the stated hypothesis.
Energy contains one such mismatch. Unconstrained assistant prose also makes claims
that the more cautious structured notes generally avoid. The logs support a bounded,
auditable workflow, not a fully reliable autonomous scientific interpretation.

The [mechanical review](action_review.json) covers 19 trial records, one model-family
approval per dataset and 46 answered native calls. Every search preserved its fixed
target, primary metric and development folds and retained its validation winner.
Reserved results were evaluated after selection, without returning them to the LLM.

## Energy

Regularizing the trees after observing large training/validation gaps was a reasonable
experiment. The unchanged linear control stayed at RMSE 98.63 while the best tree
improved from the initial tree comparisons to 93.28. The subsequent T6/RH_6 ablation
kept the regularized parameters fixed; its XGBoost change to 93.27 was negligible.
The evening mean-error diagnostic was relevant to the objective, and the LLM correctly
read its positive sign as average overestimation on the evaluated evening rows.

The [trial-4 proposal](energy/conversation.md#21-assistant) asked for eight T×RH
products but explicitly selected only the 21 existing features. The products were
created and excluded from model inputs. Trial 4 therefore had the same configuration
as trial 3; replaying the saved prediction files finds differences no larger than
5.7e-14. This did not test interactions. The LLM failed to notice and subsequently
described the interaction experiment as unsuccessful. The numerical replay audit
passed because execution matched the arguments; it did not check this semantic mismatch.

The [closing exchange](energy/conversation.md#23-assistant) first promises a humidity
ablation, then stops at four of five trials and mentions an untested calendar ablation.
Early stopping is allowed, but this unexplained planning change is incoherent.
The selected model's reserved RMSE, 105.49 versus the constant's 91.06, further prevents
claiming a useful final predictor from this search.

## Bike sharing

The sequence is intelligible: increase tree counts, add weather/commuter interactions,
ablate three low-importance features, then test a boosting-parameter bundle on the
restored feature set. The three interaction columns were actually selected in trial 3.
Unchanged linear controls and the explicit baseline resets make the comparisons
interpretable against the relevant earlier configurations. The commuter-hour RMSE
diagnostic follows the supplied objective, rather than inventing a business metric.

However, the [response after trial 2](bikes/conversation.md#16-assistant) says
"capacity isn't the bottleneck". One unsuccessful tree-count comparison cannot
establish that. The structured note is appropriately narrower. The final boosting
trial changes several settings together, so its gain cannot be assigned to one setting.
The closing note correctly describes a small validation gain and the remaining high
commuter-hour error, about 139.5 versus overall RMSE 78.33.

## Wearable activity

The agent uses disjoint participant folds and relevant sitting/standing recalls.
It tests tree count, regularization, a 112-feature subset, then class weighting.
After the subset reduced macro F1 to 0.91511, it restores all 561 features. The final
weighting comparison retains the trial-2 tree settings and changes the weighting flag;
it does not accidentally retain the weaker feature subset. Selecting trial 2 after
later failures is coherent, despite the reserved-participant score decreasing.

The [subset rationale](activity/conversation.md#21-assistant) calls gravity, angle and
magnitude features more person-invariant, but invariance was not measured. Feature
importance can motivate an ablation; it does not establish that interpretation.
Claims that training/validation gaps identify a generalization mechanism are similarly
too strong. The structured notes usually frame the changes as questions instead.

## Finance

The sequence remains useful even though the first-trial linear model wins:
regularize the trees, remove index_ret_21 with baseline parameters, combine that
ablation with the regularization settings, then disable class weighting with other
settings held fixed. LightGBM AP rises from 0.10791 to 0.15398 before the final weighting
test, while remaining below the initial linear AP of 0.16223. Retaining the initial
winner is correct, rather than treating unchanged selection as an execution failure.

The [baseline narrative](finance/conversation.md#11-assistant) incorrectly says all
trees are at or below the constant: their mean AP values are 0.10791, 0.10491 and
0.12107, each above the constant's 0.10303. The first structured note also overstates
what regularization can distinguish by referring to a training/validation gap as a
possible "binding constraint". A configuration comparison does not identify that.

The [weighting rationale](finance/conversation.md#19-assistant) motivates an AP
experiment partly from precision/recall at the default threshold. Testing weighting
is valid, but those threshold metrics do not demonstrate a ranking problem or explain
average precision. The closing note's 0.1613 linear comparison refers to the reduced
feature control; the actual retained incumbent is 0.16223 from trial 1.

## Limits

Several comparisons change parameter bundles, and the exact reference trial is not
always clear in prose. Generated findings say "on this split" despite using three
frozen folds. These are wording and interpretation weaknesses, not score corrections.
Valid metrics do not validate an explanation, and a failed experiment can still be
informative when its intended change was actually executed.

The conversations retain metric-name errors and automatic corrections. The current
interface adds missing `custom_` prefixes; that change was tested after these frozen
searches and is not credited with their behavior. The datasets were previously
inspected and no matched old/new harness ablation was run. This review cannot establish
that the harness or extra reasoning caused better predictions or better reasoning.
