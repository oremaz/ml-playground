# Reasoning check on fixed finance measurements

This is a retained study with its own recorded protocol and source version.
See the [experiment index](../README.md) for how it relates to the latest runs.

The earlier agent concluded that unsuccessful class weighting and tree-count
changes suggested a feature ceiling or temporal shift. Those experiments do not
identify either cause. Its review schema also required a hypothesis when
stopping, encouraging a retrospective explanation after the trial budget ended.

The revised prompt asks prospective questions for further experiments, allows
an empty hypothesis when stopping, and asks the agent to state unresolved
uncertainty. Observed improvement or lack of improvement is generated from saved
scores in code. It explicitly distinguishes ranking metrics from decision
thresholds and rejects causal interpretations of unsuccessful tuning.

A live local-Qwen `xhigh` replay used the same recorded finance measurements in
a shorter, fact-only context. No new models were fitted or evaluated. The new
note states that the two tested changes did not improve AP and that other
settings, features, distribution shift and sampling variation remain untested.
It does not infer a ceiling. This is a diagnostic example, not a blinded prompt
ablation or proof that unsupported explanations have been eliminated.

[Original and revised notes](replay.json) · [Complete replay](conversation.md)
· [Native messages](messages.json)
