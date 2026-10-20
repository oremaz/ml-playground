"""Observable trial facts and public research notes, independent of LLM prose."""

import math


def trial_evidence(trial, metric):
    number = trial["number"]
    facts = [{"id": f"t{number}.status", "trial": number, "value": trial["status"],
              "description": f"Trial {number} status: {trial['status']}"}]
    if trial.get("error"):
        facts.append({"id": f"t{number}.error", "trial": number, "value": trial["error"],
                      "description": trial["error"][:600]})
    for row in trial.get("summary", []):
        method = f"{row['model']}/{row['augmentation']}"
        for key, value in row.items():
            if key in ("model", "augmentation") or not isinstance(value, (int, float)) or not math.isfinite(value):
                continue
            facts.append({"id": f"t{number}.{row['model']}.{row['augmentation']}.{key}",
                          "trial": number, "value": value,
                          "description": f"{method}: validation {key} = {value:.6g}"})
    for index, change in enumerate(trial.get("execution", {}).get("changes_from_incumbent", [])):
        description = (f"Executed feature selection: {len(change['before'])} -> {len(change['after'])} features; exact lists in execution feedback."
                       if change["setting"] == "features" else f"Executed {change['setting']}: {change['before']} -> {change['after']}")
        facts.append({"id": f"t{number}.change.{index}", "trial": number, "value": change,
                      "description": description})
    for candidate in trial.get("comparison", {}).get("candidates", []):
        for key in ("delta_from_same_method_baseline", "delta_from_previous_incumbent"):
            value = candidate.get(key)
            if value is not None:
                facts.append({"id": f"t{number}.{candidate['model']}.{candidate['augmentation']}.{key}",
                              "trial": number, "value": value,
                              "description": f"{candidate['model']}/{candidate['augmentation']}: {metric} {key} = {value:+.6g}"})
    for item in trial.get("diagnostics", {}).get("diagnostics", []):
        for row in item.get("validation_folds", []):
            for key, value in row.items():
                if key != "fold":
                    facts.append({"id": f"t{number}.{item['model']}.{item['augmentation']}.fold.{int(row['fold'])}.{key}",
                                  "trial": number, "value": value,
                                  "description": f"{item['model']}/{item['augmentation']}: fold {int(row['fold'])} validation {key} = {value:.6g}"})
        for index, row in enumerate(item.get("per_class", [])):
            for key in ("precision", "recall", "f1", "rows"):
                facts.append({"id": f"t{number}.{item['model']}.{item['augmentation']}.class.{index}.{key}",
                              "trial": number, "value": row[key],
                              "description": f"{item['model']}/{item['augmentation']}: validation class {row['label']} {key} = {row[key]:.6g}"})
        for key, value in item.get("training_scores", {}).items():
            facts.append({"id": f"t{number}.{item['model']}.{item['augmentation']}.training.{key}",
                          "trial": number, "value": value,
                          "description": f"{item['model']}/{item['augmentation']}: in-sample training {key} = {value:.6g}"})
    for item in trial.get("metric_results", []):
        for result in item["metrics"]:
            if result["value"] is not None:
                facts.append({"id": f"t{number}.{item['model']}.{item['augmentation']}.{result['name']}",
                              "trial": number, "value": result["value"],
                              "description": f"{item['model']}/{item['augmentation']}: validation {result['name']} = {result['value']:.6g} on {result['rows']} rows"})
    facts.append({"id": f"t{number}.metric_definition", "trial": number, "value": metric,
                  "description": ("Average precision uses probability rankings, independently of the decision threshold."
                                  if metric == "pr_auc" else f"Selection uses validation {metric}.")})
    return facts


def observed_finding(trial, metric):
    rows = [row for row in trial.get("summary", []) if row.get(metric) is not None
            and row.get("completed_folds") == row.get("expected_folds")]
    if not rows:
        return "No complete candidate was measured in this trial. The recorded failure does not assess model quality."
    score = (min if metric in ("mae", "rmse", "smape", "mase") else max)(row[metric] for row in rows)
    incumbent = trial.get("comparison", {}).get("incumbent_before")
    if incumbent is None:
        return f"Baseline comparison completed: best validation {metric} = {score:.6g}."
    improved = score < incumbent[metric] if metric in ("mae", "rmse", "smape", "mase") else score > incumbent[metric]
    return (f"This trial's best validation {metric} = {score:.6g}; previous best = {incumbent[metric]:.6g}. "
            + ("The measured selection score improved on this split." if improved else "No measured improvement over the incumbent on this split.")
            + " This comparison does not identify the cause or establish significance.")


def research_note(state, args):
    trial = args["trial"]
    if isinstance(trial, bool) or trial != len(state["trials"]):
        raise ValueError("Review the latest trial by its exact number")
    if any(note["trial"] == trial for note in state.get("notes", [])):
        raise ValueError("This trial already has a research note")
    facts = {fact["id"]: fact for item in state["trials"] for fact in trial_evidence(item, state["config"]["metric"])}
    references = args["evidence_ids"]
    if not isinstance(references, list) or not references or not all(isinstance(key, str) and key in facts for key in references):
        raise ValueError("Cite nonempty evidence_ids returned by tools; unknown evidence cannot be used")
    if not any(facts[key]["trial"] == trial and not key.endswith("metric_definition") for key in references):
        raise ValueError("Cite an actual measurement or execution fact from the latest trial")
    for key in ("hypothesis", "alternative", "next_test", "distinguishes"):
        if not isinstance(args[key], str) or len(args[key]) > 700:
            raise ValueError(f"{key} must be a concise string, at most 700 characters")
    if not args["alternative"].strip():
        raise ValueError("Give an alternative outcome or unresolved uncertainty, not a causal conclusion")
    if args["decision"] not in ("continue", "stop"):
        raise ValueError("decision must be continue or stop")
    if args["decision"] == "continue" and not all(args[key].strip() for key in ("hypothesis", "next_test", "distinguishes")):
        raise ValueError("The next experiment must state what competing explanations it distinguishes")
    primary = [key for key, fact in facts.items() if fact["trial"] == trial and key.endswith("." + state["config"]["metric"])
               and ".training." not in key and ".fold." not in key]
    return {"trial": trial, "measured": [facts[key] for key in dict.fromkeys(primary + references)],
            "finding": observed_finding(state["trials"][trial - 1], state["config"]["metric"]),
            "hypothesis": args["hypothesis"], "alternative": args["alternative"],
            "next_test": args["next_test"], "distinguishes": args["distinguishes"], "decision": args["decision"],
            "caution": "Interpretations are untested hypotheses. A negative trial does not establish causality, calibration or optimality."}


def measured_conclusion(state):
    best = state.get("best")
    if not best:
        return "No complete benchmark candidate is available. Inspect the recorded failures."
    metric = state["config"]["metric"]
    first = [row[metric] for row in state["trials"][0].get("summary", []) if row.get(metric) is not None
             and row["completed_folds"] == row["expected_folds"]]
    baseline = (min(first) if metric in ("mae", "rmse", "smape", "mase") else max(first)) if first else None
    change = f" Baseline best: {baseline:.6g}." if baseline is not None else ""
    return (f"Best validation candidate: trial {best['trial']}, {best['model']} / {best['augmentation']}, "
            f"{metric} = {best[metric]:.6g}.{change} Completed {len(state['trials'])} trial attempts. "
            "These validation comparisons do not establish significance, causality or optimality. "
            "Final evaluation remains separate from selection.")
