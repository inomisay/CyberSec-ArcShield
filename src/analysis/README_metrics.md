# Benchmark Metrics Reference

This file documents the metrics implemented in `metrics.py`. Rates are returned
as fractions from `0.0` to `1.0`; multiply by `100` or use `percent(value)` for
percentage display.

## Confusion Matrix

Used for binary classification tasks such as unsafe vs safe, attack success vs
blocked, or refused vs answered.

| Term | Meaning |
| --- | --- |
| `TP` | True positive: predicted positive and actually positive |
| `FP` | False positive: predicted positive but actually negative |
| `TN` | True negative: predicted negative and actually negative |
| `FN` | False negative: predicted negative but actually positive |

Formula/object:

```text
ConfusionMatrix(tp, fp, tn, fn)
```

Function:

```python
ConfusionMatrix.from_labels(y_true, y_pred)
```

## Attack Success Rate (ASR)

Measures how often an attack succeeds.

```text
ASR = successful_attacks / total_attacks
```

Function:

```python
attack_success_rate(successful_attacks, total_attacks)
```

## Defense Success Rate (DSR)

Measures how often the defense blocks or prevents attacks.

```text
DSR = blocked_attacks / total_attacks
DSR = 1 - ASR_with_defense
```

Function:

```python
defense_success_rate(blocked_attacks, total_attacks)
```

## Refusal Rate

Measures how often a model refuses to answer.

```text
Refusal Rate = refusals / total_responses
```

Function:

```python
refusal_rate(refusals, total_responses)
```

## Benign Refusal Rate (BRR)

Measures over-refusal on harmless prompts.

```text
BRR = refused_benign_prompts / total_benign_prompts
BRR (%) = BRR * 100
```

Function:

```python
benign_refusal_rate(refused_benign_prompts, total_benign_prompts)
```

BRR is reported separately from attack refusal rate. Refusing a harmful attack
can be valid defense behavior, but refusing a benign prompt is overblocking.

## Precision

Measures how many predicted positives were actually positive.

```text
Precision = TP / (TP + FP)
```

Function:

```python
precision(confusion_matrix)
```

## Recall

Measures how many actual positives were detected.

```text
Recall = TP / (TP + FN)
```

Function:

```python
recall(confusion_matrix)
```

## F1 Score

Harmonic mean of precision and recall.

```text
F1 = 2 * Precision * Recall / (Precision + Recall)
```

Function:

```python
f1_score(confusion_matrix)
```

## Accuracy

Measures overall correct predictions.

```text
Accuracy = (TP + TN) / (TP + FP + TN + FN)
```

Function:

```python
accuracy(confusion_matrix)
```

## False Positive Rate

Measures how often negative examples are incorrectly marked positive.

```text
FPR = FP / (FP + TN)
```

Function:

```python
false_positive_rate(confusion_matrix)
```

## False Negative Rate

Measures how often positive examples are incorrectly missed.

```text
FNR = FN / (FN + TP)
```

Function:

```python
false_negative_rate(confusion_matrix)
```

## Confidence Interval

Used to estimate uncertainty around a proportion or mean.

For a proportion:

```text
p_hat = successes / total
CI = p_hat +/- z * sqrt(p_hat * (1 - p_hat) / total)
```

For a mean:

```text
CI = mean +/- z * sample_std / sqrt(n)
```

Functions:

```python
normal_approx_confidence_interval(successes, total, confidence=0.95)
mean_confidence_interval(values, confidence=0.95)
```

## Standard Deviation

Measures spread around the mean.

Sample standard deviation:

```text
std = sqrt(sum((x_i - mean)^2) / (n - 1))
```

Population standard deviation:

```text
std = sqrt(sum((x_i - mean)^2) / n)
```

Function:

```python
standard_deviation(values, sample=True)
```

## Mean +/- Standard Deviation

Compact summary of central tendency and spread.

```text
mean +/- std
```

Functions:

```python
mean_std(values)
mean_std_text(values)
```

## McNemar Test

Used to compare two paired classifiers or defenses on the same examples.

```text
b01 = model A wrong, model B correct
b10 = model A correct, model B wrong
chi2 = (|b01 - b10| - 1)^2 / (b01 + b10)
```

The `- 1` term is the continuity correction.

Function:

```python
mcnemar_test(b01, b10, continuity_correction=True)
```

## Cohen's Kappa

Measures agreement between two raters beyond chance.

```text
kappa = (p_observed - p_expected) / (1 - p_expected)
```

Function:

```python
cohen_kappa(rater_a, rater_b)
```

## Krippendorff's Alpha

Optional multi-rater reliability metric that supports missing labels represented
as `None`.

```text
alpha = 1 - observed_disagreement / expected_disagreement
```

Function:

```python
krippendorff_alpha(ratings)
```

## Inter-Rater Reliability

General name for measuring judge/rater agreement. For two raters, this module
uses Cohen's Kappa.

```text
IRR = Cohen's Kappa
```

Function:

```python
inter_rater_reliability(rater_a, rater_b)
```

For more than two raters:

```python
fleiss_kappa(ratings)
```

## Majority Voting Judge

Combines multiple judge labels by choosing the most common label.

```text
majority_label = argmax_label count(label)
```

Function:

```python
majority_voting_judge(votes, tie_breaker=None)
```

## Ablation Study

Measures how much performance changes when one component is removed or changed.

```text
absolute_delta = ablated_metric - baseline_metric
relative_delta = absolute_delta / baseline_metric
```

Function:

```python
ablation_delta(baseline_metric, ablated_metric)
```

## Latency Measurement

Measures elapsed time for a model call or pipeline stage.

```text
latency = end_time - start_time
average_latency = sum(latencies) / n
```

Functions:

```python
latency_measurement(start_seconds, end_seconds)
average_latency(latencies)
```

## Token Overhead

Measures extra tokens introduced by a defense, wrapper, system prompt, or
instrumentation.

```text
absolute_overhead = defended_tokens - baseline_tokens
relative_overhead = absolute_overhead / baseline_tokens
```

Function:

```python
token_overhead(defended_tokens, baseline_tokens)
```

## Context Overhead

Same idea as token overhead, but specifically for context-window usage.

```text
absolute_context_overhead = defended_context_tokens - baseline_context_tokens
relative_context_overhead = absolute_context_overhead / baseline_context_tokens
```

Function:

```python
context_overhead(defended_context_tokens, baseline_context_tokens)
```

## Refusal Speedup

Measures whether a defense makes refusal faster or slower.

```text
speedup = baseline_refusal_latency / defended_refusal_latency
latency_delta = defended_refusal_latency - baseline_refusal_latency
```

If `speedup > 1`, the defended refusal is faster. If `speedup < 1`, it is slower.

Function:

```python
refusal_speedup(baseline_refusal_latency, defended_refusal_latency)
```

## Pareto Frontier Analysis

Finds non-dominated configurations when comparing tradeoffs such as high DSR,
low ASR, low latency, and low token overhead.

A point is dominated if another point is at least as good on every objective and
strictly better on at least one objective.

Example objectives:

```text
maximize = ["defense_success_rate", "accuracy"]
minimize = ["attack_success_rate", "latency", "token_overhead"]
```

Function:

```python
pareto_frontier(points, maximize=[...], minimize=[...])
```
