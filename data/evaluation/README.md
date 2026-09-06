# Evaluation measurements

These processed measurements cover the repeated experiments listed below.
They were extracted from the recorded benchmark outputs and profiler summaries;
no new benchmark runs were performed to produce this supplement.

## Files

- [`run_measurements.csv`](run_measurements.csv): 383 metric observations,
  retaining the experiment, system, model, condition, run or profile number,
  metric, unit, and measured value. One run can contribute several metrics.
- [`summary_ci95.csv`](summary_ci95.csv): 77 summaries containing the repetition
  count, arithmetic mean, sample standard deviation, and lower and upper limits
  of the 95% confidence interval for the mean.
- [`REPRODUCIBILITY.md`](../../REPRODUCIBILITY.md): platform and execution settings.

## Experiment coverage

| `study` | Systems and conditions | Manuscript location | Repetitions per condition |
| --- | --- | --- | --- |
| `primary_concurrency` | GAZSI and Baseline, BERT-base, concurrency 1, 2, 4, 8, 16, 32 | Section 4.2, primary comparison | 5 runs |
| `model_comparison` | GAZSI and Baseline, BERT-base, GPT-2, GPT-2-Large, concurrency 16 | Section 4.6 | 5 runs |
| `model_comparison` | Triton, BERT-base and GPT-2, concurrency 16 | Section 4.6, separate serving reference | 5 runs |
| `serving_reference` | Triton, BERT-base, concurrency 1, 2, 4, 8, 16, 32 | Sections 4.2 and 4.5.2 | 5 runs |
| `burst_reference` | Triton, BERT-base, schedules 4/32/4 and 4/64/4, with each phase summarized separately | Section 4.5.3 | 5 runs |
| `shared_gpu_model_kernel` | GAZSI, BERT-base, concurrency 8, four receive queues, batch one | Section 4.3, model-kernel comparison | 3 profile comparisons |

Triton is a separate serving reference with a different request representation.
Its observations are grouped separately from those for GAZSI and Baseline.
The primary concurrency experiment and the model comparison also retain separate
study identifiers, even when their model and concurrency are the same.

Results based on one recorded run or trace are outside this supplement's coverage.
Samples within a trace and individual kernel invocations are not counted as
independent experimental repetitions.

## Columns and units

Both tables identify a group by `study`, `system`, `model`, `condition`, `metric`,
and `unit`. Group by all six columns when reproducing the summaries.

In `run_measurements.csv`, `run` identifies the repeated run or profile
comparison within that group, and `value` is its observation. Burst phases from
the same run retain the same run identifier; they are summarized separately.

In `summary_ci95.csv`, `n` is the number of repetitions, `mean` is the arithmetic
mean, `sample_sd` uses the denominator `n - 1`, and `ci95_low` and `ci95_high` are
the lower and upper confidence limits.

| `metric` | Meaning | Unit |
| --- | --- | --- |
| `throughput` | Completed requests per second in one run | requests/s |
| `mean_ms` | Mean client response time in one run | ms |
| `run_p50_ms`, `run_p90_ms`, `run_p99_ms` | A run's response-time percentile | ms |
| `run_median_ms` | A burst phase's median response time in one run | ms |
| `model_kernel_increase` | Weighted mean model-kernel time increase with network processing on the same GPU | percent |

For the percentile and burst metrics, the confidence interval estimates the mean
of the per-run percentile or phase-median values. It does not estimate a
percentile from pooled request latencies. For `model_kernel_increase`, mean and
confidence limits are expressed in percent; standard deviation and interval
half-width are measured in percentage points. Those profile comparisons use the
recorded summaries rounded to six decimal places.

## Reproducing the confidence intervals

For each group, calculate the arithmetic mean and sample standard deviation of
its observations. The nominal 95% confidence interval for the mean is

```text
mean +/- t(0.975, n - 1) * sample_sd / sqrt(n)
```

The Student's t critical values used in the table are 2.7764451051977987 for
`n = 5` and 4.302652729696142 for `n = 3`. The calculation follows the
[NIST definition of confidence limits for a mean](https://www.itl.nist.gov/div898/handbook/eda/section3/eda352.htm).
It assumes independent, approximately normally distributed observations.
With three or five repetitions, those assumptions cannot be established reliably
from these samples alone. The individual observations are retained for inspection.
The intervals are pointwise, without adjustment for multiple comparisons.

The figures retain their stated error-bar definition of one sample standard
deviation. Confidence limits are reported separately in `summary_ci95.csv`.
