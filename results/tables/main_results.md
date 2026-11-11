# Main results

Mean ± std over seeds; std is the sample standard deviation (ddof=1).

| Model | Strategy | Seeds | clean_acc | clean_f1 | mean_corrupted_acc | mean_corrupted_f1 | mean_abs_degradation | mean_rel_degradation | mCE | relative_mCE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| simple_cnn | standard | 3 | 84.5% ± 0.6% | 84.3% ± 0.6% | 59.7% ± 0.0% | 59.2% ± 0.0% | 24.7% ± 0.6% | 29.3% ± 0.5% | 100.0% ± 0.7% | 100.0% ± 2.3% |
| simple_cnn | augmix | 3 | 83.4% ± 0.6% | 83.2% ± 0.6% | 67.0% ± 0.8% | 66.7% ± 0.8% | 16.4% ± 0.3% | 19.6% ± 0.5% | 82.9% ± 1.7% | 63.6% ± 0.9% |
| simple_cnn | corruption_aware | 3 | 82.3% ± 0.4% | 82.1% ± 0.4% | 74.8% ± 0.6% | 74.6% ± 0.6% | 7.5% ± 0.2% | 9.1% ± 0.3% | 68.5% ± 1.6% | 36.1% ± 1.3% |
| resnet18 | standard | 3 | 93.7% ± 0.1% | 93.7% ± 0.1% | 71.6% ± 0.3% | 71.5% ± 0.2% | 22.1% ± 0.2% | 23.6% ± 0.3% | 68.5% ± 0.9% | 90.7% ± 1.8% |
| resnet18 | augmix | 3 | 93.7% ± 0.0% | 93.7% ± 0.0% | 83.7% ± 0.5% | 83.9% ± 0.5% | 10.0% ± 0.5% | 10.6% ± 0.5% | 40.7% ± 0.9% | 42.7% ± 1.6% |
| resnet18 | corruption_aware | 3 | 92.5% ± 0.2% | 92.5% ± 0.2% | 87.6% ± 0.1% | 87.6% ± 0.1% | 4.9% ± 0.2% | 5.3% ± 0.2% | 33.8% ± 0.3% | 25.7% ± 1.0% |
