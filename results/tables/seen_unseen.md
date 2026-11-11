# Seen and unseen corruptions

Mean ± std over seeds; std is the sample standard deviation (ddof=1).

| Model | Strategy | Group | Seeds | Accuracy | Abs. degradation |
| --- | --- | --- | --- | --- | --- |
| simple_cnn | standard | seen | 3 | 58.1% ± 0.1% | 26.4% ± 0.6% |
| simple_cnn | standard | unseen_same_family | 3 | 58.9% ± 0.1% | 25.6% ± 0.6% |
| simple_cnn | standard | unseen_other_family | 3 | 64.2% ± 0.5% | 20.2% ± 0.8% |
| simple_cnn | augmix | seen | 3 | 65.2% ± 0.9% | 18.2% ± 0.5% |
| simple_cnn | augmix | unseen_same_family | 3 | 67.0% ± 0.7% | 16.4% ± 0.3% |
| simple_cnn | augmix | unseen_other_family | 3 | 69.5% ± 0.8% | 13.9% ± 0.3% |
| simple_cnn | corruption_aware | seen | 3 | 78.0% ± 0.5% | 4.3% ± 0.1% |
| simple_cnn | corruption_aware | unseen_same_family | 3 | 73.6% ± 0.6% | 8.7% ± 0.3% |
| simple_cnn | corruption_aware | unseen_other_family | 3 | 73.8% ± 0.7% | 8.5% ± 0.3% |
| resnet18 | standard | seen | 3 | 69.6% ± 0.2% | 24.0% ± 0.2% |
| resnet18 | standard | unseen_same_family | 3 | 69.6% ± 0.3% | 24.1% ± 0.3% |
| resnet18 | standard | unseen_other_family | 3 | 79.4% ± 0.2% | 14.3% ± 0.2% |
| resnet18 | augmix | seen | 3 | 82.9% ± 0.7% | 10.8% ± 0.7% |
| resnet18 | augmix | unseen_same_family | 3 | 83.3% ± 0.5% | 10.4% ± 0.6% |
| resnet18 | augmix | unseen_other_family | 3 | 85.9% ± 0.2% | 7.8% ± 0.3% |
| resnet18 | corruption_aware | seen | 3 | 90.1% ± 0.1% | 2.3% ± 0.1% |
| resnet18 | corruption_aware | unseen_same_family | 3 | 86.5% ± 0.2% | 6.0% ± 0.2% |
| resnet18 | corruption_aware | unseen_other_family | 3 | 87.0% ± 0.2% | 5.5% ± 0.0% |
