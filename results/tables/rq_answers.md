# Data for RQ1-RQ5

Mean ± std over seeds; std is the sample standard deviation (ddof=1).
Differences are paired by seed and use only the seeds both sides have. `within seed noise`: the mean difference is smaller than the larger of the two seed stds.

## RQ1: degradation, clean to corrupted

| Model | Strategy | Clean | Corrupted | Abs. drop | Rel. drop | Drop s1 | Drop s2 | Drop s3 | Drop s4 | Drop s5 | Drop grows with severity |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| simple_cnn | standard | 84.5% ± 0.6% | 59.7% ± 0.0% | +24.7 pp | 29.3% | +10.3 pp | +17.7 pp | +24.1 pp | +31.3 pp | +40.3 pp | yes |
| simple_cnn | augmix | 83.4% ± 0.6% | 67.0% ± 0.8% | +16.4 pp | 19.6% | +6.4 pp | +10.9 pp | +15.0 pp | +20.8 pp | +28.8 pp | yes |
| simple_cnn | corruption_aware | 82.3% ± 0.4% | 74.8% ± 0.6% | +7.5 pp | 9.1% | +2.9 pp | +4.8 pp | +6.3 pp | +9.6 pp | +13.9 pp | yes |
| resnet18 | standard | 93.7% ± 0.1% | 71.6% ± 0.3% | +22.1 pp | 23.6% | +8.2 pp | +14.0 pp | +20.2 pp | +27.9 pp | +40.2 pp | yes |
| resnet18 | augmix | 93.7% ± 0.0% | 83.7% ± 0.5% | +10.0 pp | 10.6% | +3.5 pp | +5.6 pp | +8.1 pp | +12.6 pp | +20.0 pp | yes |
| resnet18 | corruption_aware | 92.5% ± 0.2% | 87.6% ± 0.1% | +4.9 pp | 5.3% | +1.8 pp | +2.7 pp | +4.0 pp | +6.3 pp | +9.7 pp | yes |

## RQ2: architecture

| Strategy | Metric | Simple CNN | ResNet-18 | ResNet-18 - CNN | Per seed | Seed noise |
| --- | --- | --- | --- | --- | --- | --- |
| standard | clean_acc | 84.5% | 93.7% | +9.2 pp | s0: +9.6 pp, s1: +9.5 pp, s2: +8.6 pp | beyond seed noise |
| standard | mean_corrupted_acc | 59.7% | 71.6% | +11.9 pp | s0: +11.6 pp, s1: +12.2 pp, s2: +11.8 pp | beyond seed noise |
| standard | mean_abs_degradation | 24.7% | 22.1% | -2.6 pp | s0: -2.0 pp, s1: -2.7 pp, s2: -3.2 pp | beyond seed noise |
| standard | mean_rel_degradation | 29.3% | 23.6% | -5.7 pp | s0: -5.1 pp, s1: -5.8 pp, s2: -6.1 pp | beyond seed noise |
| standard | mCE | 100.0% | 68.5% | -31.5 pp | s0: -30.5 pp, s1: -33.3 pp, s2: -30.8 pp | beyond seed noise |
| standard | relative_mCE | 100.0% | 90.7% | -9.3 pp | s0: -5.2 pp, s1: -11.7 pp, s2: -10.9 pp | beyond seed noise |
| augmix | clean_acc | 83.4% | 93.7% | +10.3 pp | s0: +10.0 pp, s1: +10.9 pp, s2: +9.9 pp | beyond seed noise |
| augmix | mean_corrupted_acc | 67.0% | 83.7% | +16.7 pp | s0: +15.8 pp, s1: +18.1 pp, s2: +16.2 pp | beyond seed noise |
| augmix | mean_abs_degradation | 16.4% | 10.0% | -6.4 pp | s0: -5.7 pp, s1: -7.2 pp, s2: -6.3 pp | beyond seed noise |
| augmix | mean_rel_degradation | 19.6% | 10.6% | -9.0 pp | s0: -8.2 pp, s1: -10.0 pp, s2: -8.9 pp | beyond seed noise |
| augmix | mCE | 82.9% | 40.7% | -42.2 pp | s0: -40.4 pp, s1: -45.2 pp, s2: -41.0 pp | beyond seed noise |
| augmix | relative_mCE | 63.6% | 42.7% | -21.0 pp | s0: -20.1 pp, s1: -22.0 pp, s2: -20.9 pp | beyond seed noise |
| corruption_aware | clean_acc | 82.3% | 92.5% | +10.1 pp | s0: +10.3 pp, s1: +10.5 pp, s2: +9.6 pp | beyond seed noise |
| corruption_aware | mean_corrupted_acc | 74.8% | 87.6% | +12.7 pp | s0: +13.2 pp, s1: +13.1 pp, s2: +11.9 pp | beyond seed noise |
| corruption_aware | mean_abs_degradation | 7.5% | 4.9% | -2.6 pp | s0: -2.9 pp, s1: -2.6 pp, s2: -2.3 pp | beyond seed noise |
| corruption_aware | mean_rel_degradation | 9.1% | 5.3% | -3.8 pp | s0: -4.2 pp, s1: -3.8 pp, s2: -3.4 pp | beyond seed noise |
| corruption_aware | mCE | 68.5% | 33.8% | -34.7 pp | s0: -36.0 pp, s1: -35.6 pp, s2: -32.4 pp | beyond seed noise |
| corruption_aware | relative_mCE | 36.1% | 25.7% | -10.3 pp | s0: -12.0 pp, s1: -10.1 pp, s2: -9.0 pp | beyond seed noise |

## RQ3: strategies against standard training

| Model | Strategy | Metric | Standard | Strategy value | Strategy - standard | Per seed | Seed noise |
| --- | --- | --- | --- | --- | --- | --- | --- |
| simple_cnn | augmix | clean_acc | 84.5% | 83.4% | -1.1 pp | s0: -0.4 pp, s1: -1.5 pp, s2: -1.4 pp | beyond seed noise |
| simple_cnn | augmix | mean_corrupted_acc | 59.7% | 67.0% | +7.3 pp | s0: +8.0 pp, s1: +6.5 pp, s2: +7.4 pp | beyond seed noise |
| simple_cnn | augmix | mean_abs_degradation | 24.7% | 16.4% | -8.4 pp | s0: -8.3 pp, s1: -7.9 pp, s2: -8.8 pp | beyond seed noise |
| simple_cnn | augmix | mean_rel_degradation | 29.3% | 19.6% | -9.6 pp | s0: -9.8 pp, s1: -9.1 pp, s2: -10.0 pp | beyond seed noise |
| simple_cnn | augmix | mCE | 100.0% | 82.9% | -17.1 pp | s0: -18.3 pp, s1: -15.9 pp, s2: -17.0 pp | beyond seed noise |
| simple_cnn | augmix | relative_mCE | 100.0% | 63.6% | -36.4 pp | s0: -34.0 pp, s1: -37.6 pp, s2: -37.4 pp | beyond seed noise |
| simple_cnn | corruption_aware | clean_acc | 84.5% | 82.3% | -2.2 pp | s0: -2.0 pp, s1: -2.0 pp, s2: -2.4 pp | beyond seed noise |
| simple_cnn | corruption_aware | mean_corrupted_acc | 59.7% | 74.8% | +15.1 pp | s0: +14.6 pp, s1: +14.9 pp, s2: +15.7 pp | beyond seed noise |
| simple_cnn | corruption_aware | mean_abs_degradation | 24.7% | 7.5% | -17.2 pp | s0: -16.7 pp, s1: -16.9 pp, s2: -18.2 pp | beyond seed noise |
| simple_cnn | corruption_aware | mean_rel_degradation | 29.3% | 9.1% | -20.2 pp | s0: -19.6 pp, s1: -19.8 pp, s2: -21.1 pp | beyond seed noise |
| simple_cnn | corruption_aware | mCE | 100.0% | 68.5% | -31.5 pp | s0: -30.1 pp, s1: -31.6 pp, s2: -32.7 pp | beyond seed noise |
| simple_cnn | corruption_aware | relative_mCE | 100.0% | 36.1% | -63.9 pp | s0: -60.7 pp, s1: -63.6 pp, s2: -67.5 pp | beyond seed noise |
| resnet18 | augmix | clean_acc | 93.7% | 93.7% | +0.0 pp | s0: +0.1 pp, s1: -0.0 pp, s2: -0.1 pp | within seed noise |
| resnet18 | augmix | mean_corrupted_acc | 71.6% | 83.7% | +12.1 pp | s0: +12.1 pp, s1: +12.4 pp, s2: +11.9 pp | beyond seed noise |
| resnet18 | augmix | mean_abs_degradation | 22.1% | 10.0% | -12.1 pp | s0: -12.0 pp, s1: -12.5 pp, s2: -12.0 pp | beyond seed noise |
| resnet18 | augmix | mean_rel_degradation | 23.6% | 10.6% | -13.0 pp | s0: -12.8 pp, s1: -13.3 pp, s2: -12.7 pp | beyond seed noise |
| resnet18 | augmix | mCE | 68.5% | 40.7% | -27.8 pp | s0: -28.2 pp, s1: -27.8 pp, s2: -27.3 pp | beyond seed noise |
| resnet18 | augmix | relative_mCE | 90.7% | 42.7% | -48.1 pp | s0: -48.9 pp, s1: -47.9 pp, s2: -47.5 pp | beyond seed noise |
| resnet18 | corruption_aware | clean_acc | 93.7% | 92.5% | -1.2 pp | s0: -1.3 pp, s1: -1.0 pp, s2: -1.4 pp | beyond seed noise |
| resnet18 | corruption_aware | mean_corrupted_acc | 71.6% | 87.6% | +16.0 pp | s0: +16.3 pp, s1: +15.8 pp, s2: +15.9 pp | beyond seed noise |
| resnet18 | corruption_aware | mean_abs_degradation | 22.1% | 4.9% | -17.2 pp | s0: -17.5 pp, s1: -16.8 pp, s2: -17.3 pp | beyond seed noise |
| resnet18 | corruption_aware | mean_rel_degradation | 23.6% | 5.3% | -18.3 pp | s0: -18.7 pp, s1: -17.8 pp, s2: -18.4 pp | beyond seed noise |
| resnet18 | corruption_aware | mCE | 68.5% | 33.8% | -34.7 pp | s0: -35.6 pp, s1: -34.0 pp, s2: -34.4 pp | beyond seed noise |
| resnet18 | corruption_aware | relative_mCE | 90.7% | 25.7% | -65.0 pp | s0: -67.5 pp, s1: -62.0 pp, s2: -65.6 pp | beyond seed noise |

Accuracy gain per family:

| Model | Strategy | Family | Accuracy gain vs standard | Per seed | Seed noise |
| --- | --- | --- | --- | --- | --- |
| simple_cnn | augmix | noise | +12.0 pp | s0: +14.3 pp, s1: +8.2 pp, s2: +13.4 pp | beyond seed noise |
| simple_cnn | augmix | blur | +11.2 pp | s0: +11.6 pp, s1: +11.5 pp, s2: +10.6 pp | beyond seed noise |
| simple_cnn | augmix | weather | +5.3 pp | s0: +5.4 pp, s1: +4.9 pp, s2: +5.5 pp | beyond seed noise |
| simple_cnn | augmix | digital | +2.5 pp | s0: +2.8 pp, s1: +2.4 pp, s2: +2.5 pp | beyond seed noise |
| simple_cnn | corruption_aware | noise | +34.7 pp | s0: +34.8 pp, s1: +32.9 pp, s2: +36.3 pp | beyond seed noise |
| simple_cnn | corruption_aware | blur | +15.5 pp | s0: +14.7 pp, s1: +15.9 pp, s2: +16.0 pp | beyond seed noise |
| simple_cnn | corruption_aware | weather | +9.6 pp | s0: +8.7 pp, s1: +9.7 pp, s2: +10.4 pp | beyond seed noise |
| simple_cnn | corruption_aware | digital | +6.3 pp | s0: +6.0 pp, s1: +6.4 pp, s2: +6.4 pp | beyond seed noise |
| resnet18 | augmix | noise | +23.6 pp | s0: +20.7 pp, s1: +26.7 pp, s2: +23.3 pp | beyond seed noise |
| resnet18 | augmix | blur | +15.2 pp | s0: +16.6 pp, s1: +14.5 pp, s2: +14.5 pp | beyond seed noise |
| resnet18 | augmix | weather | +6.5 pp | s0: +6.4 pp, s1: +6.5 pp, s2: +6.5 pp | beyond seed noise |
| resnet18 | augmix | digital | +6.2 pp | s0: +6.8 pp, s1: +5.8 pp, s2: +6.2 pp | beyond seed noise |
| resnet18 | corruption_aware | noise | +34.1 pp | s0: +32.5 pp, s1: +35.0 pp, s2: +34.7 pp | beyond seed noise |
| resnet18 | corruption_aware | blur | +17.5 pp | s0: +19.1 pp, s1: +16.5 pp, s2: +16.8 pp | beyond seed noise |
| resnet18 | corruption_aware | weather | +7.5 pp | s0: +7.5 pp, s1: +7.5 pp, s2: +7.5 pp | beyond seed noise |
| resnet18 | corruption_aware | digital | +9.0 pp | s0: +9.5 pp, s1: +8.6 pp, s2: +8.9 pp | beyond seed noise |

## RQ4: seen and unseen corruptions

| Model | Strategy | Group | Standard | Strategy value | Gain vs standard | Per seed | Seed noise |
| --- | --- | --- | --- | --- | --- | --- | --- |
| simple_cnn | augmix | seen | 58.1% | 65.2% | +7.2 pp | s0: +8.1 pp, s1: +6.0 pp, s2: +7.4 pp | beyond seed noise |
| simple_cnn | augmix | unseen_same_family | 58.9% | 67.0% | +8.1 pp | s0: +8.8 pp, s1: +7.3 pp, s2: +8.2 pp | beyond seed noise |
| simple_cnn | augmix | unseen_other_family | 64.2% | 69.5% | +5.3 pp | s0: +5.4 pp, s1: +4.9 pp, s2: +5.5 pp | beyond seed noise |
| simple_cnn | corruption_aware | seen | 58.1% | 78.0% | +19.9 pp | s0: +19.7 pp, s1: +19.7 pp, s2: +20.5 pp | beyond seed noise |
| simple_cnn | corruption_aware | unseen_same_family | 58.9% | 73.6% | +14.7 pp | s0: +14.4 pp, s1: +14.4 pp, s2: +15.4 pp | beyond seed noise |
| simple_cnn | corruption_aware | unseen_other_family | 64.2% | 73.8% | +9.6 pp | s0: +8.7 pp, s1: +9.7 pp, s2: +10.4 pp | beyond seed noise |
| resnet18 | augmix | seen | 69.6% | 82.9% | +13.3 pp | s0: +12.8 pp, s1: +13.8 pp, s2: +13.1 pp | beyond seed noise |
| resnet18 | augmix | unseen_same_family | 69.6% | 83.3% | +13.7 pp | s0: +13.9 pp, s1: +14.0 pp, s2: +13.3 pp | beyond seed noise |
| resnet18 | augmix | unseen_other_family | 79.4% | 85.9% | +6.5 pp | s0: +6.4 pp, s1: +6.5 pp, s2: +6.5 pp | beyond seed noise |
| resnet18 | corruption_aware | seen | 69.6% | 90.1% | +20.5 pp | s0: +20.5 pp, s1: +20.4 pp, s2: +20.5 pp | beyond seed noise |
| resnet18 | corruption_aware | unseen_same_family | 69.6% | 86.5% | +16.9 pp | s0: +17.4 pp, s1: +16.6 pp, s2: +16.7 pp | beyond seed noise |
| resnet18 | corruption_aware | unseen_other_family | 79.4% | 87.0% | +7.5 pp | s0: +7.5 pp, s1: +7.5 pp, s2: +7.5 pp | beyond seed noise |

Gain on seen corruptions minus gain on an unseen group (seed noise: stds of the per-seed gains):

| Model | Strategy | Seen gain - gain on | Difference | Per seed | Seed noise |
| --- | --- | --- | --- | --- | --- |
| simple_cnn | augmix | unseen_same_family | -0.9 pp | s0: -0.8 pp, s1: -1.3 pp, s2: -0.8 pp | within seed noise |
| simple_cnn | augmix | unseen_other_family | +1.9 pp | s0: +2.7 pp, s1: +1.2 pp, s2: +1.8 pp | beyond seed noise |
| simple_cnn | corruption_aware | unseen_same_family | +5.2 pp | s0: +5.3 pp, s1: +5.3 pp, s2: +5.1 pp | beyond seed noise |
| simple_cnn | corruption_aware | unseen_other_family | +10.3 pp | s0: +11.0 pp, s1: +10.0 pp, s2: +10.1 pp | beyond seed noise |
| resnet18 | augmix | unseen_same_family | -0.5 pp | s0: -1.1 pp, s1: -0.2 pp, s2: -0.1 pp | within seed noise |
| resnet18 | augmix | unseen_other_family | +6.8 pp | s0: +6.4 pp, s1: +7.3 pp, s2: +6.6 pp | beyond seed noise |
| resnet18 | corruption_aware | unseen_same_family | +3.6 pp | s0: +3.1 pp, s1: +3.8 pp, s2: +3.9 pp | beyond seed noise |
| resnet18 | corruption_aware | unseen_other_family | +13.0 pp | s0: +13.0 pp, s1: +12.9 pp, s2: +13.1 pp | beyond seed noise |

## RQ5: corruption families (mean absolute drop)

| Model | Strategy | noise | blur | weather | digital | Worst family | Mildest family | Worst corruption | Mildest corruption | Worst - runner-up | Seed noise |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| simple_cnn | standard | +43.1 pp | +27.4 pp | +20.2 pp | +14.3 pp | noise | digital | gaussian_noise | brightness | +15.8 pp | beyond seed noise |
| simple_cnn | augmix | +30.1 pp | +15.1 pp | +13.9 pp | +10.7 pp | noise | digital | glass_blur | brightness | +15.0 pp | beyond seed noise |
| simple_cnn | corruption_aware | +6.3 pp | +9.7 pp | +8.5 pp | +5.9 pp | blur | digital | glass_blur | brightness | +1.2 pp | beyond seed noise |
| resnet18 | standard | +39.5 pp | +25.0 pp | +14.3 pp | +14.0 pp | noise | digital | gaussian_noise | brightness | +14.6 pp | beyond seed noise |
| resnet18 | augmix | +16.0 pp | +9.8 pp | +7.8 pp | +7.8 pp | noise | weather | glass_blur | brightness | +6.2 pp | beyond seed noise |
| resnet18 | corruption_aware | +4.2 pp | +6.3 pp | +5.5 pp | +3.8 pp | blur | digital | glass_blur | contrast | +0.8 pp | beyond seed noise |

Other rank gaps (runner-up - third, second mildest - mildest; mean absolute drop):

| Model | Strategy | Kind | Gap | Pair | Mean gap | Per seed | Seed noise |
| --- | --- | --- | --- | --- | --- | --- | --- |
| simple_cnn | standard | family | second_gap | blur - weather | +7.1 pp | s0: +7.4 pp, s1: +7.4 pp, s2: +6.6 pp | beyond seed noise |
| simple_cnn | standard | family | mildest_gap | weather - digital | +5.9 pp | s0: +5.4 pp, s1: +6.2 pp, s2: +6.2 pp | beyond seed noise |
| simple_cnn | standard | corruption | worst_gap | gaussian_noise - glass_blur | +0.2 pp | s0: -0.3 pp, s1: -0.0 pp, s2: +1.0 pp | within seed noise |
| simple_cnn | standard | corruption | second_gap | glass_blur - shot_noise | +9.4 pp | s0: +10.2 pp, s1: +9.2 pp, s2: +8.8 pp | beyond seed noise |
| simple_cnn | standard | corruption | mildest_gap | jpeg_compression - brightness | +8.2 pp | s0: +7.9 pp, s1: +8.0 pp, s2: +8.7 pp | beyond seed noise |
| simple_cnn | augmix | family | second_gap | blur - weather | +1.2 pp | s0: +1.2 pp, s1: +0.7 pp, s2: +1.5 pp | beyond seed noise |
| simple_cnn | augmix | family | mildest_gap | weather - digital | +3.2 pp | s0: +2.7 pp, s1: +3.7 pp, s2: +3.2 pp | beyond seed noise |
| simple_cnn | augmix | corruption | worst_gap | glass_blur - gaussian_noise | +2.6 pp | s0: +3.9 pp, s1: +1.0 pp, s2: +3.0 pp | beyond seed noise |
| simple_cnn | augmix | corruption | second_gap | gaussian_noise - shot_noise | +8.7 pp | s0: +7.8 pp, s1: +9.2 pp, s2: +9.0 pp | beyond seed noise |
| simple_cnn | augmix | corruption | mildest_gap | defocus_blur - brightness | +2.5 pp | s0: +2.3 pp, s1: +2.4 pp, s2: +2.7 pp | beyond seed noise |
| simple_cnn | corruption_aware | family | second_gap | weather - noise | +2.2 pp | s0: +2.3 pp, s1: +2.5 pp, s2: +1.8 pp | beyond seed noise |
| simple_cnn | corruption_aware | family | mildest_gap | noise - digital | +0.4 pp | s0: +0.4 pp, s1: +0.3 pp, s2: +0.5 pp | beyond seed noise |
| simple_cnn | corruption_aware | corruption | worst_gap | glass_blur - pixelate | +7.8 pp | s0: +8.1 pp, s1: +8.9 pp, s2: +6.4 pp | beyond seed noise |
| simple_cnn | corruption_aware | corruption | second_gap | pixelate - snow | +1.7 pp | s0: +1.2 pp, s1: +1.4 pp, s2: +2.4 pp | beyond seed noise |
| simple_cnn | corruption_aware | corruption | mildest_gap | defocus_blur - brightness | +1.4 pp | s0: +1.6 pp, s1: +1.3 pp, s2: +1.3 pp | beyond seed noise |
| resnet18 | standard | family | second_gap | blur - weather | +10.7 pp | s0: +12.2 pp, s1: +9.8 pp, s2: +10.1 pp | beyond seed noise |
| resnet18 | standard | family | mildest_gap | weather - digital | +0.2 pp | s0: -0.1 pp, s1: +0.4 pp, s2: +0.3 pp | within seed noise |
| resnet18 | standard | corruption | worst_gap | gaussian_noise - glass_blur | +0.5 pp | s0: -3.9 pp, s1: +3.0 pp, s2: +2.3 pp | within seed noise |
| resnet18 | standard | corruption | second_gap | glass_blur - impulse_noise | +9.4 pp | s0: +13.7 pp, s1: +7.6 pp, s2: +7.0 pp | beyond seed noise |
| resnet18 | standard | corruption | mildest_gap | fog - brightness | +6.3 pp | s0: +6.3 pp, s1: +6.0 pp, s2: +6.6 pp | beyond seed noise |
| resnet18 | augmix | family | second_gap | blur - digital | +2.0 pp | s0: +2.3 pp, s1: +1.5 pp, s2: +2.1 pp | beyond seed noise |
| resnet18 | augmix | family | mildest_gap | digital - weather | +0.0 pp | s0: -0.3 pp, s1: +0.3 pp, s2: +0.1 pp | within seed noise |
| resnet18 | augmix | corruption | worst_gap | glass_blur - gaussian_noise | +6.9 pp | s0: +5.2 pp, s1: +8.8 pp, s2: +6.6 pp | beyond seed noise |
| resnet18 | augmix | corruption | second_gap | gaussian_noise - impulse_noise | +5.9 pp | s0: +7.1 pp, s1: +4.2 pp, s2: +6.4 pp | beyond seed noise |
| resnet18 | augmix | corruption | mildest_gap | defocus_blur - brightness | +1.0 pp | s0: +1.2 pp, s1: +0.8 pp, s2: +1.1 pp | beyond seed noise |
| resnet18 | corruption_aware | family | second_gap | weather - noise | +1.3 pp | s0: +1.4 pp, s1: +1.0 pp, s2: +1.4 pp | beyond seed noise |
| resnet18 | corruption_aware | family | mildest_gap | noise - digital | +0.4 pp | s0: +0.4 pp, s1: +0.5 pp, s2: +0.3 pp | beyond seed noise |
| resnet18 | corruption_aware | corruption | worst_gap | glass_blur - pixelate | +6.1 pp | s0: +5.7 pp, s1: +6.4 pp, s2: +6.3 pp | beyond seed noise |
| resnet18 | corruption_aware | corruption | second_gap | pixelate - motion_blur | +0.6 pp | s0: -0.1 pp, s1: +1.3 pp, s2: +0.5 pp | beyond seed noise |
| resnet18 | corruption_aware | corruption | mildest_gap | brightness - contrast | +0.2 pp | s0: +0.1 pp, s1: +0.5 pp, s2: +0.1 pp | beyond seed noise |
