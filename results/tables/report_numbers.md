# Report numbers

Mean ± std over seeds; std is the sample standard deviation (ddof=1).

| Key | Meaning | Formula | Source |
| --- | --- | --- | --- |
| `compute.cnn.augmix.colab_train_minutes` | Mean training time per Colab run, minutes, simple_cnn + augmix | mean(train_time_s) / 60 without the Kaggle runs | results/runs.csv |
| `compute.cnn.augmix.time_ratio_vs_standard` | Colab training time of simple_cnn + augmix over simple_cnn + standard | mean(train_time_s) / mean(train_time_s of standard), Colab runs | results/runs.csv |
| `compute.cnn.augmix.train_minutes` | Mean training time per run, minutes, simple_cnn + augmix (all platforms) | mean(train_time_s) / 60 | results/runs.csv |
| `compute.cnn.corruption_aware.colab_train_minutes` | Mean training time per Colab run, minutes, simple_cnn + corruption_aware | mean(train_time_s) / 60 without the Kaggle runs | results/runs.csv |
| `compute.cnn.corruption_aware.time_ratio_vs_standard` | Colab training time of simple_cnn + corruption_aware over simple_cnn + standard | mean(train_time_s) / mean(train_time_s of standard), Colab runs | results/runs.csv |
| `compute.cnn.corruption_aware.train_minutes` | Mean training time per run, minutes, simple_cnn + corruption_aware (all platforms) | mean(train_time_s) / 60 | results/runs.csv |
| `compute.cnn.standard.colab_train_minutes` | Mean training time per Colab run, minutes, simple_cnn + standard | mean(train_time_s) / 60 without the Kaggle runs | results/runs.csv |
| `compute.cnn.standard.train_minutes` | Mean training time per run, minutes, simple_cnn + standard (all platforms) | mean(train_time_s) / 60 | results/runs.csv |
| `compute.kaggle_runs` | Runs trained on Kaggle | D9 | results/runs.csv |
| `compute.resnet18.augmix.colab_train_minutes` | Mean training time per Colab run, minutes, resnet18 + augmix | mean(train_time_s) / 60 without the Kaggle runs | results/runs.csv |
| `compute.resnet18.augmix.time_ratio_vs_standard` | Colab training time of resnet18 + augmix over resnet18 + standard | mean(train_time_s) / mean(train_time_s of standard), Colab runs | results/runs.csv |
| `compute.resnet18.augmix.train_minutes` | Mean training time per run, minutes, resnet18 + augmix (all platforms) | mean(train_time_s) / 60 | results/runs.csv |
| `compute.resnet18.corruption_aware.colab_train_minutes` | Mean training time per Colab run, minutes, resnet18 + corruption_aware | mean(train_time_s) / 60 without the Kaggle runs | results/runs.csv |
| `compute.resnet18.corruption_aware.time_ratio_vs_standard` | Colab training time of resnet18 + corruption_aware over resnet18 + standard | mean(train_time_s) / mean(train_time_s of standard), Colab runs | results/runs.csv |
| `compute.resnet18.corruption_aware.train_minutes` | Mean training time per run, minutes, resnet18 + corruption_aware (all platforms) | mean(train_time_s) / 60 | results/runs.csv |
| `compute.resnet18.standard.colab_train_minutes` | Mean training time per Colab run, minutes, resnet18 + standard | mean(train_time_s) / 60 without the Kaggle runs | results/runs.csv |
| `compute.resnet18.standard.train_minutes` | Mean training time per run, minutes, resnet18 + standard (all platforms) | mean(train_time_s) / 60 | results/runs.csv |
| `compute.train_hours_total` | Training time of all runs, hours (finished epochs only, no evaluation or setup) | sum(train_time_s) / 3600 | results/runs.csv |
| `confusion.cnn.augmix.clean__s0.second_class` | second predicted class (seed mean), clean__s0, simple_cnn + augmix | second largest seed-mean share of predictions | raw/{run_id}_confusion.npz |
| `confusion.cnn.augmix.clean__s0.second_share` | Share of all predictions on the second class, clean__s0, simple_cnn + augmix | seed mean of (predictions of the class / all predictions) | raw/{run_id}_confusion.npz |
| `confusion.cnn.augmix.clean__s0.top_class` | top predicted class (seed mean), clean__s0, simple_cnn + augmix | argmax over classes of the seed-mean share of predictions | raw/{run_id}_confusion.npz |
| `confusion.cnn.augmix.clean__s0.top_class_every_seed` | Top class is the most predicted in every seed, clean__s0, simple_cnn + augmix | per-seed argmax | raw/{run_id}_confusion.npz |
| `confusion.cnn.augmix.clean__s0.top_share` | Share of all predictions on the top class, clean__s0, simple_cnn + augmix | seed mean of (predictions of the class / all predictions) | raw/{run_id}_confusion.npz |
| `confusion.cnn.augmix.fog__s5.second_class` | second predicted class (seed mean), fog__s5, simple_cnn + augmix | second largest seed-mean share of predictions | raw/{run_id}_confusion.npz |
| `confusion.cnn.augmix.fog__s5.second_share` | Share of all predictions on the second class, fog__s5, simple_cnn + augmix | seed mean of (predictions of the class / all predictions) | raw/{run_id}_confusion.npz |
| `confusion.cnn.augmix.fog__s5.top_class` | top predicted class (seed mean), fog__s5, simple_cnn + augmix | argmax over classes of the seed-mean share of predictions | raw/{run_id}_confusion.npz |
| `confusion.cnn.augmix.fog__s5.top_class_every_seed` | Top class is the most predicted in every seed, fog__s5, simple_cnn + augmix | per-seed argmax | raw/{run_id}_confusion.npz |
| `confusion.cnn.augmix.fog__s5.top_share` | Share of all predictions on the top class, fog__s5, simple_cnn + augmix | seed mean of (predictions of the class / all predictions) | raw/{run_id}_confusion.npz |
| `confusion.cnn.augmix.gaussian_noise__s5.second_class` | second predicted class (seed mean), gaussian_noise__s5, simple_cnn + augmix | second largest seed-mean share of predictions | raw/{run_id}_confusion.npz |
| `confusion.cnn.augmix.gaussian_noise__s5.second_share` | Share of all predictions on the second class, gaussian_noise__s5, simple_cnn + augmix | seed mean of (predictions of the class / all predictions) | raw/{run_id}_confusion.npz |
| `confusion.cnn.augmix.gaussian_noise__s5.top_class` | top predicted class (seed mean), gaussian_noise__s5, simple_cnn + augmix | argmax over classes of the seed-mean share of predictions | raw/{run_id}_confusion.npz |
| `confusion.cnn.augmix.gaussian_noise__s5.top_class_every_seed` | Top class is the most predicted in every seed, gaussian_noise__s5, simple_cnn + augmix | per-seed argmax | raw/{run_id}_confusion.npz |
| `confusion.cnn.augmix.gaussian_noise__s5.top_share` | Share of all predictions on the top class, gaussian_noise__s5, simple_cnn + augmix | seed mean of (predictions of the class / all predictions) | raw/{run_id}_confusion.npz |
| `confusion.cnn.corruption_aware.clean__s0.second_class` | second predicted class (seed mean), clean__s0, simple_cnn + corruption_aware | second largest seed-mean share of predictions | raw/{run_id}_confusion.npz |
| `confusion.cnn.corruption_aware.clean__s0.second_share` | Share of all predictions on the second class, clean__s0, simple_cnn + corruption_aware | seed mean of (predictions of the class / all predictions) | raw/{run_id}_confusion.npz |
| `confusion.cnn.corruption_aware.clean__s0.top_class` | top predicted class (seed mean), clean__s0, simple_cnn + corruption_aware | argmax over classes of the seed-mean share of predictions | raw/{run_id}_confusion.npz |
| `confusion.cnn.corruption_aware.clean__s0.top_class_every_seed` | Top class is the most predicted in every seed, clean__s0, simple_cnn + corruption_aware | per-seed argmax | raw/{run_id}_confusion.npz |
| `confusion.cnn.corruption_aware.clean__s0.top_share` | Share of all predictions on the top class, clean__s0, simple_cnn + corruption_aware | seed mean of (predictions of the class / all predictions) | raw/{run_id}_confusion.npz |
| `confusion.cnn.corruption_aware.fog__s5.second_class` | second predicted class (seed mean), fog__s5, simple_cnn + corruption_aware | second largest seed-mean share of predictions | raw/{run_id}_confusion.npz |
| `confusion.cnn.corruption_aware.fog__s5.second_share` | Share of all predictions on the second class, fog__s5, simple_cnn + corruption_aware | seed mean of (predictions of the class / all predictions) | raw/{run_id}_confusion.npz |
| `confusion.cnn.corruption_aware.fog__s5.top_class` | top predicted class (seed mean), fog__s5, simple_cnn + corruption_aware | argmax over classes of the seed-mean share of predictions | raw/{run_id}_confusion.npz |
| `confusion.cnn.corruption_aware.fog__s5.top_class_every_seed` | Top class is the most predicted in every seed, fog__s5, simple_cnn + corruption_aware | per-seed argmax | raw/{run_id}_confusion.npz |
| `confusion.cnn.corruption_aware.fog__s5.top_share` | Share of all predictions on the top class, fog__s5, simple_cnn + corruption_aware | seed mean of (predictions of the class / all predictions) | raw/{run_id}_confusion.npz |
| `confusion.cnn.corruption_aware.gaussian_noise__s5.second_class` | second predicted class (seed mean), gaussian_noise__s5, simple_cnn + corruption_aware | second largest seed-mean share of predictions | raw/{run_id}_confusion.npz |
| `confusion.cnn.corruption_aware.gaussian_noise__s5.second_share` | Share of all predictions on the second class, gaussian_noise__s5, simple_cnn + corruption_aware | seed mean of (predictions of the class / all predictions) | raw/{run_id}_confusion.npz |
| `confusion.cnn.corruption_aware.gaussian_noise__s5.top_class` | top predicted class (seed mean), gaussian_noise__s5, simple_cnn + corruption_aware | argmax over classes of the seed-mean share of predictions | raw/{run_id}_confusion.npz |
| `confusion.cnn.corruption_aware.gaussian_noise__s5.top_class_every_seed` | Top class is the most predicted in every seed, gaussian_noise__s5, simple_cnn + corruption_aware | per-seed argmax | raw/{run_id}_confusion.npz |
| `confusion.cnn.corruption_aware.gaussian_noise__s5.top_share` | Share of all predictions on the top class, gaussian_noise__s5, simple_cnn + corruption_aware | seed mean of (predictions of the class / all predictions) | raw/{run_id}_confusion.npz |
| `confusion.cnn.standard.clean__s0.second_class` | second predicted class (seed mean), clean__s0, simple_cnn + standard | second largest seed-mean share of predictions | raw/{run_id}_confusion.npz |
| `confusion.cnn.standard.clean__s0.second_share` | Share of all predictions on the second class, clean__s0, simple_cnn + standard | seed mean of (predictions of the class / all predictions) | raw/{run_id}_confusion.npz |
| `confusion.cnn.standard.clean__s0.top_class` | top predicted class (seed mean), clean__s0, simple_cnn + standard | argmax over classes of the seed-mean share of predictions | raw/{run_id}_confusion.npz |
| `confusion.cnn.standard.clean__s0.top_class_every_seed` | Top class is the most predicted in every seed, clean__s0, simple_cnn + standard | per-seed argmax | raw/{run_id}_confusion.npz |
| `confusion.cnn.standard.clean__s0.top_share` | Share of all predictions on the top class, clean__s0, simple_cnn + standard | seed mean of (predictions of the class / all predictions) | raw/{run_id}_confusion.npz |
| `confusion.cnn.standard.fog__s5.second_class` | second predicted class (seed mean), fog__s5, simple_cnn + standard | second largest seed-mean share of predictions | raw/{run_id}_confusion.npz |
| `confusion.cnn.standard.fog__s5.second_share` | Share of all predictions on the second class, fog__s5, simple_cnn + standard | seed mean of (predictions of the class / all predictions) | raw/{run_id}_confusion.npz |
| `confusion.cnn.standard.fog__s5.top_class` | top predicted class (seed mean), fog__s5, simple_cnn + standard | argmax over classes of the seed-mean share of predictions | raw/{run_id}_confusion.npz |
| `confusion.cnn.standard.fog__s5.top_class_every_seed` | Top class is the most predicted in every seed, fog__s5, simple_cnn + standard | per-seed argmax | raw/{run_id}_confusion.npz |
| `confusion.cnn.standard.fog__s5.top_share` | Share of all predictions on the top class, fog__s5, simple_cnn + standard | seed mean of (predictions of the class / all predictions) | raw/{run_id}_confusion.npz |
| `confusion.cnn.standard.gaussian_noise__s5.second_class` | second predicted class (seed mean), gaussian_noise__s5, simple_cnn + standard | second largest seed-mean share of predictions | raw/{run_id}_confusion.npz |
| `confusion.cnn.standard.gaussian_noise__s5.second_share` | Share of all predictions on the second class, gaussian_noise__s5, simple_cnn + standard | seed mean of (predictions of the class / all predictions) | raw/{run_id}_confusion.npz |
| `confusion.cnn.standard.gaussian_noise__s5.top_class` | top predicted class (seed mean), gaussian_noise__s5, simple_cnn + standard | argmax over classes of the seed-mean share of predictions | raw/{run_id}_confusion.npz |
| `confusion.cnn.standard.gaussian_noise__s5.top_class_every_seed` | Top class is the most predicted in every seed, gaussian_noise__s5, simple_cnn + standard | per-seed argmax | raw/{run_id}_confusion.npz |
| `confusion.cnn.standard.gaussian_noise__s5.top_share` | Share of all predictions on the top class, gaussian_noise__s5, simple_cnn + standard | seed mean of (predictions of the class / all predictions) | raw/{run_id}_confusion.npz |
| `confusion.resnet18.augmix.clean__s0.second_class` | second predicted class (seed mean), clean__s0, resnet18 + augmix | second largest seed-mean share of predictions | raw/{run_id}_confusion.npz |
| `confusion.resnet18.augmix.clean__s0.second_share` | Share of all predictions on the second class, clean__s0, resnet18 + augmix | seed mean of (predictions of the class / all predictions) | raw/{run_id}_confusion.npz |
| `confusion.resnet18.augmix.clean__s0.top_class` | top predicted class (seed mean), clean__s0, resnet18 + augmix | argmax over classes of the seed-mean share of predictions | raw/{run_id}_confusion.npz |
| `confusion.resnet18.augmix.clean__s0.top_class_every_seed` | Top class is the most predicted in every seed, clean__s0, resnet18 + augmix | per-seed argmax | raw/{run_id}_confusion.npz |
| `confusion.resnet18.augmix.clean__s0.top_share` | Share of all predictions on the top class, clean__s0, resnet18 + augmix | seed mean of (predictions of the class / all predictions) | raw/{run_id}_confusion.npz |
| `confusion.resnet18.augmix.fog__s5.second_class` | second predicted class (seed mean), fog__s5, resnet18 + augmix | second largest seed-mean share of predictions | raw/{run_id}_confusion.npz |
| `confusion.resnet18.augmix.fog__s5.second_share` | Share of all predictions on the second class, fog__s5, resnet18 + augmix | seed mean of (predictions of the class / all predictions) | raw/{run_id}_confusion.npz |
| `confusion.resnet18.augmix.fog__s5.top_class` | top predicted class (seed mean), fog__s5, resnet18 + augmix | argmax over classes of the seed-mean share of predictions | raw/{run_id}_confusion.npz |
| `confusion.resnet18.augmix.fog__s5.top_class_every_seed` | Top class is the most predicted in every seed, fog__s5, resnet18 + augmix | per-seed argmax | raw/{run_id}_confusion.npz |
| `confusion.resnet18.augmix.fog__s5.top_share` | Share of all predictions on the top class, fog__s5, resnet18 + augmix | seed mean of (predictions of the class / all predictions) | raw/{run_id}_confusion.npz |
| `confusion.resnet18.augmix.gaussian_noise__s5.second_class` | second predicted class (seed mean), gaussian_noise__s5, resnet18 + augmix | second largest seed-mean share of predictions | raw/{run_id}_confusion.npz |
| `confusion.resnet18.augmix.gaussian_noise__s5.second_share` | Share of all predictions on the second class, gaussian_noise__s5, resnet18 + augmix | seed mean of (predictions of the class / all predictions) | raw/{run_id}_confusion.npz |
| `confusion.resnet18.augmix.gaussian_noise__s5.top_class` | top predicted class (seed mean), gaussian_noise__s5, resnet18 + augmix | argmax over classes of the seed-mean share of predictions | raw/{run_id}_confusion.npz |
| `confusion.resnet18.augmix.gaussian_noise__s5.top_class_every_seed` | Top class is the most predicted in every seed, gaussian_noise__s5, resnet18 + augmix | per-seed argmax | raw/{run_id}_confusion.npz |
| `confusion.resnet18.augmix.gaussian_noise__s5.top_share` | Share of all predictions on the top class, gaussian_noise__s5, resnet18 + augmix | seed mean of (predictions of the class / all predictions) | raw/{run_id}_confusion.npz |
| `confusion.resnet18.corruption_aware.clean__s0.second_class` | second predicted class (seed mean), clean__s0, resnet18 + corruption_aware | second largest seed-mean share of predictions | raw/{run_id}_confusion.npz |
| `confusion.resnet18.corruption_aware.clean__s0.second_share` | Share of all predictions on the second class, clean__s0, resnet18 + corruption_aware | seed mean of (predictions of the class / all predictions) | raw/{run_id}_confusion.npz |
| `confusion.resnet18.corruption_aware.clean__s0.top_class` | top predicted class (seed mean), clean__s0, resnet18 + corruption_aware | argmax over classes of the seed-mean share of predictions | raw/{run_id}_confusion.npz |
| `confusion.resnet18.corruption_aware.clean__s0.top_class_every_seed` | Top class is the most predicted in every seed, clean__s0, resnet18 + corruption_aware | per-seed argmax | raw/{run_id}_confusion.npz |
| `confusion.resnet18.corruption_aware.clean__s0.top_share` | Share of all predictions on the top class, clean__s0, resnet18 + corruption_aware | seed mean of (predictions of the class / all predictions) | raw/{run_id}_confusion.npz |
| `confusion.resnet18.corruption_aware.fog__s5.second_class` | second predicted class (seed mean), fog__s5, resnet18 + corruption_aware | second largest seed-mean share of predictions | raw/{run_id}_confusion.npz |
| `confusion.resnet18.corruption_aware.fog__s5.second_share` | Share of all predictions on the second class, fog__s5, resnet18 + corruption_aware | seed mean of (predictions of the class / all predictions) | raw/{run_id}_confusion.npz |
| `confusion.resnet18.corruption_aware.fog__s5.top_class` | top predicted class (seed mean), fog__s5, resnet18 + corruption_aware | argmax over classes of the seed-mean share of predictions | raw/{run_id}_confusion.npz |
| `confusion.resnet18.corruption_aware.fog__s5.top_class_every_seed` | Top class is the most predicted in every seed, fog__s5, resnet18 + corruption_aware | per-seed argmax | raw/{run_id}_confusion.npz |
| `confusion.resnet18.corruption_aware.fog__s5.top_share` | Share of all predictions on the top class, fog__s5, resnet18 + corruption_aware | seed mean of (predictions of the class / all predictions) | raw/{run_id}_confusion.npz |
| `confusion.resnet18.corruption_aware.gaussian_noise__s5.second_class` | second predicted class (seed mean), gaussian_noise__s5, resnet18 + corruption_aware | second largest seed-mean share of predictions | raw/{run_id}_confusion.npz |
| `confusion.resnet18.corruption_aware.gaussian_noise__s5.second_share` | Share of all predictions on the second class, gaussian_noise__s5, resnet18 + corruption_aware | seed mean of (predictions of the class / all predictions) | raw/{run_id}_confusion.npz |
| `confusion.resnet18.corruption_aware.gaussian_noise__s5.top_class` | top predicted class (seed mean), gaussian_noise__s5, resnet18 + corruption_aware | argmax over classes of the seed-mean share of predictions | raw/{run_id}_confusion.npz |
| `confusion.resnet18.corruption_aware.gaussian_noise__s5.top_class_every_seed` | Top class is the most predicted in every seed, gaussian_noise__s5, resnet18 + corruption_aware | per-seed argmax | raw/{run_id}_confusion.npz |
| `confusion.resnet18.corruption_aware.gaussian_noise__s5.top_share` | Share of all predictions on the top class, gaussian_noise__s5, resnet18 + corruption_aware | seed mean of (predictions of the class / all predictions) | raw/{run_id}_confusion.npz |
| `confusion.resnet18.standard.clean__s0.second_class` | second predicted class (seed mean), clean__s0, resnet18 + standard | second largest seed-mean share of predictions | raw/{run_id}_confusion.npz |
| `confusion.resnet18.standard.clean__s0.second_share` | Share of all predictions on the second class, clean__s0, resnet18 + standard | seed mean of (predictions of the class / all predictions) | raw/{run_id}_confusion.npz |
| `confusion.resnet18.standard.clean__s0.top_class` | top predicted class (seed mean), clean__s0, resnet18 + standard | argmax over classes of the seed-mean share of predictions | raw/{run_id}_confusion.npz |
| `confusion.resnet18.standard.clean__s0.top_class_every_seed` | Top class is the most predicted in every seed, clean__s0, resnet18 + standard | per-seed argmax | raw/{run_id}_confusion.npz |
| `confusion.resnet18.standard.clean__s0.top_share` | Share of all predictions on the top class, clean__s0, resnet18 + standard | seed mean of (predictions of the class / all predictions) | raw/{run_id}_confusion.npz |
| `confusion.resnet18.standard.fog__s5.second_class` | second predicted class (seed mean), fog__s5, resnet18 + standard | second largest seed-mean share of predictions | raw/{run_id}_confusion.npz |
| `confusion.resnet18.standard.fog__s5.second_share` | Share of all predictions on the second class, fog__s5, resnet18 + standard | seed mean of (predictions of the class / all predictions) | raw/{run_id}_confusion.npz |
| `confusion.resnet18.standard.fog__s5.top_class` | top predicted class (seed mean), fog__s5, resnet18 + standard | argmax over classes of the seed-mean share of predictions | raw/{run_id}_confusion.npz |
| `confusion.resnet18.standard.fog__s5.top_class_every_seed` | Top class is the most predicted in every seed, fog__s5, resnet18 + standard | per-seed argmax | raw/{run_id}_confusion.npz |
| `confusion.resnet18.standard.fog__s5.top_share` | Share of all predictions on the top class, fog__s5, resnet18 + standard | seed mean of (predictions of the class / all predictions) | raw/{run_id}_confusion.npz |
| `confusion.resnet18.standard.gaussian_noise__s5.second_class` | second predicted class (seed mean), gaussian_noise__s5, resnet18 + standard | second largest seed-mean share of predictions | raw/{run_id}_confusion.npz |
| `confusion.resnet18.standard.gaussian_noise__s5.second_share` | Share of all predictions on the second class, gaussian_noise__s5, resnet18 + standard | seed mean of (predictions of the class / all predictions) | raw/{run_id}_confusion.npz |
| `confusion.resnet18.standard.gaussian_noise__s5.top_class` | top predicted class (seed mean), gaussian_noise__s5, resnet18 + standard | argmax over classes of the seed-mean share of predictions | raw/{run_id}_confusion.npz |
| `confusion.resnet18.standard.gaussian_noise__s5.top_class_every_seed` | Top class is the most predicted in every seed, gaussian_noise__s5, resnet18 + standard | per-seed argmax | raw/{run_id}_confusion.npz |
| `confusion.resnet18.standard.gaussian_noise__s5.top_share` | Share of all predictions on the top class, gaussian_noise__s5, resnet18 + standard | seed mean of (predictions of the class / all predictions) | raw/{run_id}_confusion.npz |
| `groups.cnn.augmix.seen.abs_degradation.mean` | abs_degradation of simple_cnn + augmix, seen (mean) | mean over seeds of the group's abs_degradation | seen_unseen.csv |
| `groups.cnn.augmix.seen.abs_degradation.std` | abs_degradation of simple_cnn + augmix, seen (std) | std over seeds of the group's abs_degradation, ddof=1 | seen_unseen.csv |
| `groups.cnn.augmix.seen.accuracy.mean` | accuracy of simple_cnn + augmix, seen (mean) | mean over seeds of the group's accuracy | seen_unseen.csv |
| `groups.cnn.augmix.seen.accuracy.std` | accuracy of simple_cnn + augmix, seen (std) | std over seeds of the group's accuracy, ddof=1 | seen_unseen.csv |
| `groups.cnn.augmix.unseen_other_family.abs_degradation.mean` | abs_degradation of simple_cnn + augmix, unseen_other_family (mean) | mean over seeds of the group's abs_degradation | seen_unseen.csv |
| `groups.cnn.augmix.unseen_other_family.abs_degradation.std` | abs_degradation of simple_cnn + augmix, unseen_other_family (std) | std over seeds of the group's abs_degradation, ddof=1 | seen_unseen.csv |
| `groups.cnn.augmix.unseen_other_family.accuracy.mean` | accuracy of simple_cnn + augmix, unseen_other_family (mean) | mean over seeds of the group's accuracy | seen_unseen.csv |
| `groups.cnn.augmix.unseen_other_family.accuracy.std` | accuracy of simple_cnn + augmix, unseen_other_family (std) | std over seeds of the group's accuracy, ddof=1 | seen_unseen.csv |
| `groups.cnn.augmix.unseen_same_family.abs_degradation.mean` | abs_degradation of simple_cnn + augmix, unseen_same_family (mean) | mean over seeds of the group's abs_degradation | seen_unseen.csv |
| `groups.cnn.augmix.unseen_same_family.abs_degradation.std` | abs_degradation of simple_cnn + augmix, unseen_same_family (std) | std over seeds of the group's abs_degradation, ddof=1 | seen_unseen.csv |
| `groups.cnn.augmix.unseen_same_family.accuracy.mean` | accuracy of simple_cnn + augmix, unseen_same_family (mean) | mean over seeds of the group's accuracy | seen_unseen.csv |
| `groups.cnn.augmix.unseen_same_family.accuracy.std` | accuracy of simple_cnn + augmix, unseen_same_family (std) | std over seeds of the group's accuracy, ddof=1 | seen_unseen.csv |
| `groups.cnn.corruption_aware.seen.abs_degradation.mean` | abs_degradation of simple_cnn + corruption_aware, seen (mean) | mean over seeds of the group's abs_degradation | seen_unseen.csv |
| `groups.cnn.corruption_aware.seen.abs_degradation.std` | abs_degradation of simple_cnn + corruption_aware, seen (std) | std over seeds of the group's abs_degradation, ddof=1 | seen_unseen.csv |
| `groups.cnn.corruption_aware.seen.accuracy.mean` | accuracy of simple_cnn + corruption_aware, seen (mean) | mean over seeds of the group's accuracy | seen_unseen.csv |
| `groups.cnn.corruption_aware.seen.accuracy.std` | accuracy of simple_cnn + corruption_aware, seen (std) | std over seeds of the group's accuracy, ddof=1 | seen_unseen.csv |
| `groups.cnn.corruption_aware.unseen_other_family.abs_degradation.mean` | abs_degradation of simple_cnn + corruption_aware, unseen_other_family (mean) | mean over seeds of the group's abs_degradation | seen_unseen.csv |
| `groups.cnn.corruption_aware.unseen_other_family.abs_degradation.std` | abs_degradation of simple_cnn + corruption_aware, unseen_other_family (std) | std over seeds of the group's abs_degradation, ddof=1 | seen_unseen.csv |
| `groups.cnn.corruption_aware.unseen_other_family.accuracy.mean` | accuracy of simple_cnn + corruption_aware, unseen_other_family (mean) | mean over seeds of the group's accuracy | seen_unseen.csv |
| `groups.cnn.corruption_aware.unseen_other_family.accuracy.std` | accuracy of simple_cnn + corruption_aware, unseen_other_family (std) | std over seeds of the group's accuracy, ddof=1 | seen_unseen.csv |
| `groups.cnn.corruption_aware.unseen_same_family.abs_degradation.mean` | abs_degradation of simple_cnn + corruption_aware, unseen_same_family (mean) | mean over seeds of the group's abs_degradation | seen_unseen.csv |
| `groups.cnn.corruption_aware.unseen_same_family.abs_degradation.std` | abs_degradation of simple_cnn + corruption_aware, unseen_same_family (std) | std over seeds of the group's abs_degradation, ddof=1 | seen_unseen.csv |
| `groups.cnn.corruption_aware.unseen_same_family.accuracy.mean` | accuracy of simple_cnn + corruption_aware, unseen_same_family (mean) | mean over seeds of the group's accuracy | seen_unseen.csv |
| `groups.cnn.corruption_aware.unseen_same_family.accuracy.std` | accuracy of simple_cnn + corruption_aware, unseen_same_family (std) | std over seeds of the group's accuracy, ddof=1 | seen_unseen.csv |
| `groups.cnn.standard.seen.abs_degradation.mean` | abs_degradation of simple_cnn + standard, seen (mean) | mean over seeds of the group's abs_degradation | seen_unseen.csv |
| `groups.cnn.standard.seen.abs_degradation.std` | abs_degradation of simple_cnn + standard, seen (std) | std over seeds of the group's abs_degradation, ddof=1 | seen_unseen.csv |
| `groups.cnn.standard.seen.accuracy.mean` | accuracy of simple_cnn + standard, seen (mean) | mean over seeds of the group's accuracy | seen_unseen.csv |
| `groups.cnn.standard.seen.accuracy.std` | accuracy of simple_cnn + standard, seen (std) | std over seeds of the group's accuracy, ddof=1 | seen_unseen.csv |
| `groups.cnn.standard.unseen_other_family.abs_degradation.mean` | abs_degradation of simple_cnn + standard, unseen_other_family (mean) | mean over seeds of the group's abs_degradation | seen_unseen.csv |
| `groups.cnn.standard.unseen_other_family.abs_degradation.std` | abs_degradation of simple_cnn + standard, unseen_other_family (std) | std over seeds of the group's abs_degradation, ddof=1 | seen_unseen.csv |
| `groups.cnn.standard.unseen_other_family.accuracy.mean` | accuracy of simple_cnn + standard, unseen_other_family (mean) | mean over seeds of the group's accuracy | seen_unseen.csv |
| `groups.cnn.standard.unseen_other_family.accuracy.std` | accuracy of simple_cnn + standard, unseen_other_family (std) | std over seeds of the group's accuracy, ddof=1 | seen_unseen.csv |
| `groups.cnn.standard.unseen_same_family.abs_degradation.mean` | abs_degradation of simple_cnn + standard, unseen_same_family (mean) | mean over seeds of the group's abs_degradation | seen_unseen.csv |
| `groups.cnn.standard.unseen_same_family.abs_degradation.std` | abs_degradation of simple_cnn + standard, unseen_same_family (std) | std over seeds of the group's abs_degradation, ddof=1 | seen_unseen.csv |
| `groups.cnn.standard.unseen_same_family.accuracy.mean` | accuracy of simple_cnn + standard, unseen_same_family (mean) | mean over seeds of the group's accuracy | seen_unseen.csv |
| `groups.cnn.standard.unseen_same_family.accuracy.std` | accuracy of simple_cnn + standard, unseen_same_family (std) | std over seeds of the group's accuracy, ddof=1 | seen_unseen.csv |
| `groups.resnet18.augmix.seen.abs_degradation.mean` | abs_degradation of resnet18 + augmix, seen (mean) | mean over seeds of the group's abs_degradation | seen_unseen.csv |
| `groups.resnet18.augmix.seen.abs_degradation.std` | abs_degradation of resnet18 + augmix, seen (std) | std over seeds of the group's abs_degradation, ddof=1 | seen_unseen.csv |
| `groups.resnet18.augmix.seen.accuracy.mean` | accuracy of resnet18 + augmix, seen (mean) | mean over seeds of the group's accuracy | seen_unseen.csv |
| `groups.resnet18.augmix.seen.accuracy.std` | accuracy of resnet18 + augmix, seen (std) | std over seeds of the group's accuracy, ddof=1 | seen_unseen.csv |
| `groups.resnet18.augmix.unseen_other_family.abs_degradation.mean` | abs_degradation of resnet18 + augmix, unseen_other_family (mean) | mean over seeds of the group's abs_degradation | seen_unseen.csv |
| `groups.resnet18.augmix.unseen_other_family.abs_degradation.std` | abs_degradation of resnet18 + augmix, unseen_other_family (std) | std over seeds of the group's abs_degradation, ddof=1 | seen_unseen.csv |
| `groups.resnet18.augmix.unseen_other_family.accuracy.mean` | accuracy of resnet18 + augmix, unseen_other_family (mean) | mean over seeds of the group's accuracy | seen_unseen.csv |
| `groups.resnet18.augmix.unseen_other_family.accuracy.std` | accuracy of resnet18 + augmix, unseen_other_family (std) | std over seeds of the group's accuracy, ddof=1 | seen_unseen.csv |
| `groups.resnet18.augmix.unseen_same_family.abs_degradation.mean` | abs_degradation of resnet18 + augmix, unseen_same_family (mean) | mean over seeds of the group's abs_degradation | seen_unseen.csv |
| `groups.resnet18.augmix.unseen_same_family.abs_degradation.std` | abs_degradation of resnet18 + augmix, unseen_same_family (std) | std over seeds of the group's abs_degradation, ddof=1 | seen_unseen.csv |
| `groups.resnet18.augmix.unseen_same_family.accuracy.mean` | accuracy of resnet18 + augmix, unseen_same_family (mean) | mean over seeds of the group's accuracy | seen_unseen.csv |
| `groups.resnet18.augmix.unseen_same_family.accuracy.std` | accuracy of resnet18 + augmix, unseen_same_family (std) | std over seeds of the group's accuracy, ddof=1 | seen_unseen.csv |
| `groups.resnet18.corruption_aware.seen.abs_degradation.mean` | abs_degradation of resnet18 + corruption_aware, seen (mean) | mean over seeds of the group's abs_degradation | seen_unseen.csv |
| `groups.resnet18.corruption_aware.seen.abs_degradation.std` | abs_degradation of resnet18 + corruption_aware, seen (std) | std over seeds of the group's abs_degradation, ddof=1 | seen_unseen.csv |
| `groups.resnet18.corruption_aware.seen.accuracy.mean` | accuracy of resnet18 + corruption_aware, seen (mean) | mean over seeds of the group's accuracy | seen_unseen.csv |
| `groups.resnet18.corruption_aware.seen.accuracy.std` | accuracy of resnet18 + corruption_aware, seen (std) | std over seeds of the group's accuracy, ddof=1 | seen_unseen.csv |
| `groups.resnet18.corruption_aware.unseen_other_family.abs_degradation.mean` | abs_degradation of resnet18 + corruption_aware, unseen_other_family (mean) | mean over seeds of the group's abs_degradation | seen_unseen.csv |
| `groups.resnet18.corruption_aware.unseen_other_family.abs_degradation.std` | abs_degradation of resnet18 + corruption_aware, unseen_other_family (std) | std over seeds of the group's abs_degradation, ddof=1 | seen_unseen.csv |
| `groups.resnet18.corruption_aware.unseen_other_family.accuracy.mean` | accuracy of resnet18 + corruption_aware, unseen_other_family (mean) | mean over seeds of the group's accuracy | seen_unseen.csv |
| `groups.resnet18.corruption_aware.unseen_other_family.accuracy.std` | accuracy of resnet18 + corruption_aware, unseen_other_family (std) | std over seeds of the group's accuracy, ddof=1 | seen_unseen.csv |
| `groups.resnet18.corruption_aware.unseen_same_family.abs_degradation.mean` | abs_degradation of resnet18 + corruption_aware, unseen_same_family (mean) | mean over seeds of the group's abs_degradation | seen_unseen.csv |
| `groups.resnet18.corruption_aware.unseen_same_family.abs_degradation.std` | abs_degradation of resnet18 + corruption_aware, unseen_same_family (std) | std over seeds of the group's abs_degradation, ddof=1 | seen_unseen.csv |
| `groups.resnet18.corruption_aware.unseen_same_family.accuracy.mean` | accuracy of resnet18 + corruption_aware, unseen_same_family (mean) | mean over seeds of the group's accuracy | seen_unseen.csv |
| `groups.resnet18.corruption_aware.unseen_same_family.accuracy.std` | accuracy of resnet18 + corruption_aware, unseen_same_family (std) | std over seeds of the group's accuracy, ddof=1 | seen_unseen.csv |
| `groups.resnet18.standard.seen.abs_degradation.mean` | abs_degradation of resnet18 + standard, seen (mean) | mean over seeds of the group's abs_degradation | seen_unseen.csv |
| `groups.resnet18.standard.seen.abs_degradation.std` | abs_degradation of resnet18 + standard, seen (std) | std over seeds of the group's abs_degradation, ddof=1 | seen_unseen.csv |
| `groups.resnet18.standard.seen.accuracy.mean` | accuracy of resnet18 + standard, seen (mean) | mean over seeds of the group's accuracy | seen_unseen.csv |
| `groups.resnet18.standard.seen.accuracy.std` | accuracy of resnet18 + standard, seen (std) | std over seeds of the group's accuracy, ddof=1 | seen_unseen.csv |
| `groups.resnet18.standard.unseen_other_family.abs_degradation.mean` | abs_degradation of resnet18 + standard, unseen_other_family (mean) | mean over seeds of the group's abs_degradation | seen_unseen.csv |
| `groups.resnet18.standard.unseen_other_family.abs_degradation.std` | abs_degradation of resnet18 + standard, unseen_other_family (std) | std over seeds of the group's abs_degradation, ddof=1 | seen_unseen.csv |
| `groups.resnet18.standard.unseen_other_family.accuracy.mean` | accuracy of resnet18 + standard, unseen_other_family (mean) | mean over seeds of the group's accuracy | seen_unseen.csv |
| `groups.resnet18.standard.unseen_other_family.accuracy.std` | accuracy of resnet18 + standard, unseen_other_family (std) | std over seeds of the group's accuracy, ddof=1 | seen_unseen.csv |
| `groups.resnet18.standard.unseen_same_family.abs_degradation.mean` | abs_degradation of resnet18 + standard, unseen_same_family (mean) | mean over seeds of the group's abs_degradation | seen_unseen.csv |
| `groups.resnet18.standard.unseen_same_family.abs_degradation.std` | abs_degradation of resnet18 + standard, unseen_same_family (std) | std over seeds of the group's abs_degradation, ddof=1 | seen_unseen.csv |
| `groups.resnet18.standard.unseen_same_family.accuracy.mean` | accuracy of resnet18 + standard, unseen_same_family (mean) | mean over seeds of the group's accuracy | seen_unseen.csv |
| `groups.resnet18.standard.unseen_same_family.accuracy.std` | accuracy of resnet18 + standard, unseen_same_family (std) | std over seeds of the group's accuracy, ddof=1 | seen_unseen.csv |
| `main.cnn.augmix.clean_acc.mean` | clean_acc of simple_cnn + augmix (mean) | mean over seeds of clean_acc | main_results.csv |
| `main.cnn.augmix.clean_acc.std` | clean_acc of simple_cnn + augmix (std) | std over seeds of clean_acc, ddof=1 | main_results.csv |
| `main.cnn.augmix.clean_f1.mean` | clean_f1 of simple_cnn + augmix (mean) | mean over seeds of clean_f1 | main_results.csv |
| `main.cnn.augmix.clean_f1.std` | clean_f1 of simple_cnn + augmix (std) | std over seeds of clean_f1, ddof=1 | main_results.csv |
| `main.cnn.augmix.mce.mean` | mCE of simple_cnn + augmix (mean) | mean over seeds of mCE | main_results.csv |
| `main.cnn.augmix.mce.std` | mCE of simple_cnn + augmix (std) | std over seeds of mCE, ddof=1 | main_results.csv |
| `main.cnn.augmix.mean_abs_degradation.mean` | mean_abs_degradation of simple_cnn + augmix (mean) | mean over seeds of mean_abs_degradation | main_results.csv |
| `main.cnn.augmix.mean_abs_degradation.std` | mean_abs_degradation of simple_cnn + augmix (std) | std over seeds of mean_abs_degradation, ddof=1 | main_results.csv |
| `main.cnn.augmix.mean_corrupted_acc.mean` | mean_corrupted_acc of simple_cnn + augmix (mean) | mean over seeds of mean_corrupted_acc | main_results.csv |
| `main.cnn.augmix.mean_corrupted_acc.std` | mean_corrupted_acc of simple_cnn + augmix (std) | std over seeds of mean_corrupted_acc, ddof=1 | main_results.csv |
| `main.cnn.augmix.mean_corrupted_f1.mean` | mean_corrupted_f1 of simple_cnn + augmix (mean) | mean over seeds of mean_corrupted_f1 | main_results.csv |
| `main.cnn.augmix.mean_corrupted_f1.std` | mean_corrupted_f1 of simple_cnn + augmix (std) | std over seeds of mean_corrupted_f1, ddof=1 | main_results.csv |
| `main.cnn.augmix.mean_rel_degradation.mean` | mean_rel_degradation of simple_cnn + augmix (mean) | mean over seeds of mean_rel_degradation | main_results.csv |
| `main.cnn.augmix.mean_rel_degradation.std` | mean_rel_degradation of simple_cnn + augmix (std) | std over seeds of mean_rel_degradation, ddof=1 | main_results.csv |
| `main.cnn.augmix.n_seeds` | Seeds of simple_cnn + augmix | count | main_results.csv |
| `main.cnn.augmix.relative_mce.mean` | relative_mCE of simple_cnn + augmix (mean) | mean over seeds of relative_mCE | main_results.csv |
| `main.cnn.augmix.relative_mce.std` | relative_mCE of simple_cnn + augmix (std) | std over seeds of relative_mCE, ddof=1 | main_results.csv |
| `main.cnn.corruption_aware.clean_acc.mean` | clean_acc of simple_cnn + corruption_aware (mean) | mean over seeds of clean_acc | main_results.csv |
| `main.cnn.corruption_aware.clean_acc.std` | clean_acc of simple_cnn + corruption_aware (std) | std over seeds of clean_acc, ddof=1 | main_results.csv |
| `main.cnn.corruption_aware.clean_f1.mean` | clean_f1 of simple_cnn + corruption_aware (mean) | mean over seeds of clean_f1 | main_results.csv |
| `main.cnn.corruption_aware.clean_f1.std` | clean_f1 of simple_cnn + corruption_aware (std) | std over seeds of clean_f1, ddof=1 | main_results.csv |
| `main.cnn.corruption_aware.mce.mean` | mCE of simple_cnn + corruption_aware (mean) | mean over seeds of mCE | main_results.csv |
| `main.cnn.corruption_aware.mce.std` | mCE of simple_cnn + corruption_aware (std) | std over seeds of mCE, ddof=1 | main_results.csv |
| `main.cnn.corruption_aware.mean_abs_degradation.mean` | mean_abs_degradation of simple_cnn + corruption_aware (mean) | mean over seeds of mean_abs_degradation | main_results.csv |
| `main.cnn.corruption_aware.mean_abs_degradation.std` | mean_abs_degradation of simple_cnn + corruption_aware (std) | std over seeds of mean_abs_degradation, ddof=1 | main_results.csv |
| `main.cnn.corruption_aware.mean_corrupted_acc.mean` | mean_corrupted_acc of simple_cnn + corruption_aware (mean) | mean over seeds of mean_corrupted_acc | main_results.csv |
| `main.cnn.corruption_aware.mean_corrupted_acc.std` | mean_corrupted_acc of simple_cnn + corruption_aware (std) | std over seeds of mean_corrupted_acc, ddof=1 | main_results.csv |
| `main.cnn.corruption_aware.mean_corrupted_f1.mean` | mean_corrupted_f1 of simple_cnn + corruption_aware (mean) | mean over seeds of mean_corrupted_f1 | main_results.csv |
| `main.cnn.corruption_aware.mean_corrupted_f1.std` | mean_corrupted_f1 of simple_cnn + corruption_aware (std) | std over seeds of mean_corrupted_f1, ddof=1 | main_results.csv |
| `main.cnn.corruption_aware.mean_rel_degradation.mean` | mean_rel_degradation of simple_cnn + corruption_aware (mean) | mean over seeds of mean_rel_degradation | main_results.csv |
| `main.cnn.corruption_aware.mean_rel_degradation.std` | mean_rel_degradation of simple_cnn + corruption_aware (std) | std over seeds of mean_rel_degradation, ddof=1 | main_results.csv |
| `main.cnn.corruption_aware.n_seeds` | Seeds of simple_cnn + corruption_aware | count | main_results.csv |
| `main.cnn.corruption_aware.relative_mce.mean` | relative_mCE of simple_cnn + corruption_aware (mean) | mean over seeds of relative_mCE | main_results.csv |
| `main.cnn.corruption_aware.relative_mce.std` | relative_mCE of simple_cnn + corruption_aware (std) | std over seeds of relative_mCE, ddof=1 | main_results.csv |
| `main.cnn.standard.clean_acc.mean` | clean_acc of simple_cnn + standard (mean) | mean over seeds of clean_acc | main_results.csv |
| `main.cnn.standard.clean_acc.std` | clean_acc of simple_cnn + standard (std) | std over seeds of clean_acc, ddof=1 | main_results.csv |
| `main.cnn.standard.clean_f1.mean` | clean_f1 of simple_cnn + standard (mean) | mean over seeds of clean_f1 | main_results.csv |
| `main.cnn.standard.clean_f1.std` | clean_f1 of simple_cnn + standard (std) | std over seeds of clean_f1, ddof=1 | main_results.csv |
| `main.cnn.standard.mce.mean` | mCE of simple_cnn + standard (mean) | mean over seeds of mCE | main_results.csv |
| `main.cnn.standard.mce.std` | mCE of simple_cnn + standard (std) | std over seeds of mCE, ddof=1 | main_results.csv |
| `main.cnn.standard.mean_abs_degradation.mean` | mean_abs_degradation of simple_cnn + standard (mean) | mean over seeds of mean_abs_degradation | main_results.csv |
| `main.cnn.standard.mean_abs_degradation.std` | mean_abs_degradation of simple_cnn + standard (std) | std over seeds of mean_abs_degradation, ddof=1 | main_results.csv |
| `main.cnn.standard.mean_corrupted_acc.mean` | mean_corrupted_acc of simple_cnn + standard (mean) | mean over seeds of mean_corrupted_acc | main_results.csv |
| `main.cnn.standard.mean_corrupted_acc.std` | mean_corrupted_acc of simple_cnn + standard (std) | std over seeds of mean_corrupted_acc, ddof=1 | main_results.csv |
| `main.cnn.standard.mean_corrupted_f1.mean` | mean_corrupted_f1 of simple_cnn + standard (mean) | mean over seeds of mean_corrupted_f1 | main_results.csv |
| `main.cnn.standard.mean_corrupted_f1.std` | mean_corrupted_f1 of simple_cnn + standard (std) | std over seeds of mean_corrupted_f1, ddof=1 | main_results.csv |
| `main.cnn.standard.mean_rel_degradation.mean` | mean_rel_degradation of simple_cnn + standard (mean) | mean over seeds of mean_rel_degradation | main_results.csv |
| `main.cnn.standard.mean_rel_degradation.std` | mean_rel_degradation of simple_cnn + standard (std) | std over seeds of mean_rel_degradation, ddof=1 | main_results.csv |
| `main.cnn.standard.n_seeds` | Seeds of simple_cnn + standard | count | main_results.csv |
| `main.cnn.standard.relative_mce.mean` | relative_mCE of simple_cnn + standard (mean) | mean over seeds of relative_mCE | main_results.csv |
| `main.cnn.standard.relative_mce.std` | relative_mCE of simple_cnn + standard (std) | std over seeds of relative_mCE, ddof=1 | main_results.csv |
| `main.resnet18.augmix.clean_acc.mean` | clean_acc of resnet18 + augmix (mean) | mean over seeds of clean_acc | main_results.csv |
| `main.resnet18.augmix.clean_acc.std` | clean_acc of resnet18 + augmix (std) | std over seeds of clean_acc, ddof=1 | main_results.csv |
| `main.resnet18.augmix.clean_f1.mean` | clean_f1 of resnet18 + augmix (mean) | mean over seeds of clean_f1 | main_results.csv |
| `main.resnet18.augmix.clean_f1.std` | clean_f1 of resnet18 + augmix (std) | std over seeds of clean_f1, ddof=1 | main_results.csv |
| `main.resnet18.augmix.mce.mean` | mCE of resnet18 + augmix (mean) | mean over seeds of mCE | main_results.csv |
| `main.resnet18.augmix.mce.std` | mCE of resnet18 + augmix (std) | std over seeds of mCE, ddof=1 | main_results.csv |
| `main.resnet18.augmix.mean_abs_degradation.mean` | mean_abs_degradation of resnet18 + augmix (mean) | mean over seeds of mean_abs_degradation | main_results.csv |
| `main.resnet18.augmix.mean_abs_degradation.std` | mean_abs_degradation of resnet18 + augmix (std) | std over seeds of mean_abs_degradation, ddof=1 | main_results.csv |
| `main.resnet18.augmix.mean_corrupted_acc.mean` | mean_corrupted_acc of resnet18 + augmix (mean) | mean over seeds of mean_corrupted_acc | main_results.csv |
| `main.resnet18.augmix.mean_corrupted_acc.std` | mean_corrupted_acc of resnet18 + augmix (std) | std over seeds of mean_corrupted_acc, ddof=1 | main_results.csv |
| `main.resnet18.augmix.mean_corrupted_f1.mean` | mean_corrupted_f1 of resnet18 + augmix (mean) | mean over seeds of mean_corrupted_f1 | main_results.csv |
| `main.resnet18.augmix.mean_corrupted_f1.std` | mean_corrupted_f1 of resnet18 + augmix (std) | std over seeds of mean_corrupted_f1, ddof=1 | main_results.csv |
| `main.resnet18.augmix.mean_rel_degradation.mean` | mean_rel_degradation of resnet18 + augmix (mean) | mean over seeds of mean_rel_degradation | main_results.csv |
| `main.resnet18.augmix.mean_rel_degradation.std` | mean_rel_degradation of resnet18 + augmix (std) | std over seeds of mean_rel_degradation, ddof=1 | main_results.csv |
| `main.resnet18.augmix.n_seeds` | Seeds of resnet18 + augmix | count | main_results.csv |
| `main.resnet18.augmix.relative_mce.mean` | relative_mCE of resnet18 + augmix (mean) | mean over seeds of relative_mCE | main_results.csv |
| `main.resnet18.augmix.relative_mce.std` | relative_mCE of resnet18 + augmix (std) | std over seeds of relative_mCE, ddof=1 | main_results.csv |
| `main.resnet18.corruption_aware.clean_acc.mean` | clean_acc of resnet18 + corruption_aware (mean) | mean over seeds of clean_acc | main_results.csv |
| `main.resnet18.corruption_aware.clean_acc.std` | clean_acc of resnet18 + corruption_aware (std) | std over seeds of clean_acc, ddof=1 | main_results.csv |
| `main.resnet18.corruption_aware.clean_f1.mean` | clean_f1 of resnet18 + corruption_aware (mean) | mean over seeds of clean_f1 | main_results.csv |
| `main.resnet18.corruption_aware.clean_f1.std` | clean_f1 of resnet18 + corruption_aware (std) | std over seeds of clean_f1, ddof=1 | main_results.csv |
| `main.resnet18.corruption_aware.mce.mean` | mCE of resnet18 + corruption_aware (mean) | mean over seeds of mCE | main_results.csv |
| `main.resnet18.corruption_aware.mce.std` | mCE of resnet18 + corruption_aware (std) | std over seeds of mCE, ddof=1 | main_results.csv |
| `main.resnet18.corruption_aware.mean_abs_degradation.mean` | mean_abs_degradation of resnet18 + corruption_aware (mean) | mean over seeds of mean_abs_degradation | main_results.csv |
| `main.resnet18.corruption_aware.mean_abs_degradation.std` | mean_abs_degradation of resnet18 + corruption_aware (std) | std over seeds of mean_abs_degradation, ddof=1 | main_results.csv |
| `main.resnet18.corruption_aware.mean_corrupted_acc.mean` | mean_corrupted_acc of resnet18 + corruption_aware (mean) | mean over seeds of mean_corrupted_acc | main_results.csv |
| `main.resnet18.corruption_aware.mean_corrupted_acc.std` | mean_corrupted_acc of resnet18 + corruption_aware (std) | std over seeds of mean_corrupted_acc, ddof=1 | main_results.csv |
| `main.resnet18.corruption_aware.mean_corrupted_f1.mean` | mean_corrupted_f1 of resnet18 + corruption_aware (mean) | mean over seeds of mean_corrupted_f1 | main_results.csv |
| `main.resnet18.corruption_aware.mean_corrupted_f1.std` | mean_corrupted_f1 of resnet18 + corruption_aware (std) | std over seeds of mean_corrupted_f1, ddof=1 | main_results.csv |
| `main.resnet18.corruption_aware.mean_rel_degradation.mean` | mean_rel_degradation of resnet18 + corruption_aware (mean) | mean over seeds of mean_rel_degradation | main_results.csv |
| `main.resnet18.corruption_aware.mean_rel_degradation.std` | mean_rel_degradation of resnet18 + corruption_aware (std) | std over seeds of mean_rel_degradation, ddof=1 | main_results.csv |
| `main.resnet18.corruption_aware.n_seeds` | Seeds of resnet18 + corruption_aware | count | main_results.csv |
| `main.resnet18.corruption_aware.relative_mce.mean` | relative_mCE of resnet18 + corruption_aware (mean) | mean over seeds of relative_mCE | main_results.csv |
| `main.resnet18.corruption_aware.relative_mce.std` | relative_mCE of resnet18 + corruption_aware (std) | std over seeds of relative_mCE, ddof=1 | main_results.csv |
| `main.resnet18.standard.clean_acc.mean` | clean_acc of resnet18 + standard (mean) | mean over seeds of clean_acc | main_results.csv |
| `main.resnet18.standard.clean_acc.std` | clean_acc of resnet18 + standard (std) | std over seeds of clean_acc, ddof=1 | main_results.csv |
| `main.resnet18.standard.clean_f1.mean` | clean_f1 of resnet18 + standard (mean) | mean over seeds of clean_f1 | main_results.csv |
| `main.resnet18.standard.clean_f1.std` | clean_f1 of resnet18 + standard (std) | std over seeds of clean_f1, ddof=1 | main_results.csv |
| `main.resnet18.standard.mce.mean` | mCE of resnet18 + standard (mean) | mean over seeds of mCE | main_results.csv |
| `main.resnet18.standard.mce.std` | mCE of resnet18 + standard (std) | std over seeds of mCE, ddof=1 | main_results.csv |
| `main.resnet18.standard.mean_abs_degradation.mean` | mean_abs_degradation of resnet18 + standard (mean) | mean over seeds of mean_abs_degradation | main_results.csv |
| `main.resnet18.standard.mean_abs_degradation.std` | mean_abs_degradation of resnet18 + standard (std) | std over seeds of mean_abs_degradation, ddof=1 | main_results.csv |
| `main.resnet18.standard.mean_corrupted_acc.mean` | mean_corrupted_acc of resnet18 + standard (mean) | mean over seeds of mean_corrupted_acc | main_results.csv |
| `main.resnet18.standard.mean_corrupted_acc.std` | mean_corrupted_acc of resnet18 + standard (std) | std over seeds of mean_corrupted_acc, ddof=1 | main_results.csv |
| `main.resnet18.standard.mean_corrupted_f1.mean` | mean_corrupted_f1 of resnet18 + standard (mean) | mean over seeds of mean_corrupted_f1 | main_results.csv |
| `main.resnet18.standard.mean_corrupted_f1.std` | mean_corrupted_f1 of resnet18 + standard (std) | std over seeds of mean_corrupted_f1, ddof=1 | main_results.csv |
| `main.resnet18.standard.mean_rel_degradation.mean` | mean_rel_degradation of resnet18 + standard (mean) | mean over seeds of mean_rel_degradation | main_results.csv |
| `main.resnet18.standard.mean_rel_degradation.std` | mean_rel_degradation of resnet18 + standard (std) | std over seeds of mean_rel_degradation, ddof=1 | main_results.csv |
| `main.resnet18.standard.n_seeds` | Seeds of resnet18 + standard | count | main_results.csv |
| `main.resnet18.standard.relative_mce.mean` | relative_mCE of resnet18 + standard (mean) | mean over seeds of relative_mCE | main_results.csv |
| `main.resnet18.standard.relative_mce.std` | relative_mCE of resnet18 + standard (std) | std over seeds of relative_mCE, ddof=1 | main_results.csv |
| `meta.baseline_seeds` | Seeds in the mCE baseline | count | results/raw/ |
| `meta.n_runs` | Runs analysed | count of result CSVs | results/raw/ |
| `meta.status` | FULL or PARTIAL data | all 18 runs present? | results/raw/ |
| `meta.std_ddof` | ddof of every std | constant | src/analysis.py |
| `rq1.cnn.augmix.abs_drop_sev1` | Absolute accuracy drop at severity 1, simple_cnn + augmix | seed mean of (clean - mean accuracy at the severity) | per_severity.csv |
| `rq1.cnn.augmix.abs_drop_sev2` | Absolute accuracy drop at severity 2, simple_cnn + augmix | seed mean of (clean - mean accuracy at the severity) | per_severity.csv |
| `rq1.cnn.augmix.abs_drop_sev3` | Absolute accuracy drop at severity 3, simple_cnn + augmix | seed mean of (clean - mean accuracy at the severity) | per_severity.csv |
| `rq1.cnn.augmix.abs_drop_sev4` | Absolute accuracy drop at severity 4, simple_cnn + augmix | seed mean of (clean - mean accuracy at the severity) | per_severity.csv |
| `rq1.cnn.augmix.abs_drop_sev5` | Absolute accuracy drop at severity 5, simple_cnn + augmix | seed mean of (clean - mean accuracy at the severity) | per_severity.csv |
| `rq1.cnn.augmix.drop_grows_with_severity` | Mean drop rises at every severity step, simple_cnn + augmix | drop at s < drop at s+1, s = 1..4 | per_severity.csv |
| `rq1.cnn.augmix.rel_drop_sev1` | Relative accuracy drop at severity 1, simple_cnn + augmix | seed mean of (clean - mean accuracy at the severity) / clean | per_severity.csv |
| `rq1.cnn.augmix.rel_drop_sev2` | Relative accuracy drop at severity 2, simple_cnn + augmix | seed mean of (clean - mean accuracy at the severity) / clean | per_severity.csv |
| `rq1.cnn.augmix.rel_drop_sev3` | Relative accuracy drop at severity 3, simple_cnn + augmix | seed mean of (clean - mean accuracy at the severity) / clean | per_severity.csv |
| `rq1.cnn.augmix.rel_drop_sev4` | Relative accuracy drop at severity 4, simple_cnn + augmix | seed mean of (clean - mean accuracy at the severity) / clean | per_severity.csv |
| `rq1.cnn.augmix.rel_drop_sev5` | Relative accuracy drop at severity 5, simple_cnn + augmix | seed mean of (clean - mean accuracy at the severity) / clean | per_severity.csv |
| `rq1.cnn.corruption_aware.abs_drop_sev1` | Absolute accuracy drop at severity 1, simple_cnn + corruption_aware | seed mean of (clean - mean accuracy at the severity) | per_severity.csv |
| `rq1.cnn.corruption_aware.abs_drop_sev2` | Absolute accuracy drop at severity 2, simple_cnn + corruption_aware | seed mean of (clean - mean accuracy at the severity) | per_severity.csv |
| `rq1.cnn.corruption_aware.abs_drop_sev3` | Absolute accuracy drop at severity 3, simple_cnn + corruption_aware | seed mean of (clean - mean accuracy at the severity) | per_severity.csv |
| `rq1.cnn.corruption_aware.abs_drop_sev4` | Absolute accuracy drop at severity 4, simple_cnn + corruption_aware | seed mean of (clean - mean accuracy at the severity) | per_severity.csv |
| `rq1.cnn.corruption_aware.abs_drop_sev5` | Absolute accuracy drop at severity 5, simple_cnn + corruption_aware | seed mean of (clean - mean accuracy at the severity) | per_severity.csv |
| `rq1.cnn.corruption_aware.drop_grows_with_severity` | Mean drop rises at every severity step, simple_cnn + corruption_aware | drop at s < drop at s+1, s = 1..4 | per_severity.csv |
| `rq1.cnn.corruption_aware.rel_drop_sev1` | Relative accuracy drop at severity 1, simple_cnn + corruption_aware | seed mean of (clean - mean accuracy at the severity) / clean | per_severity.csv |
| `rq1.cnn.corruption_aware.rel_drop_sev2` | Relative accuracy drop at severity 2, simple_cnn + corruption_aware | seed mean of (clean - mean accuracy at the severity) / clean | per_severity.csv |
| `rq1.cnn.corruption_aware.rel_drop_sev3` | Relative accuracy drop at severity 3, simple_cnn + corruption_aware | seed mean of (clean - mean accuracy at the severity) / clean | per_severity.csv |
| `rq1.cnn.corruption_aware.rel_drop_sev4` | Relative accuracy drop at severity 4, simple_cnn + corruption_aware | seed mean of (clean - mean accuracy at the severity) / clean | per_severity.csv |
| `rq1.cnn.corruption_aware.rel_drop_sev5` | Relative accuracy drop at severity 5, simple_cnn + corruption_aware | seed mean of (clean - mean accuracy at the severity) / clean | per_severity.csv |
| `rq1.cnn.standard.abs_drop_sev1` | Absolute accuracy drop at severity 1, simple_cnn + standard | seed mean of (clean - mean accuracy at the severity) | per_severity.csv |
| `rq1.cnn.standard.abs_drop_sev2` | Absolute accuracy drop at severity 2, simple_cnn + standard | seed mean of (clean - mean accuracy at the severity) | per_severity.csv |
| `rq1.cnn.standard.abs_drop_sev3` | Absolute accuracy drop at severity 3, simple_cnn + standard | seed mean of (clean - mean accuracy at the severity) | per_severity.csv |
| `rq1.cnn.standard.abs_drop_sev4` | Absolute accuracy drop at severity 4, simple_cnn + standard | seed mean of (clean - mean accuracy at the severity) | per_severity.csv |
| `rq1.cnn.standard.abs_drop_sev5` | Absolute accuracy drop at severity 5, simple_cnn + standard | seed mean of (clean - mean accuracy at the severity) | per_severity.csv |
| `rq1.cnn.standard.drop_grows_with_severity` | Mean drop rises at every severity step, simple_cnn + standard | drop at s < drop at s+1, s = 1..4 | per_severity.csv |
| `rq1.cnn.standard.rel_drop_sev1` | Relative accuracy drop at severity 1, simple_cnn + standard | seed mean of (clean - mean accuracy at the severity) / clean | per_severity.csv |
| `rq1.cnn.standard.rel_drop_sev2` | Relative accuracy drop at severity 2, simple_cnn + standard | seed mean of (clean - mean accuracy at the severity) / clean | per_severity.csv |
| `rq1.cnn.standard.rel_drop_sev3` | Relative accuracy drop at severity 3, simple_cnn + standard | seed mean of (clean - mean accuracy at the severity) / clean | per_severity.csv |
| `rq1.cnn.standard.rel_drop_sev4` | Relative accuracy drop at severity 4, simple_cnn + standard | seed mean of (clean - mean accuracy at the severity) / clean | per_severity.csv |
| `rq1.cnn.standard.rel_drop_sev5` | Relative accuracy drop at severity 5, simple_cnn + standard | seed mean of (clean - mean accuracy at the severity) / clean | per_severity.csv |
| `rq1.resnet18.augmix.abs_drop_sev1` | Absolute accuracy drop at severity 1, resnet18 + augmix | seed mean of (clean - mean accuracy at the severity) | per_severity.csv |
| `rq1.resnet18.augmix.abs_drop_sev2` | Absolute accuracy drop at severity 2, resnet18 + augmix | seed mean of (clean - mean accuracy at the severity) | per_severity.csv |
| `rq1.resnet18.augmix.abs_drop_sev3` | Absolute accuracy drop at severity 3, resnet18 + augmix | seed mean of (clean - mean accuracy at the severity) | per_severity.csv |
| `rq1.resnet18.augmix.abs_drop_sev4` | Absolute accuracy drop at severity 4, resnet18 + augmix | seed mean of (clean - mean accuracy at the severity) | per_severity.csv |
| `rq1.resnet18.augmix.abs_drop_sev5` | Absolute accuracy drop at severity 5, resnet18 + augmix | seed mean of (clean - mean accuracy at the severity) | per_severity.csv |
| `rq1.resnet18.augmix.drop_grows_with_severity` | Mean drop rises at every severity step, resnet18 + augmix | drop at s < drop at s+1, s = 1..4 | per_severity.csv |
| `rq1.resnet18.augmix.rel_drop_sev1` | Relative accuracy drop at severity 1, resnet18 + augmix | seed mean of (clean - mean accuracy at the severity) / clean | per_severity.csv |
| `rq1.resnet18.augmix.rel_drop_sev2` | Relative accuracy drop at severity 2, resnet18 + augmix | seed mean of (clean - mean accuracy at the severity) / clean | per_severity.csv |
| `rq1.resnet18.augmix.rel_drop_sev3` | Relative accuracy drop at severity 3, resnet18 + augmix | seed mean of (clean - mean accuracy at the severity) / clean | per_severity.csv |
| `rq1.resnet18.augmix.rel_drop_sev4` | Relative accuracy drop at severity 4, resnet18 + augmix | seed mean of (clean - mean accuracy at the severity) / clean | per_severity.csv |
| `rq1.resnet18.augmix.rel_drop_sev5` | Relative accuracy drop at severity 5, resnet18 + augmix | seed mean of (clean - mean accuracy at the severity) / clean | per_severity.csv |
| `rq1.resnet18.corruption_aware.abs_drop_sev1` | Absolute accuracy drop at severity 1, resnet18 + corruption_aware | seed mean of (clean - mean accuracy at the severity) | per_severity.csv |
| `rq1.resnet18.corruption_aware.abs_drop_sev2` | Absolute accuracy drop at severity 2, resnet18 + corruption_aware | seed mean of (clean - mean accuracy at the severity) | per_severity.csv |
| `rq1.resnet18.corruption_aware.abs_drop_sev3` | Absolute accuracy drop at severity 3, resnet18 + corruption_aware | seed mean of (clean - mean accuracy at the severity) | per_severity.csv |
| `rq1.resnet18.corruption_aware.abs_drop_sev4` | Absolute accuracy drop at severity 4, resnet18 + corruption_aware | seed mean of (clean - mean accuracy at the severity) | per_severity.csv |
| `rq1.resnet18.corruption_aware.abs_drop_sev5` | Absolute accuracy drop at severity 5, resnet18 + corruption_aware | seed mean of (clean - mean accuracy at the severity) | per_severity.csv |
| `rq1.resnet18.corruption_aware.drop_grows_with_severity` | Mean drop rises at every severity step, resnet18 + corruption_aware | drop at s < drop at s+1, s = 1..4 | per_severity.csv |
| `rq1.resnet18.corruption_aware.rel_drop_sev1` | Relative accuracy drop at severity 1, resnet18 + corruption_aware | seed mean of (clean - mean accuracy at the severity) / clean | per_severity.csv |
| `rq1.resnet18.corruption_aware.rel_drop_sev2` | Relative accuracy drop at severity 2, resnet18 + corruption_aware | seed mean of (clean - mean accuracy at the severity) / clean | per_severity.csv |
| `rq1.resnet18.corruption_aware.rel_drop_sev3` | Relative accuracy drop at severity 3, resnet18 + corruption_aware | seed mean of (clean - mean accuracy at the severity) / clean | per_severity.csv |
| `rq1.resnet18.corruption_aware.rel_drop_sev4` | Relative accuracy drop at severity 4, resnet18 + corruption_aware | seed mean of (clean - mean accuracy at the severity) / clean | per_severity.csv |
| `rq1.resnet18.corruption_aware.rel_drop_sev5` | Relative accuracy drop at severity 5, resnet18 + corruption_aware | seed mean of (clean - mean accuracy at the severity) / clean | per_severity.csv |
| `rq1.resnet18.standard.abs_drop_sev1` | Absolute accuracy drop at severity 1, resnet18 + standard | seed mean of (clean - mean accuracy at the severity) | per_severity.csv |
| `rq1.resnet18.standard.abs_drop_sev2` | Absolute accuracy drop at severity 2, resnet18 + standard | seed mean of (clean - mean accuracy at the severity) | per_severity.csv |
| `rq1.resnet18.standard.abs_drop_sev3` | Absolute accuracy drop at severity 3, resnet18 + standard | seed mean of (clean - mean accuracy at the severity) | per_severity.csv |
| `rq1.resnet18.standard.abs_drop_sev4` | Absolute accuracy drop at severity 4, resnet18 + standard | seed mean of (clean - mean accuracy at the severity) | per_severity.csv |
| `rq1.resnet18.standard.abs_drop_sev5` | Absolute accuracy drop at severity 5, resnet18 + standard | seed mean of (clean - mean accuracy at the severity) | per_severity.csv |
| `rq1.resnet18.standard.drop_grows_with_severity` | Mean drop rises at every severity step, resnet18 + standard | drop at s < drop at s+1, s = 1..4 | per_severity.csv |
| `rq1.resnet18.standard.rel_drop_sev1` | Relative accuracy drop at severity 1, resnet18 + standard | seed mean of (clean - mean accuracy at the severity) / clean | per_severity.csv |
| `rq1.resnet18.standard.rel_drop_sev2` | Relative accuracy drop at severity 2, resnet18 + standard | seed mean of (clean - mean accuracy at the severity) / clean | per_severity.csv |
| `rq1.resnet18.standard.rel_drop_sev3` | Relative accuracy drop at severity 3, resnet18 + standard | seed mean of (clean - mean accuracy at the severity) / clean | per_severity.csv |
| `rq1.resnet18.standard.rel_drop_sev4` | Relative accuracy drop at severity 4, resnet18 + standard | seed mean of (clean - mean accuracy at the severity) / clean | per_severity.csv |
| `rq1.resnet18.standard.rel_drop_sev5` | Relative accuracy drop at severity 5, resnet18 + standard | seed mean of (clean - mean accuracy at the severity) / clean | per_severity.csv |
| `rq2.augmix.clean_acc.resnet18_minus_cnn` | ResNet-18 minus Simple CNN, clean_acc, augmix | mean over shared seeds of (a - b) | main_results.csv |
| `rq2.augmix.clean_acc.resnet18_minus_cnn.seed0` | ResNet-18 minus Simple CNN, clean_acc, augmix, seed 0 | a - b | main_results.csv |
| `rq2.augmix.clean_acc.resnet18_minus_cnn.seed1` | ResNet-18 minus Simple CNN, clean_acc, augmix, seed 1 | a - b | main_results.csv |
| `rq2.augmix.clean_acc.resnet18_minus_cnn.seed2` | ResNet-18 minus Simple CNN, clean_acc, augmix, seed 2 | a - b | main_results.csv |
| `rq2.augmix.clean_acc.resnet18_minus_cnn.seed_noise` | Seed-noise label: ResNet-18 minus Simple CNN, clean_acc, augmix | \|mean difference\| < larger of the two seed stds | main_results.csv |
| `rq2.augmix.mce.resnet18_minus_cnn` | ResNet-18 minus Simple CNN, mCE, augmix | mean over shared seeds of (a - b) | main_results.csv |
| `rq2.augmix.mce.resnet18_minus_cnn.seed0` | ResNet-18 minus Simple CNN, mCE, augmix, seed 0 | a - b | main_results.csv |
| `rq2.augmix.mce.resnet18_minus_cnn.seed1` | ResNet-18 minus Simple CNN, mCE, augmix, seed 1 | a - b | main_results.csv |
| `rq2.augmix.mce.resnet18_minus_cnn.seed2` | ResNet-18 minus Simple CNN, mCE, augmix, seed 2 | a - b | main_results.csv |
| `rq2.augmix.mce.resnet18_minus_cnn.seed_noise` | Seed-noise label: ResNet-18 minus Simple CNN, mCE, augmix | \|mean difference\| < larger of the two seed stds | main_results.csv |
| `rq2.augmix.mean_abs_degradation.resnet18_minus_cnn` | ResNet-18 minus Simple CNN, mean_abs_degradation, augmix | mean over shared seeds of (a - b) | main_results.csv |
| `rq2.augmix.mean_abs_degradation.resnet18_minus_cnn.seed0` | ResNet-18 minus Simple CNN, mean_abs_degradation, augmix, seed 0 | a - b | main_results.csv |
| `rq2.augmix.mean_abs_degradation.resnet18_minus_cnn.seed1` | ResNet-18 minus Simple CNN, mean_abs_degradation, augmix, seed 1 | a - b | main_results.csv |
| `rq2.augmix.mean_abs_degradation.resnet18_minus_cnn.seed2` | ResNet-18 minus Simple CNN, mean_abs_degradation, augmix, seed 2 | a - b | main_results.csv |
| `rq2.augmix.mean_abs_degradation.resnet18_minus_cnn.seed_noise` | Seed-noise label: ResNet-18 minus Simple CNN, mean_abs_degradation, augmix | \|mean difference\| < larger of the two seed stds | main_results.csv |
| `rq2.augmix.mean_corrupted_acc.resnet18_minus_cnn` | ResNet-18 minus Simple CNN, mean_corrupted_acc, augmix | mean over shared seeds of (a - b) | main_results.csv |
| `rq2.augmix.mean_corrupted_acc.resnet18_minus_cnn.seed0` | ResNet-18 minus Simple CNN, mean_corrupted_acc, augmix, seed 0 | a - b | main_results.csv |
| `rq2.augmix.mean_corrupted_acc.resnet18_minus_cnn.seed1` | ResNet-18 minus Simple CNN, mean_corrupted_acc, augmix, seed 1 | a - b | main_results.csv |
| `rq2.augmix.mean_corrupted_acc.resnet18_minus_cnn.seed2` | ResNet-18 minus Simple CNN, mean_corrupted_acc, augmix, seed 2 | a - b | main_results.csv |
| `rq2.augmix.mean_corrupted_acc.resnet18_minus_cnn.seed_noise` | Seed-noise label: ResNet-18 minus Simple CNN, mean_corrupted_acc, augmix | \|mean difference\| < larger of the two seed stds | main_results.csv |
| `rq2.augmix.mean_rel_degradation.resnet18_minus_cnn` | ResNet-18 minus Simple CNN, mean_rel_degradation, augmix | mean over shared seeds of (a - b) | main_results.csv |
| `rq2.augmix.mean_rel_degradation.resnet18_minus_cnn.seed0` | ResNet-18 minus Simple CNN, mean_rel_degradation, augmix, seed 0 | a - b | main_results.csv |
| `rq2.augmix.mean_rel_degradation.resnet18_minus_cnn.seed1` | ResNet-18 minus Simple CNN, mean_rel_degradation, augmix, seed 1 | a - b | main_results.csv |
| `rq2.augmix.mean_rel_degradation.resnet18_minus_cnn.seed2` | ResNet-18 minus Simple CNN, mean_rel_degradation, augmix, seed 2 | a - b | main_results.csv |
| `rq2.augmix.mean_rel_degradation.resnet18_minus_cnn.seed_noise` | Seed-noise label: ResNet-18 minus Simple CNN, mean_rel_degradation, augmix | \|mean difference\| < larger of the two seed stds | main_results.csv |
| `rq2.augmix.relative_mce.resnet18_minus_cnn` | ResNet-18 minus Simple CNN, relative_mCE, augmix | mean over shared seeds of (a - b) | main_results.csv |
| `rq2.augmix.relative_mce.resnet18_minus_cnn.seed0` | ResNet-18 minus Simple CNN, relative_mCE, augmix, seed 0 | a - b | main_results.csv |
| `rq2.augmix.relative_mce.resnet18_minus_cnn.seed1` | ResNet-18 minus Simple CNN, relative_mCE, augmix, seed 1 | a - b | main_results.csv |
| `rq2.augmix.relative_mce.resnet18_minus_cnn.seed2` | ResNet-18 minus Simple CNN, relative_mCE, augmix, seed 2 | a - b | main_results.csv |
| `rq2.augmix.relative_mce.resnet18_minus_cnn.seed_noise` | Seed-noise label: ResNet-18 minus Simple CNN, relative_mCE, augmix | \|mean difference\| < larger of the two seed stds | main_results.csv |
| `rq2.corruption_aware.clean_acc.resnet18_minus_cnn` | ResNet-18 minus Simple CNN, clean_acc, corruption_aware | mean over shared seeds of (a - b) | main_results.csv |
| `rq2.corruption_aware.clean_acc.resnet18_minus_cnn.seed0` | ResNet-18 minus Simple CNN, clean_acc, corruption_aware, seed 0 | a - b | main_results.csv |
| `rq2.corruption_aware.clean_acc.resnet18_minus_cnn.seed1` | ResNet-18 minus Simple CNN, clean_acc, corruption_aware, seed 1 | a - b | main_results.csv |
| `rq2.corruption_aware.clean_acc.resnet18_minus_cnn.seed2` | ResNet-18 minus Simple CNN, clean_acc, corruption_aware, seed 2 | a - b | main_results.csv |
| `rq2.corruption_aware.clean_acc.resnet18_minus_cnn.seed_noise` | Seed-noise label: ResNet-18 minus Simple CNN, clean_acc, corruption_aware | \|mean difference\| < larger of the two seed stds | main_results.csv |
| `rq2.corruption_aware.mce.resnet18_minus_cnn` | ResNet-18 minus Simple CNN, mCE, corruption_aware | mean over shared seeds of (a - b) | main_results.csv |
| `rq2.corruption_aware.mce.resnet18_minus_cnn.seed0` | ResNet-18 minus Simple CNN, mCE, corruption_aware, seed 0 | a - b | main_results.csv |
| `rq2.corruption_aware.mce.resnet18_minus_cnn.seed1` | ResNet-18 minus Simple CNN, mCE, corruption_aware, seed 1 | a - b | main_results.csv |
| `rq2.corruption_aware.mce.resnet18_minus_cnn.seed2` | ResNet-18 minus Simple CNN, mCE, corruption_aware, seed 2 | a - b | main_results.csv |
| `rq2.corruption_aware.mce.resnet18_minus_cnn.seed_noise` | Seed-noise label: ResNet-18 minus Simple CNN, mCE, corruption_aware | \|mean difference\| < larger of the two seed stds | main_results.csv |
| `rq2.corruption_aware.mean_abs_degradation.resnet18_minus_cnn` | ResNet-18 minus Simple CNN, mean_abs_degradation, corruption_aware | mean over shared seeds of (a - b) | main_results.csv |
| `rq2.corruption_aware.mean_abs_degradation.resnet18_minus_cnn.seed0` | ResNet-18 minus Simple CNN, mean_abs_degradation, corruption_aware, seed 0 | a - b | main_results.csv |
| `rq2.corruption_aware.mean_abs_degradation.resnet18_minus_cnn.seed1` | ResNet-18 minus Simple CNN, mean_abs_degradation, corruption_aware, seed 1 | a - b | main_results.csv |
| `rq2.corruption_aware.mean_abs_degradation.resnet18_minus_cnn.seed2` | ResNet-18 minus Simple CNN, mean_abs_degradation, corruption_aware, seed 2 | a - b | main_results.csv |
| `rq2.corruption_aware.mean_abs_degradation.resnet18_minus_cnn.seed_noise` | Seed-noise label: ResNet-18 minus Simple CNN, mean_abs_degradation, corruption_aware | \|mean difference\| < larger of the two seed stds | main_results.csv |
| `rq2.corruption_aware.mean_corrupted_acc.resnet18_minus_cnn` | ResNet-18 minus Simple CNN, mean_corrupted_acc, corruption_aware | mean over shared seeds of (a - b) | main_results.csv |
| `rq2.corruption_aware.mean_corrupted_acc.resnet18_minus_cnn.seed0` | ResNet-18 minus Simple CNN, mean_corrupted_acc, corruption_aware, seed 0 | a - b | main_results.csv |
| `rq2.corruption_aware.mean_corrupted_acc.resnet18_minus_cnn.seed1` | ResNet-18 minus Simple CNN, mean_corrupted_acc, corruption_aware, seed 1 | a - b | main_results.csv |
| `rq2.corruption_aware.mean_corrupted_acc.resnet18_minus_cnn.seed2` | ResNet-18 minus Simple CNN, mean_corrupted_acc, corruption_aware, seed 2 | a - b | main_results.csv |
| `rq2.corruption_aware.mean_corrupted_acc.resnet18_minus_cnn.seed_noise` | Seed-noise label: ResNet-18 minus Simple CNN, mean_corrupted_acc, corruption_aware | \|mean difference\| < larger of the two seed stds | main_results.csv |
| `rq2.corruption_aware.mean_rel_degradation.resnet18_minus_cnn` | ResNet-18 minus Simple CNN, mean_rel_degradation, corruption_aware | mean over shared seeds of (a - b) | main_results.csv |
| `rq2.corruption_aware.mean_rel_degradation.resnet18_minus_cnn.seed0` | ResNet-18 minus Simple CNN, mean_rel_degradation, corruption_aware, seed 0 | a - b | main_results.csv |
| `rq2.corruption_aware.mean_rel_degradation.resnet18_minus_cnn.seed1` | ResNet-18 minus Simple CNN, mean_rel_degradation, corruption_aware, seed 1 | a - b | main_results.csv |
| `rq2.corruption_aware.mean_rel_degradation.resnet18_minus_cnn.seed2` | ResNet-18 minus Simple CNN, mean_rel_degradation, corruption_aware, seed 2 | a - b | main_results.csv |
| `rq2.corruption_aware.mean_rel_degradation.resnet18_minus_cnn.seed_noise` | Seed-noise label: ResNet-18 minus Simple CNN, mean_rel_degradation, corruption_aware | \|mean difference\| < larger of the two seed stds | main_results.csv |
| `rq2.corruption_aware.relative_mce.resnet18_minus_cnn` | ResNet-18 minus Simple CNN, relative_mCE, corruption_aware | mean over shared seeds of (a - b) | main_results.csv |
| `rq2.corruption_aware.relative_mce.resnet18_minus_cnn.seed0` | ResNet-18 minus Simple CNN, relative_mCE, corruption_aware, seed 0 | a - b | main_results.csv |
| `rq2.corruption_aware.relative_mce.resnet18_minus_cnn.seed1` | ResNet-18 minus Simple CNN, relative_mCE, corruption_aware, seed 1 | a - b | main_results.csv |
| `rq2.corruption_aware.relative_mce.resnet18_minus_cnn.seed2` | ResNet-18 minus Simple CNN, relative_mCE, corruption_aware, seed 2 | a - b | main_results.csv |
| `rq2.corruption_aware.relative_mce.resnet18_minus_cnn.seed_noise` | Seed-noise label: ResNet-18 minus Simple CNN, relative_mCE, corruption_aware | \|mean difference\| < larger of the two seed stds | main_results.csv |
| `rq2.standard.clean_acc.resnet18_minus_cnn` | ResNet-18 minus Simple CNN, clean_acc, standard | mean over shared seeds of (a - b) | main_results.csv |
| `rq2.standard.clean_acc.resnet18_minus_cnn.seed0` | ResNet-18 minus Simple CNN, clean_acc, standard, seed 0 | a - b | main_results.csv |
| `rq2.standard.clean_acc.resnet18_minus_cnn.seed1` | ResNet-18 minus Simple CNN, clean_acc, standard, seed 1 | a - b | main_results.csv |
| `rq2.standard.clean_acc.resnet18_minus_cnn.seed2` | ResNet-18 minus Simple CNN, clean_acc, standard, seed 2 | a - b | main_results.csv |
| `rq2.standard.clean_acc.resnet18_minus_cnn.seed_noise` | Seed-noise label: ResNet-18 minus Simple CNN, clean_acc, standard | \|mean difference\| < larger of the two seed stds | main_results.csv |
| `rq2.standard.mce.resnet18_minus_cnn` | ResNet-18 minus Simple CNN, mCE, standard | mean over shared seeds of (a - b) | main_results.csv |
| `rq2.standard.mce.resnet18_minus_cnn.seed0` | ResNet-18 minus Simple CNN, mCE, standard, seed 0 | a - b | main_results.csv |
| `rq2.standard.mce.resnet18_minus_cnn.seed1` | ResNet-18 minus Simple CNN, mCE, standard, seed 1 | a - b | main_results.csv |
| `rq2.standard.mce.resnet18_minus_cnn.seed2` | ResNet-18 minus Simple CNN, mCE, standard, seed 2 | a - b | main_results.csv |
| `rq2.standard.mce.resnet18_minus_cnn.seed_noise` | Seed-noise label: ResNet-18 minus Simple CNN, mCE, standard | \|mean difference\| < larger of the two seed stds | main_results.csv |
| `rq2.standard.mean_abs_degradation.resnet18_minus_cnn` | ResNet-18 minus Simple CNN, mean_abs_degradation, standard | mean over shared seeds of (a - b) | main_results.csv |
| `rq2.standard.mean_abs_degradation.resnet18_minus_cnn.seed0` | ResNet-18 minus Simple CNN, mean_abs_degradation, standard, seed 0 | a - b | main_results.csv |
| `rq2.standard.mean_abs_degradation.resnet18_minus_cnn.seed1` | ResNet-18 minus Simple CNN, mean_abs_degradation, standard, seed 1 | a - b | main_results.csv |
| `rq2.standard.mean_abs_degradation.resnet18_minus_cnn.seed2` | ResNet-18 minus Simple CNN, mean_abs_degradation, standard, seed 2 | a - b | main_results.csv |
| `rq2.standard.mean_abs_degradation.resnet18_minus_cnn.seed_noise` | Seed-noise label: ResNet-18 minus Simple CNN, mean_abs_degradation, standard | \|mean difference\| < larger of the two seed stds | main_results.csv |
| `rq2.standard.mean_corrupted_acc.resnet18_minus_cnn` | ResNet-18 minus Simple CNN, mean_corrupted_acc, standard | mean over shared seeds of (a - b) | main_results.csv |
| `rq2.standard.mean_corrupted_acc.resnet18_minus_cnn.seed0` | ResNet-18 minus Simple CNN, mean_corrupted_acc, standard, seed 0 | a - b | main_results.csv |
| `rq2.standard.mean_corrupted_acc.resnet18_minus_cnn.seed1` | ResNet-18 minus Simple CNN, mean_corrupted_acc, standard, seed 1 | a - b | main_results.csv |
| `rq2.standard.mean_corrupted_acc.resnet18_minus_cnn.seed2` | ResNet-18 minus Simple CNN, mean_corrupted_acc, standard, seed 2 | a - b | main_results.csv |
| `rq2.standard.mean_corrupted_acc.resnet18_minus_cnn.seed_noise` | Seed-noise label: ResNet-18 minus Simple CNN, mean_corrupted_acc, standard | \|mean difference\| < larger of the two seed stds | main_results.csv |
| `rq2.standard.mean_rel_degradation.resnet18_minus_cnn` | ResNet-18 minus Simple CNN, mean_rel_degradation, standard | mean over shared seeds of (a - b) | main_results.csv |
| `rq2.standard.mean_rel_degradation.resnet18_minus_cnn.seed0` | ResNet-18 minus Simple CNN, mean_rel_degradation, standard, seed 0 | a - b | main_results.csv |
| `rq2.standard.mean_rel_degradation.resnet18_minus_cnn.seed1` | ResNet-18 minus Simple CNN, mean_rel_degradation, standard, seed 1 | a - b | main_results.csv |
| `rq2.standard.mean_rel_degradation.resnet18_minus_cnn.seed2` | ResNet-18 minus Simple CNN, mean_rel_degradation, standard, seed 2 | a - b | main_results.csv |
| `rq2.standard.mean_rel_degradation.resnet18_minus_cnn.seed_noise` | Seed-noise label: ResNet-18 minus Simple CNN, mean_rel_degradation, standard | \|mean difference\| < larger of the two seed stds | main_results.csv |
| `rq2.standard.relative_mce.resnet18_minus_cnn` | ResNet-18 minus Simple CNN, relative_mCE, standard | mean over shared seeds of (a - b) | main_results.csv |
| `rq2.standard.relative_mce.resnet18_minus_cnn.seed0` | ResNet-18 minus Simple CNN, relative_mCE, standard, seed 0 | a - b | main_results.csv |
| `rq2.standard.relative_mce.resnet18_minus_cnn.seed1` | ResNet-18 minus Simple CNN, relative_mCE, standard, seed 1 | a - b | main_results.csv |
| `rq2.standard.relative_mce.resnet18_minus_cnn.seed2` | ResNet-18 minus Simple CNN, relative_mCE, standard, seed 2 | a - b | main_results.csv |
| `rq2.standard.relative_mce.resnet18_minus_cnn.seed_noise` | Seed-noise label: ResNet-18 minus Simple CNN, relative_mCE, standard | \|mean difference\| < larger of the two seed stds | main_results.csv |
| `rq3.cnn.augmix.blur_acc_gain_vs_standard` | augmix minus standard, accuracy on blur, simple_cnn | mean over shared seeds of (a - b) | per_family.csv |
| `rq3.cnn.augmix.blur_acc_gain_vs_standard.seed0` | augmix minus standard, accuracy on blur, simple_cnn, seed 0 | a - b | per_family.csv |
| `rq3.cnn.augmix.blur_acc_gain_vs_standard.seed1` | augmix minus standard, accuracy on blur, simple_cnn, seed 1 | a - b | per_family.csv |
| `rq3.cnn.augmix.blur_acc_gain_vs_standard.seed2` | augmix minus standard, accuracy on blur, simple_cnn, seed 2 | a - b | per_family.csv |
| `rq3.cnn.augmix.blur_acc_gain_vs_standard.seed_noise` | Seed-noise label: augmix minus standard, accuracy on blur, simple_cnn | \|mean difference\| < larger of the two seed stds | per_family.csv |
| `rq3.cnn.augmix.clean_acc.diff_vs_standard` | augmix minus standard, clean_acc, simple_cnn | mean over shared seeds of (a - b) | main_results.csv |
| `rq3.cnn.augmix.clean_acc.diff_vs_standard.seed0` | augmix minus standard, clean_acc, simple_cnn, seed 0 | a - b | main_results.csv |
| `rq3.cnn.augmix.clean_acc.diff_vs_standard.seed1` | augmix minus standard, clean_acc, simple_cnn, seed 1 | a - b | main_results.csv |
| `rq3.cnn.augmix.clean_acc.diff_vs_standard.seed2` | augmix minus standard, clean_acc, simple_cnn, seed 2 | a - b | main_results.csv |
| `rq3.cnn.augmix.clean_acc.diff_vs_standard.seed_noise` | Seed-noise label: augmix minus standard, clean_acc, simple_cnn | \|mean difference\| < larger of the two seed stds | main_results.csv |
| `rq3.cnn.augmix.digital_acc_gain_vs_standard` | augmix minus standard, accuracy on digital, simple_cnn | mean over shared seeds of (a - b) | per_family.csv |
| `rq3.cnn.augmix.digital_acc_gain_vs_standard.seed0` | augmix minus standard, accuracy on digital, simple_cnn, seed 0 | a - b | per_family.csv |
| `rq3.cnn.augmix.digital_acc_gain_vs_standard.seed1` | augmix minus standard, accuracy on digital, simple_cnn, seed 1 | a - b | per_family.csv |
| `rq3.cnn.augmix.digital_acc_gain_vs_standard.seed2` | augmix minus standard, accuracy on digital, simple_cnn, seed 2 | a - b | per_family.csv |
| `rq3.cnn.augmix.digital_acc_gain_vs_standard.seed_noise` | Seed-noise label: augmix minus standard, accuracy on digital, simple_cnn | \|mean difference\| < larger of the two seed stds | per_family.csv |
| `rq3.cnn.augmix.mce.diff_vs_standard` | augmix minus standard, mCE, simple_cnn | mean over shared seeds of (a - b) | main_results.csv |
| `rq3.cnn.augmix.mce.diff_vs_standard.seed0` | augmix minus standard, mCE, simple_cnn, seed 0 | a - b | main_results.csv |
| `rq3.cnn.augmix.mce.diff_vs_standard.seed1` | augmix minus standard, mCE, simple_cnn, seed 1 | a - b | main_results.csv |
| `rq3.cnn.augmix.mce.diff_vs_standard.seed2` | augmix minus standard, mCE, simple_cnn, seed 2 | a - b | main_results.csv |
| `rq3.cnn.augmix.mce.diff_vs_standard.seed_noise` | Seed-noise label: augmix minus standard, mCE, simple_cnn | \|mean difference\| < larger of the two seed stds | main_results.csv |
| `rq3.cnn.augmix.mean_abs_degradation.diff_vs_standard` | augmix minus standard, mean_abs_degradation, simple_cnn | mean over shared seeds of (a - b) | main_results.csv |
| `rq3.cnn.augmix.mean_abs_degradation.diff_vs_standard.seed0` | augmix minus standard, mean_abs_degradation, simple_cnn, seed 0 | a - b | main_results.csv |
| `rq3.cnn.augmix.mean_abs_degradation.diff_vs_standard.seed1` | augmix minus standard, mean_abs_degradation, simple_cnn, seed 1 | a - b | main_results.csv |
| `rq3.cnn.augmix.mean_abs_degradation.diff_vs_standard.seed2` | augmix minus standard, mean_abs_degradation, simple_cnn, seed 2 | a - b | main_results.csv |
| `rq3.cnn.augmix.mean_abs_degradation.diff_vs_standard.seed_noise` | Seed-noise label: augmix minus standard, mean_abs_degradation, simple_cnn | \|mean difference\| < larger of the two seed stds | main_results.csv |
| `rq3.cnn.augmix.mean_corrupted_acc.diff_vs_standard` | augmix minus standard, mean_corrupted_acc, simple_cnn | mean over shared seeds of (a - b) | main_results.csv |
| `rq3.cnn.augmix.mean_corrupted_acc.diff_vs_standard.seed0` | augmix minus standard, mean_corrupted_acc, simple_cnn, seed 0 | a - b | main_results.csv |
| `rq3.cnn.augmix.mean_corrupted_acc.diff_vs_standard.seed1` | augmix minus standard, mean_corrupted_acc, simple_cnn, seed 1 | a - b | main_results.csv |
| `rq3.cnn.augmix.mean_corrupted_acc.diff_vs_standard.seed2` | augmix minus standard, mean_corrupted_acc, simple_cnn, seed 2 | a - b | main_results.csv |
| `rq3.cnn.augmix.mean_corrupted_acc.diff_vs_standard.seed_noise` | Seed-noise label: augmix minus standard, mean_corrupted_acc, simple_cnn | \|mean difference\| < larger of the two seed stds | main_results.csv |
| `rq3.cnn.augmix.mean_rel_degradation.diff_vs_standard` | augmix minus standard, mean_rel_degradation, simple_cnn | mean over shared seeds of (a - b) | main_results.csv |
| `rq3.cnn.augmix.mean_rel_degradation.diff_vs_standard.seed0` | augmix minus standard, mean_rel_degradation, simple_cnn, seed 0 | a - b | main_results.csv |
| `rq3.cnn.augmix.mean_rel_degradation.diff_vs_standard.seed1` | augmix minus standard, mean_rel_degradation, simple_cnn, seed 1 | a - b | main_results.csv |
| `rq3.cnn.augmix.mean_rel_degradation.diff_vs_standard.seed2` | augmix minus standard, mean_rel_degradation, simple_cnn, seed 2 | a - b | main_results.csv |
| `rq3.cnn.augmix.mean_rel_degradation.diff_vs_standard.seed_noise` | Seed-noise label: augmix minus standard, mean_rel_degradation, simple_cnn | \|mean difference\| < larger of the two seed stds | main_results.csv |
| `rq3.cnn.augmix.noise_acc_gain_vs_standard` | augmix minus standard, accuracy on noise, simple_cnn | mean over shared seeds of (a - b) | per_family.csv |
| `rq3.cnn.augmix.noise_acc_gain_vs_standard.seed0` | augmix minus standard, accuracy on noise, simple_cnn, seed 0 | a - b | per_family.csv |
| `rq3.cnn.augmix.noise_acc_gain_vs_standard.seed1` | augmix minus standard, accuracy on noise, simple_cnn, seed 1 | a - b | per_family.csv |
| `rq3.cnn.augmix.noise_acc_gain_vs_standard.seed2` | augmix minus standard, accuracy on noise, simple_cnn, seed 2 | a - b | per_family.csv |
| `rq3.cnn.augmix.noise_acc_gain_vs_standard.seed_noise` | Seed-noise label: augmix minus standard, accuracy on noise, simple_cnn | \|mean difference\| < larger of the two seed stds | per_family.csv |
| `rq3.cnn.augmix.relative_mce.diff_vs_standard` | augmix minus standard, relative_mCE, simple_cnn | mean over shared seeds of (a - b) | main_results.csv |
| `rq3.cnn.augmix.relative_mce.diff_vs_standard.seed0` | augmix minus standard, relative_mCE, simple_cnn, seed 0 | a - b | main_results.csv |
| `rq3.cnn.augmix.relative_mce.diff_vs_standard.seed1` | augmix minus standard, relative_mCE, simple_cnn, seed 1 | a - b | main_results.csv |
| `rq3.cnn.augmix.relative_mce.diff_vs_standard.seed2` | augmix minus standard, relative_mCE, simple_cnn, seed 2 | a - b | main_results.csv |
| `rq3.cnn.augmix.relative_mce.diff_vs_standard.seed_noise` | Seed-noise label: augmix minus standard, relative_mCE, simple_cnn | \|mean difference\| < larger of the two seed stds | main_results.csv |
| `rq3.cnn.augmix.weather_acc_gain_vs_standard` | augmix minus standard, accuracy on weather, simple_cnn | mean over shared seeds of (a - b) | per_family.csv |
| `rq3.cnn.augmix.weather_acc_gain_vs_standard.seed0` | augmix minus standard, accuracy on weather, simple_cnn, seed 0 | a - b | per_family.csv |
| `rq3.cnn.augmix.weather_acc_gain_vs_standard.seed1` | augmix minus standard, accuracy on weather, simple_cnn, seed 1 | a - b | per_family.csv |
| `rq3.cnn.augmix.weather_acc_gain_vs_standard.seed2` | augmix minus standard, accuracy on weather, simple_cnn, seed 2 | a - b | per_family.csv |
| `rq3.cnn.augmix.weather_acc_gain_vs_standard.seed_noise` | Seed-noise label: augmix minus standard, accuracy on weather, simple_cnn | \|mean difference\| < larger of the two seed stds | per_family.csv |
| `rq3.cnn.corruption_aware.blur_acc_gain_vs_standard` | corruption_aware minus standard, accuracy on blur, simple_cnn | mean over shared seeds of (a - b) | per_family.csv |
| `rq3.cnn.corruption_aware.blur_acc_gain_vs_standard.seed0` | corruption_aware minus standard, accuracy on blur, simple_cnn, seed 0 | a - b | per_family.csv |
| `rq3.cnn.corruption_aware.blur_acc_gain_vs_standard.seed1` | corruption_aware minus standard, accuracy on blur, simple_cnn, seed 1 | a - b | per_family.csv |
| `rq3.cnn.corruption_aware.blur_acc_gain_vs_standard.seed2` | corruption_aware minus standard, accuracy on blur, simple_cnn, seed 2 | a - b | per_family.csv |
| `rq3.cnn.corruption_aware.blur_acc_gain_vs_standard.seed_noise` | Seed-noise label: corruption_aware minus standard, accuracy on blur, simple_cnn | \|mean difference\| < larger of the two seed stds | per_family.csv |
| `rq3.cnn.corruption_aware.clean_acc.diff_vs_standard` | corruption_aware minus standard, clean_acc, simple_cnn | mean over shared seeds of (a - b) | main_results.csv |
| `rq3.cnn.corruption_aware.clean_acc.diff_vs_standard.seed0` | corruption_aware minus standard, clean_acc, simple_cnn, seed 0 | a - b | main_results.csv |
| `rq3.cnn.corruption_aware.clean_acc.diff_vs_standard.seed1` | corruption_aware minus standard, clean_acc, simple_cnn, seed 1 | a - b | main_results.csv |
| `rq3.cnn.corruption_aware.clean_acc.diff_vs_standard.seed2` | corruption_aware minus standard, clean_acc, simple_cnn, seed 2 | a - b | main_results.csv |
| `rq3.cnn.corruption_aware.clean_acc.diff_vs_standard.seed_noise` | Seed-noise label: corruption_aware minus standard, clean_acc, simple_cnn | \|mean difference\| < larger of the two seed stds | main_results.csv |
| `rq3.cnn.corruption_aware.digital_acc_gain_vs_standard` | corruption_aware minus standard, accuracy on digital, simple_cnn | mean over shared seeds of (a - b) | per_family.csv |
| `rq3.cnn.corruption_aware.digital_acc_gain_vs_standard.seed0` | corruption_aware minus standard, accuracy on digital, simple_cnn, seed 0 | a - b | per_family.csv |
| `rq3.cnn.corruption_aware.digital_acc_gain_vs_standard.seed1` | corruption_aware minus standard, accuracy on digital, simple_cnn, seed 1 | a - b | per_family.csv |
| `rq3.cnn.corruption_aware.digital_acc_gain_vs_standard.seed2` | corruption_aware minus standard, accuracy on digital, simple_cnn, seed 2 | a - b | per_family.csv |
| `rq3.cnn.corruption_aware.digital_acc_gain_vs_standard.seed_noise` | Seed-noise label: corruption_aware minus standard, accuracy on digital, simple_cnn | \|mean difference\| < larger of the two seed stds | per_family.csv |
| `rq3.cnn.corruption_aware.mce.diff_vs_standard` | corruption_aware minus standard, mCE, simple_cnn | mean over shared seeds of (a - b) | main_results.csv |
| `rq3.cnn.corruption_aware.mce.diff_vs_standard.seed0` | corruption_aware minus standard, mCE, simple_cnn, seed 0 | a - b | main_results.csv |
| `rq3.cnn.corruption_aware.mce.diff_vs_standard.seed1` | corruption_aware minus standard, mCE, simple_cnn, seed 1 | a - b | main_results.csv |
| `rq3.cnn.corruption_aware.mce.diff_vs_standard.seed2` | corruption_aware minus standard, mCE, simple_cnn, seed 2 | a - b | main_results.csv |
| `rq3.cnn.corruption_aware.mce.diff_vs_standard.seed_noise` | Seed-noise label: corruption_aware minus standard, mCE, simple_cnn | \|mean difference\| < larger of the two seed stds | main_results.csv |
| `rq3.cnn.corruption_aware.mean_abs_degradation.diff_vs_standard` | corruption_aware minus standard, mean_abs_degradation, simple_cnn | mean over shared seeds of (a - b) | main_results.csv |
| `rq3.cnn.corruption_aware.mean_abs_degradation.diff_vs_standard.seed0` | corruption_aware minus standard, mean_abs_degradation, simple_cnn, seed 0 | a - b | main_results.csv |
| `rq3.cnn.corruption_aware.mean_abs_degradation.diff_vs_standard.seed1` | corruption_aware minus standard, mean_abs_degradation, simple_cnn, seed 1 | a - b | main_results.csv |
| `rq3.cnn.corruption_aware.mean_abs_degradation.diff_vs_standard.seed2` | corruption_aware minus standard, mean_abs_degradation, simple_cnn, seed 2 | a - b | main_results.csv |
| `rq3.cnn.corruption_aware.mean_abs_degradation.diff_vs_standard.seed_noise` | Seed-noise label: corruption_aware minus standard, mean_abs_degradation, simple_cnn | \|mean difference\| < larger of the two seed stds | main_results.csv |
| `rq3.cnn.corruption_aware.mean_corrupted_acc.diff_vs_standard` | corruption_aware minus standard, mean_corrupted_acc, simple_cnn | mean over shared seeds of (a - b) | main_results.csv |
| `rq3.cnn.corruption_aware.mean_corrupted_acc.diff_vs_standard.seed0` | corruption_aware minus standard, mean_corrupted_acc, simple_cnn, seed 0 | a - b | main_results.csv |
| `rq3.cnn.corruption_aware.mean_corrupted_acc.diff_vs_standard.seed1` | corruption_aware minus standard, mean_corrupted_acc, simple_cnn, seed 1 | a - b | main_results.csv |
| `rq3.cnn.corruption_aware.mean_corrupted_acc.diff_vs_standard.seed2` | corruption_aware minus standard, mean_corrupted_acc, simple_cnn, seed 2 | a - b | main_results.csv |
| `rq3.cnn.corruption_aware.mean_corrupted_acc.diff_vs_standard.seed_noise` | Seed-noise label: corruption_aware minus standard, mean_corrupted_acc, simple_cnn | \|mean difference\| < larger of the two seed stds | main_results.csv |
| `rq3.cnn.corruption_aware.mean_rel_degradation.diff_vs_standard` | corruption_aware minus standard, mean_rel_degradation, simple_cnn | mean over shared seeds of (a - b) | main_results.csv |
| `rq3.cnn.corruption_aware.mean_rel_degradation.diff_vs_standard.seed0` | corruption_aware minus standard, mean_rel_degradation, simple_cnn, seed 0 | a - b | main_results.csv |
| `rq3.cnn.corruption_aware.mean_rel_degradation.diff_vs_standard.seed1` | corruption_aware minus standard, mean_rel_degradation, simple_cnn, seed 1 | a - b | main_results.csv |
| `rq3.cnn.corruption_aware.mean_rel_degradation.diff_vs_standard.seed2` | corruption_aware minus standard, mean_rel_degradation, simple_cnn, seed 2 | a - b | main_results.csv |
| `rq3.cnn.corruption_aware.mean_rel_degradation.diff_vs_standard.seed_noise` | Seed-noise label: corruption_aware minus standard, mean_rel_degradation, simple_cnn | \|mean difference\| < larger of the two seed stds | main_results.csv |
| `rq3.cnn.corruption_aware.noise_acc_gain_vs_standard` | corruption_aware minus standard, accuracy on noise, simple_cnn | mean over shared seeds of (a - b) | per_family.csv |
| `rq3.cnn.corruption_aware.noise_acc_gain_vs_standard.seed0` | corruption_aware minus standard, accuracy on noise, simple_cnn, seed 0 | a - b | per_family.csv |
| `rq3.cnn.corruption_aware.noise_acc_gain_vs_standard.seed1` | corruption_aware minus standard, accuracy on noise, simple_cnn, seed 1 | a - b | per_family.csv |
| `rq3.cnn.corruption_aware.noise_acc_gain_vs_standard.seed2` | corruption_aware minus standard, accuracy on noise, simple_cnn, seed 2 | a - b | per_family.csv |
| `rq3.cnn.corruption_aware.noise_acc_gain_vs_standard.seed_noise` | Seed-noise label: corruption_aware minus standard, accuracy on noise, simple_cnn | \|mean difference\| < larger of the two seed stds | per_family.csv |
| `rq3.cnn.corruption_aware.relative_mce.diff_vs_standard` | corruption_aware minus standard, relative_mCE, simple_cnn | mean over shared seeds of (a - b) | main_results.csv |
| `rq3.cnn.corruption_aware.relative_mce.diff_vs_standard.seed0` | corruption_aware minus standard, relative_mCE, simple_cnn, seed 0 | a - b | main_results.csv |
| `rq3.cnn.corruption_aware.relative_mce.diff_vs_standard.seed1` | corruption_aware minus standard, relative_mCE, simple_cnn, seed 1 | a - b | main_results.csv |
| `rq3.cnn.corruption_aware.relative_mce.diff_vs_standard.seed2` | corruption_aware minus standard, relative_mCE, simple_cnn, seed 2 | a - b | main_results.csv |
| `rq3.cnn.corruption_aware.relative_mce.diff_vs_standard.seed_noise` | Seed-noise label: corruption_aware minus standard, relative_mCE, simple_cnn | \|mean difference\| < larger of the two seed stds | main_results.csv |
| `rq3.cnn.corruption_aware.weather_acc_gain_vs_standard` | corruption_aware minus standard, accuracy on weather, simple_cnn | mean over shared seeds of (a - b) | per_family.csv |
| `rq3.cnn.corruption_aware.weather_acc_gain_vs_standard.seed0` | corruption_aware minus standard, accuracy on weather, simple_cnn, seed 0 | a - b | per_family.csv |
| `rq3.cnn.corruption_aware.weather_acc_gain_vs_standard.seed1` | corruption_aware minus standard, accuracy on weather, simple_cnn, seed 1 | a - b | per_family.csv |
| `rq3.cnn.corruption_aware.weather_acc_gain_vs_standard.seed2` | corruption_aware minus standard, accuracy on weather, simple_cnn, seed 2 | a - b | per_family.csv |
| `rq3.cnn.corruption_aware.weather_acc_gain_vs_standard.seed_noise` | Seed-noise label: corruption_aware minus standard, accuracy on weather, simple_cnn | \|mean difference\| < larger of the two seed stds | per_family.csv |
| `rq3.resnet18.augmix.blur_acc_gain_vs_standard` | augmix minus standard, accuracy on blur, resnet18 | mean over shared seeds of (a - b) | per_family.csv |
| `rq3.resnet18.augmix.blur_acc_gain_vs_standard.seed0` | augmix minus standard, accuracy on blur, resnet18, seed 0 | a - b | per_family.csv |
| `rq3.resnet18.augmix.blur_acc_gain_vs_standard.seed1` | augmix minus standard, accuracy on blur, resnet18, seed 1 | a - b | per_family.csv |
| `rq3.resnet18.augmix.blur_acc_gain_vs_standard.seed2` | augmix minus standard, accuracy on blur, resnet18, seed 2 | a - b | per_family.csv |
| `rq3.resnet18.augmix.blur_acc_gain_vs_standard.seed_noise` | Seed-noise label: augmix minus standard, accuracy on blur, resnet18 | \|mean difference\| < larger of the two seed stds | per_family.csv |
| `rq3.resnet18.augmix.clean_acc.diff_vs_standard` | augmix minus standard, clean_acc, resnet18 | mean over shared seeds of (a - b) | main_results.csv |
| `rq3.resnet18.augmix.clean_acc.diff_vs_standard.seed0` | augmix minus standard, clean_acc, resnet18, seed 0 | a - b | main_results.csv |
| `rq3.resnet18.augmix.clean_acc.diff_vs_standard.seed1` | augmix minus standard, clean_acc, resnet18, seed 1 | a - b | main_results.csv |
| `rq3.resnet18.augmix.clean_acc.diff_vs_standard.seed2` | augmix minus standard, clean_acc, resnet18, seed 2 | a - b | main_results.csv |
| `rq3.resnet18.augmix.clean_acc.diff_vs_standard.seed_noise` | Seed-noise label: augmix minus standard, clean_acc, resnet18 | \|mean difference\| < larger of the two seed stds | main_results.csv |
| `rq3.resnet18.augmix.digital_acc_gain_vs_standard` | augmix minus standard, accuracy on digital, resnet18 | mean over shared seeds of (a - b) | per_family.csv |
| `rq3.resnet18.augmix.digital_acc_gain_vs_standard.seed0` | augmix minus standard, accuracy on digital, resnet18, seed 0 | a - b | per_family.csv |
| `rq3.resnet18.augmix.digital_acc_gain_vs_standard.seed1` | augmix minus standard, accuracy on digital, resnet18, seed 1 | a - b | per_family.csv |
| `rq3.resnet18.augmix.digital_acc_gain_vs_standard.seed2` | augmix minus standard, accuracy on digital, resnet18, seed 2 | a - b | per_family.csv |
| `rq3.resnet18.augmix.digital_acc_gain_vs_standard.seed_noise` | Seed-noise label: augmix minus standard, accuracy on digital, resnet18 | \|mean difference\| < larger of the two seed stds | per_family.csv |
| `rq3.resnet18.augmix.mce.diff_vs_standard` | augmix minus standard, mCE, resnet18 | mean over shared seeds of (a - b) | main_results.csv |
| `rq3.resnet18.augmix.mce.diff_vs_standard.seed0` | augmix minus standard, mCE, resnet18, seed 0 | a - b | main_results.csv |
| `rq3.resnet18.augmix.mce.diff_vs_standard.seed1` | augmix minus standard, mCE, resnet18, seed 1 | a - b | main_results.csv |
| `rq3.resnet18.augmix.mce.diff_vs_standard.seed2` | augmix minus standard, mCE, resnet18, seed 2 | a - b | main_results.csv |
| `rq3.resnet18.augmix.mce.diff_vs_standard.seed_noise` | Seed-noise label: augmix minus standard, mCE, resnet18 | \|mean difference\| < larger of the two seed stds | main_results.csv |
| `rq3.resnet18.augmix.mean_abs_degradation.diff_vs_standard` | augmix minus standard, mean_abs_degradation, resnet18 | mean over shared seeds of (a - b) | main_results.csv |
| `rq3.resnet18.augmix.mean_abs_degradation.diff_vs_standard.seed0` | augmix minus standard, mean_abs_degradation, resnet18, seed 0 | a - b | main_results.csv |
| `rq3.resnet18.augmix.mean_abs_degradation.diff_vs_standard.seed1` | augmix minus standard, mean_abs_degradation, resnet18, seed 1 | a - b | main_results.csv |
| `rq3.resnet18.augmix.mean_abs_degradation.diff_vs_standard.seed2` | augmix minus standard, mean_abs_degradation, resnet18, seed 2 | a - b | main_results.csv |
| `rq3.resnet18.augmix.mean_abs_degradation.diff_vs_standard.seed_noise` | Seed-noise label: augmix minus standard, mean_abs_degradation, resnet18 | \|mean difference\| < larger of the two seed stds | main_results.csv |
| `rq3.resnet18.augmix.mean_corrupted_acc.diff_vs_standard` | augmix minus standard, mean_corrupted_acc, resnet18 | mean over shared seeds of (a - b) | main_results.csv |
| `rq3.resnet18.augmix.mean_corrupted_acc.diff_vs_standard.seed0` | augmix minus standard, mean_corrupted_acc, resnet18, seed 0 | a - b | main_results.csv |
| `rq3.resnet18.augmix.mean_corrupted_acc.diff_vs_standard.seed1` | augmix minus standard, mean_corrupted_acc, resnet18, seed 1 | a - b | main_results.csv |
| `rq3.resnet18.augmix.mean_corrupted_acc.diff_vs_standard.seed2` | augmix minus standard, mean_corrupted_acc, resnet18, seed 2 | a - b | main_results.csv |
| `rq3.resnet18.augmix.mean_corrupted_acc.diff_vs_standard.seed_noise` | Seed-noise label: augmix minus standard, mean_corrupted_acc, resnet18 | \|mean difference\| < larger of the two seed stds | main_results.csv |
| `rq3.resnet18.augmix.mean_rel_degradation.diff_vs_standard` | augmix minus standard, mean_rel_degradation, resnet18 | mean over shared seeds of (a - b) | main_results.csv |
| `rq3.resnet18.augmix.mean_rel_degradation.diff_vs_standard.seed0` | augmix minus standard, mean_rel_degradation, resnet18, seed 0 | a - b | main_results.csv |
| `rq3.resnet18.augmix.mean_rel_degradation.diff_vs_standard.seed1` | augmix minus standard, mean_rel_degradation, resnet18, seed 1 | a - b | main_results.csv |
| `rq3.resnet18.augmix.mean_rel_degradation.diff_vs_standard.seed2` | augmix minus standard, mean_rel_degradation, resnet18, seed 2 | a - b | main_results.csv |
| `rq3.resnet18.augmix.mean_rel_degradation.diff_vs_standard.seed_noise` | Seed-noise label: augmix minus standard, mean_rel_degradation, resnet18 | \|mean difference\| < larger of the two seed stds | main_results.csv |
| `rq3.resnet18.augmix.noise_acc_gain_vs_standard` | augmix minus standard, accuracy on noise, resnet18 | mean over shared seeds of (a - b) | per_family.csv |
| `rq3.resnet18.augmix.noise_acc_gain_vs_standard.seed0` | augmix minus standard, accuracy on noise, resnet18, seed 0 | a - b | per_family.csv |
| `rq3.resnet18.augmix.noise_acc_gain_vs_standard.seed1` | augmix minus standard, accuracy on noise, resnet18, seed 1 | a - b | per_family.csv |
| `rq3.resnet18.augmix.noise_acc_gain_vs_standard.seed2` | augmix minus standard, accuracy on noise, resnet18, seed 2 | a - b | per_family.csv |
| `rq3.resnet18.augmix.noise_acc_gain_vs_standard.seed_noise` | Seed-noise label: augmix minus standard, accuracy on noise, resnet18 | \|mean difference\| < larger of the two seed stds | per_family.csv |
| `rq3.resnet18.augmix.relative_mce.diff_vs_standard` | augmix minus standard, relative_mCE, resnet18 | mean over shared seeds of (a - b) | main_results.csv |
| `rq3.resnet18.augmix.relative_mce.diff_vs_standard.seed0` | augmix minus standard, relative_mCE, resnet18, seed 0 | a - b | main_results.csv |
| `rq3.resnet18.augmix.relative_mce.diff_vs_standard.seed1` | augmix minus standard, relative_mCE, resnet18, seed 1 | a - b | main_results.csv |
| `rq3.resnet18.augmix.relative_mce.diff_vs_standard.seed2` | augmix minus standard, relative_mCE, resnet18, seed 2 | a - b | main_results.csv |
| `rq3.resnet18.augmix.relative_mce.diff_vs_standard.seed_noise` | Seed-noise label: augmix minus standard, relative_mCE, resnet18 | \|mean difference\| < larger of the two seed stds | main_results.csv |
| `rq3.resnet18.augmix.weather_acc_gain_vs_standard` | augmix minus standard, accuracy on weather, resnet18 | mean over shared seeds of (a - b) | per_family.csv |
| `rq3.resnet18.augmix.weather_acc_gain_vs_standard.seed0` | augmix minus standard, accuracy on weather, resnet18, seed 0 | a - b | per_family.csv |
| `rq3.resnet18.augmix.weather_acc_gain_vs_standard.seed1` | augmix minus standard, accuracy on weather, resnet18, seed 1 | a - b | per_family.csv |
| `rq3.resnet18.augmix.weather_acc_gain_vs_standard.seed2` | augmix minus standard, accuracy on weather, resnet18, seed 2 | a - b | per_family.csv |
| `rq3.resnet18.augmix.weather_acc_gain_vs_standard.seed_noise` | Seed-noise label: augmix minus standard, accuracy on weather, resnet18 | \|mean difference\| < larger of the two seed stds | per_family.csv |
| `rq3.resnet18.corruption_aware.blur_acc_gain_vs_standard` | corruption_aware minus standard, accuracy on blur, resnet18 | mean over shared seeds of (a - b) | per_family.csv |
| `rq3.resnet18.corruption_aware.blur_acc_gain_vs_standard.seed0` | corruption_aware minus standard, accuracy on blur, resnet18, seed 0 | a - b | per_family.csv |
| `rq3.resnet18.corruption_aware.blur_acc_gain_vs_standard.seed1` | corruption_aware minus standard, accuracy on blur, resnet18, seed 1 | a - b | per_family.csv |
| `rq3.resnet18.corruption_aware.blur_acc_gain_vs_standard.seed2` | corruption_aware minus standard, accuracy on blur, resnet18, seed 2 | a - b | per_family.csv |
| `rq3.resnet18.corruption_aware.blur_acc_gain_vs_standard.seed_noise` | Seed-noise label: corruption_aware minus standard, accuracy on blur, resnet18 | \|mean difference\| < larger of the two seed stds | per_family.csv |
| `rq3.resnet18.corruption_aware.clean_acc.diff_vs_standard` | corruption_aware minus standard, clean_acc, resnet18 | mean over shared seeds of (a - b) | main_results.csv |
| `rq3.resnet18.corruption_aware.clean_acc.diff_vs_standard.seed0` | corruption_aware minus standard, clean_acc, resnet18, seed 0 | a - b | main_results.csv |
| `rq3.resnet18.corruption_aware.clean_acc.diff_vs_standard.seed1` | corruption_aware minus standard, clean_acc, resnet18, seed 1 | a - b | main_results.csv |
| `rq3.resnet18.corruption_aware.clean_acc.diff_vs_standard.seed2` | corruption_aware minus standard, clean_acc, resnet18, seed 2 | a - b | main_results.csv |
| `rq3.resnet18.corruption_aware.clean_acc.diff_vs_standard.seed_noise` | Seed-noise label: corruption_aware minus standard, clean_acc, resnet18 | \|mean difference\| < larger of the two seed stds | main_results.csv |
| `rq3.resnet18.corruption_aware.digital_acc_gain_vs_standard` | corruption_aware minus standard, accuracy on digital, resnet18 | mean over shared seeds of (a - b) | per_family.csv |
| `rq3.resnet18.corruption_aware.digital_acc_gain_vs_standard.seed0` | corruption_aware minus standard, accuracy on digital, resnet18, seed 0 | a - b | per_family.csv |
| `rq3.resnet18.corruption_aware.digital_acc_gain_vs_standard.seed1` | corruption_aware minus standard, accuracy on digital, resnet18, seed 1 | a - b | per_family.csv |
| `rq3.resnet18.corruption_aware.digital_acc_gain_vs_standard.seed2` | corruption_aware minus standard, accuracy on digital, resnet18, seed 2 | a - b | per_family.csv |
| `rq3.resnet18.corruption_aware.digital_acc_gain_vs_standard.seed_noise` | Seed-noise label: corruption_aware minus standard, accuracy on digital, resnet18 | \|mean difference\| < larger of the two seed stds | per_family.csv |
| `rq3.resnet18.corruption_aware.mce.diff_vs_standard` | corruption_aware minus standard, mCE, resnet18 | mean over shared seeds of (a - b) | main_results.csv |
| `rq3.resnet18.corruption_aware.mce.diff_vs_standard.seed0` | corruption_aware minus standard, mCE, resnet18, seed 0 | a - b | main_results.csv |
| `rq3.resnet18.corruption_aware.mce.diff_vs_standard.seed1` | corruption_aware minus standard, mCE, resnet18, seed 1 | a - b | main_results.csv |
| `rq3.resnet18.corruption_aware.mce.diff_vs_standard.seed2` | corruption_aware minus standard, mCE, resnet18, seed 2 | a - b | main_results.csv |
| `rq3.resnet18.corruption_aware.mce.diff_vs_standard.seed_noise` | Seed-noise label: corruption_aware minus standard, mCE, resnet18 | \|mean difference\| < larger of the two seed stds | main_results.csv |
| `rq3.resnet18.corruption_aware.mean_abs_degradation.diff_vs_standard` | corruption_aware minus standard, mean_abs_degradation, resnet18 | mean over shared seeds of (a - b) | main_results.csv |
| `rq3.resnet18.corruption_aware.mean_abs_degradation.diff_vs_standard.seed0` | corruption_aware minus standard, mean_abs_degradation, resnet18, seed 0 | a - b | main_results.csv |
| `rq3.resnet18.corruption_aware.mean_abs_degradation.diff_vs_standard.seed1` | corruption_aware minus standard, mean_abs_degradation, resnet18, seed 1 | a - b | main_results.csv |
| `rq3.resnet18.corruption_aware.mean_abs_degradation.diff_vs_standard.seed2` | corruption_aware minus standard, mean_abs_degradation, resnet18, seed 2 | a - b | main_results.csv |
| `rq3.resnet18.corruption_aware.mean_abs_degradation.diff_vs_standard.seed_noise` | Seed-noise label: corruption_aware minus standard, mean_abs_degradation, resnet18 | \|mean difference\| < larger of the two seed stds | main_results.csv |
| `rq3.resnet18.corruption_aware.mean_corrupted_acc.diff_vs_standard` | corruption_aware minus standard, mean_corrupted_acc, resnet18 | mean over shared seeds of (a - b) | main_results.csv |
| `rq3.resnet18.corruption_aware.mean_corrupted_acc.diff_vs_standard.seed0` | corruption_aware minus standard, mean_corrupted_acc, resnet18, seed 0 | a - b | main_results.csv |
| `rq3.resnet18.corruption_aware.mean_corrupted_acc.diff_vs_standard.seed1` | corruption_aware minus standard, mean_corrupted_acc, resnet18, seed 1 | a - b | main_results.csv |
| `rq3.resnet18.corruption_aware.mean_corrupted_acc.diff_vs_standard.seed2` | corruption_aware minus standard, mean_corrupted_acc, resnet18, seed 2 | a - b | main_results.csv |
| `rq3.resnet18.corruption_aware.mean_corrupted_acc.diff_vs_standard.seed_noise` | Seed-noise label: corruption_aware minus standard, mean_corrupted_acc, resnet18 | \|mean difference\| < larger of the two seed stds | main_results.csv |
| `rq3.resnet18.corruption_aware.mean_rel_degradation.diff_vs_standard` | corruption_aware minus standard, mean_rel_degradation, resnet18 | mean over shared seeds of (a - b) | main_results.csv |
| `rq3.resnet18.corruption_aware.mean_rel_degradation.diff_vs_standard.seed0` | corruption_aware minus standard, mean_rel_degradation, resnet18, seed 0 | a - b | main_results.csv |
| `rq3.resnet18.corruption_aware.mean_rel_degradation.diff_vs_standard.seed1` | corruption_aware minus standard, mean_rel_degradation, resnet18, seed 1 | a - b | main_results.csv |
| `rq3.resnet18.corruption_aware.mean_rel_degradation.diff_vs_standard.seed2` | corruption_aware minus standard, mean_rel_degradation, resnet18, seed 2 | a - b | main_results.csv |
| `rq3.resnet18.corruption_aware.mean_rel_degradation.diff_vs_standard.seed_noise` | Seed-noise label: corruption_aware minus standard, mean_rel_degradation, resnet18 | \|mean difference\| < larger of the two seed stds | main_results.csv |
| `rq3.resnet18.corruption_aware.noise_acc_gain_vs_standard` | corruption_aware minus standard, accuracy on noise, resnet18 | mean over shared seeds of (a - b) | per_family.csv |
| `rq3.resnet18.corruption_aware.noise_acc_gain_vs_standard.seed0` | corruption_aware minus standard, accuracy on noise, resnet18, seed 0 | a - b | per_family.csv |
| `rq3.resnet18.corruption_aware.noise_acc_gain_vs_standard.seed1` | corruption_aware minus standard, accuracy on noise, resnet18, seed 1 | a - b | per_family.csv |
| `rq3.resnet18.corruption_aware.noise_acc_gain_vs_standard.seed2` | corruption_aware minus standard, accuracy on noise, resnet18, seed 2 | a - b | per_family.csv |
| `rq3.resnet18.corruption_aware.noise_acc_gain_vs_standard.seed_noise` | Seed-noise label: corruption_aware minus standard, accuracy on noise, resnet18 | \|mean difference\| < larger of the two seed stds | per_family.csv |
| `rq3.resnet18.corruption_aware.relative_mce.diff_vs_standard` | corruption_aware minus standard, relative_mCE, resnet18 | mean over shared seeds of (a - b) | main_results.csv |
| `rq3.resnet18.corruption_aware.relative_mce.diff_vs_standard.seed0` | corruption_aware minus standard, relative_mCE, resnet18, seed 0 | a - b | main_results.csv |
| `rq3.resnet18.corruption_aware.relative_mce.diff_vs_standard.seed1` | corruption_aware minus standard, relative_mCE, resnet18, seed 1 | a - b | main_results.csv |
| `rq3.resnet18.corruption_aware.relative_mce.diff_vs_standard.seed2` | corruption_aware minus standard, relative_mCE, resnet18, seed 2 | a - b | main_results.csv |
| `rq3.resnet18.corruption_aware.relative_mce.diff_vs_standard.seed_noise` | Seed-noise label: corruption_aware minus standard, relative_mCE, resnet18 | \|mean difference\| < larger of the two seed stds | main_results.csv |
| `rq3.resnet18.corruption_aware.weather_acc_gain_vs_standard` | corruption_aware minus standard, accuracy on weather, resnet18 | mean over shared seeds of (a - b) | per_family.csv |
| `rq3.resnet18.corruption_aware.weather_acc_gain_vs_standard.seed0` | corruption_aware minus standard, accuracy on weather, resnet18, seed 0 | a - b | per_family.csv |
| `rq3.resnet18.corruption_aware.weather_acc_gain_vs_standard.seed1` | corruption_aware minus standard, accuracy on weather, resnet18, seed 1 | a - b | per_family.csv |
| `rq3.resnet18.corruption_aware.weather_acc_gain_vs_standard.seed2` | corruption_aware minus standard, accuracy on weather, resnet18, seed 2 | a - b | per_family.csv |
| `rq3.resnet18.corruption_aware.weather_acc_gain_vs_standard.seed_noise` | Seed-noise label: corruption_aware minus standard, accuracy on weather, resnet18 | \|mean difference\| < larger of the two seed stds | per_family.csv |
| `rq4.cnn.augmix.seen_gain_vs_standard` | augmix minus standard, accuracy on seen, simple_cnn | mean over shared seeds of (a - b) | seen_unseen.csv |
| `rq4.cnn.augmix.seen_gain_vs_standard.seed0` | augmix minus standard, accuracy on seen, simple_cnn, seed 0 | a - b | seen_unseen.csv |
| `rq4.cnn.augmix.seen_gain_vs_standard.seed1` | augmix minus standard, accuracy on seen, simple_cnn, seed 1 | a - b | seen_unseen.csv |
| `rq4.cnn.augmix.seen_gain_vs_standard.seed2` | augmix minus standard, accuracy on seen, simple_cnn, seed 2 | a - b | seen_unseen.csv |
| `rq4.cnn.augmix.seen_gain_vs_standard.seed_noise` | Seed-noise label: augmix minus standard, accuracy on seen, simple_cnn | \|mean difference\| < larger of the two seed stds | seen_unseen.csv |
| `rq4.cnn.augmix.seen_minus_unseen_other_family_gain` | Gain vs standard on seen minus gain on unseen_other_family, augmix, simple_cnn | mean over shared seeds of (a - b) | seen_unseen.csv |
| `rq4.cnn.augmix.seen_minus_unseen_other_family_gain.seed0` | Gain vs standard on seen minus gain on unseen_other_family, augmix, simple_cnn, seed 0 | a - b | seen_unseen.csv |
| `rq4.cnn.augmix.seen_minus_unseen_other_family_gain.seed1` | Gain vs standard on seen minus gain on unseen_other_family, augmix, simple_cnn, seed 1 | a - b | seen_unseen.csv |
| `rq4.cnn.augmix.seen_minus_unseen_other_family_gain.seed2` | Gain vs standard on seen minus gain on unseen_other_family, augmix, simple_cnn, seed 2 | a - b | seen_unseen.csv |
| `rq4.cnn.augmix.seen_minus_unseen_other_family_gain.seed_noise` | Seed-noise label: Gain vs standard on seen minus gain on unseen_other_family, augmix, simple_cnn | \|mean difference\| < larger of the two seed stds | seen_unseen.csv |
| `rq4.cnn.augmix.seen_minus_unseen_same_family_gain` | Gain vs standard on seen minus gain on unseen_same_family, augmix, simple_cnn | mean over shared seeds of (a - b) | seen_unseen.csv |
| `rq4.cnn.augmix.seen_minus_unseen_same_family_gain.seed0` | Gain vs standard on seen minus gain on unseen_same_family, augmix, simple_cnn, seed 0 | a - b | seen_unseen.csv |
| `rq4.cnn.augmix.seen_minus_unseen_same_family_gain.seed1` | Gain vs standard on seen minus gain on unseen_same_family, augmix, simple_cnn, seed 1 | a - b | seen_unseen.csv |
| `rq4.cnn.augmix.seen_minus_unseen_same_family_gain.seed2` | Gain vs standard on seen minus gain on unseen_same_family, augmix, simple_cnn, seed 2 | a - b | seen_unseen.csv |
| `rq4.cnn.augmix.seen_minus_unseen_same_family_gain.seed_noise` | Seed-noise label: Gain vs standard on seen minus gain on unseen_same_family, augmix, simple_cnn | \|mean difference\| < larger of the two seed stds | seen_unseen.csv |
| `rq4.cnn.augmix.unseen_other_family_gain_vs_standard` | augmix minus standard, accuracy on unseen_other_family, simple_cnn | mean over shared seeds of (a - b) | seen_unseen.csv |
| `rq4.cnn.augmix.unseen_other_family_gain_vs_standard.seed0` | augmix minus standard, accuracy on unseen_other_family, simple_cnn, seed 0 | a - b | seen_unseen.csv |
| `rq4.cnn.augmix.unseen_other_family_gain_vs_standard.seed1` | augmix minus standard, accuracy on unseen_other_family, simple_cnn, seed 1 | a - b | seen_unseen.csv |
| `rq4.cnn.augmix.unseen_other_family_gain_vs_standard.seed2` | augmix minus standard, accuracy on unseen_other_family, simple_cnn, seed 2 | a - b | seen_unseen.csv |
| `rq4.cnn.augmix.unseen_other_family_gain_vs_standard.seed_noise` | Seed-noise label: augmix minus standard, accuracy on unseen_other_family, simple_cnn | \|mean difference\| < larger of the two seed stds | seen_unseen.csv |
| `rq4.cnn.augmix.unseen_same_family_gain_vs_standard` | augmix minus standard, accuracy on unseen_same_family, simple_cnn | mean over shared seeds of (a - b) | seen_unseen.csv |
| `rq4.cnn.augmix.unseen_same_family_gain_vs_standard.seed0` | augmix minus standard, accuracy on unseen_same_family, simple_cnn, seed 0 | a - b | seen_unseen.csv |
| `rq4.cnn.augmix.unseen_same_family_gain_vs_standard.seed1` | augmix minus standard, accuracy on unseen_same_family, simple_cnn, seed 1 | a - b | seen_unseen.csv |
| `rq4.cnn.augmix.unseen_same_family_gain_vs_standard.seed2` | augmix minus standard, accuracy on unseen_same_family, simple_cnn, seed 2 | a - b | seen_unseen.csv |
| `rq4.cnn.augmix.unseen_same_family_gain_vs_standard.seed_noise` | Seed-noise label: augmix minus standard, accuracy on unseen_same_family, simple_cnn | \|mean difference\| < larger of the two seed stds | seen_unseen.csv |
| `rq4.cnn.corruption_aware.seen_gain_vs_standard` | corruption_aware minus standard, accuracy on seen, simple_cnn | mean over shared seeds of (a - b) | seen_unseen.csv |
| `rq4.cnn.corruption_aware.seen_gain_vs_standard.seed0` | corruption_aware minus standard, accuracy on seen, simple_cnn, seed 0 | a - b | seen_unseen.csv |
| `rq4.cnn.corruption_aware.seen_gain_vs_standard.seed1` | corruption_aware minus standard, accuracy on seen, simple_cnn, seed 1 | a - b | seen_unseen.csv |
| `rq4.cnn.corruption_aware.seen_gain_vs_standard.seed2` | corruption_aware minus standard, accuracy on seen, simple_cnn, seed 2 | a - b | seen_unseen.csv |
| `rq4.cnn.corruption_aware.seen_gain_vs_standard.seed_noise` | Seed-noise label: corruption_aware minus standard, accuracy on seen, simple_cnn | \|mean difference\| < larger of the two seed stds | seen_unseen.csv |
| `rq4.cnn.corruption_aware.seen_minus_unseen_other_family_gain` | Gain vs standard on seen minus gain on unseen_other_family, corruption_aware, simple_cnn | mean over shared seeds of (a - b) | seen_unseen.csv |
| `rq4.cnn.corruption_aware.seen_minus_unseen_other_family_gain.seed0` | Gain vs standard on seen minus gain on unseen_other_family, corruption_aware, simple_cnn, seed 0 | a - b | seen_unseen.csv |
| `rq4.cnn.corruption_aware.seen_minus_unseen_other_family_gain.seed1` | Gain vs standard on seen minus gain on unseen_other_family, corruption_aware, simple_cnn, seed 1 | a - b | seen_unseen.csv |
| `rq4.cnn.corruption_aware.seen_minus_unseen_other_family_gain.seed2` | Gain vs standard on seen minus gain on unseen_other_family, corruption_aware, simple_cnn, seed 2 | a - b | seen_unseen.csv |
| `rq4.cnn.corruption_aware.seen_minus_unseen_other_family_gain.seed_noise` | Seed-noise label: Gain vs standard on seen minus gain on unseen_other_family, corruption_aware, simple_cnn | \|mean difference\| < larger of the two seed stds | seen_unseen.csv |
| `rq4.cnn.corruption_aware.seen_minus_unseen_same_family_gain` | Gain vs standard on seen minus gain on unseen_same_family, corruption_aware, simple_cnn | mean over shared seeds of (a - b) | seen_unseen.csv |
| `rq4.cnn.corruption_aware.seen_minus_unseen_same_family_gain.seed0` | Gain vs standard on seen minus gain on unseen_same_family, corruption_aware, simple_cnn, seed 0 | a - b | seen_unseen.csv |
| `rq4.cnn.corruption_aware.seen_minus_unseen_same_family_gain.seed1` | Gain vs standard on seen minus gain on unseen_same_family, corruption_aware, simple_cnn, seed 1 | a - b | seen_unseen.csv |
| `rq4.cnn.corruption_aware.seen_minus_unseen_same_family_gain.seed2` | Gain vs standard on seen minus gain on unseen_same_family, corruption_aware, simple_cnn, seed 2 | a - b | seen_unseen.csv |
| `rq4.cnn.corruption_aware.seen_minus_unseen_same_family_gain.seed_noise` | Seed-noise label: Gain vs standard on seen minus gain on unseen_same_family, corruption_aware, simple_cnn | \|mean difference\| < larger of the two seed stds | seen_unseen.csv |
| `rq4.cnn.corruption_aware.unseen_other_family_gain_vs_standard` | corruption_aware minus standard, accuracy on unseen_other_family, simple_cnn | mean over shared seeds of (a - b) | seen_unseen.csv |
| `rq4.cnn.corruption_aware.unseen_other_family_gain_vs_standard.seed0` | corruption_aware minus standard, accuracy on unseen_other_family, simple_cnn, seed 0 | a - b | seen_unseen.csv |
| `rq4.cnn.corruption_aware.unseen_other_family_gain_vs_standard.seed1` | corruption_aware minus standard, accuracy on unseen_other_family, simple_cnn, seed 1 | a - b | seen_unseen.csv |
| `rq4.cnn.corruption_aware.unseen_other_family_gain_vs_standard.seed2` | corruption_aware minus standard, accuracy on unseen_other_family, simple_cnn, seed 2 | a - b | seen_unseen.csv |
| `rq4.cnn.corruption_aware.unseen_other_family_gain_vs_standard.seed_noise` | Seed-noise label: corruption_aware minus standard, accuracy on unseen_other_family, simple_cnn | \|mean difference\| < larger of the two seed stds | seen_unseen.csv |
| `rq4.cnn.corruption_aware.unseen_same_family_gain_vs_standard` | corruption_aware minus standard, accuracy on unseen_same_family, simple_cnn | mean over shared seeds of (a - b) | seen_unseen.csv |
| `rq4.cnn.corruption_aware.unseen_same_family_gain_vs_standard.seed0` | corruption_aware minus standard, accuracy on unseen_same_family, simple_cnn, seed 0 | a - b | seen_unseen.csv |
| `rq4.cnn.corruption_aware.unseen_same_family_gain_vs_standard.seed1` | corruption_aware minus standard, accuracy on unseen_same_family, simple_cnn, seed 1 | a - b | seen_unseen.csv |
| `rq4.cnn.corruption_aware.unseen_same_family_gain_vs_standard.seed2` | corruption_aware minus standard, accuracy on unseen_same_family, simple_cnn, seed 2 | a - b | seen_unseen.csv |
| `rq4.cnn.corruption_aware.unseen_same_family_gain_vs_standard.seed_noise` | Seed-noise label: corruption_aware minus standard, accuracy on unseen_same_family, simple_cnn | \|mean difference\| < larger of the two seed stds | seen_unseen.csv |
| `rq4.resnet18.augmix.seen_gain_vs_standard` | augmix minus standard, accuracy on seen, resnet18 | mean over shared seeds of (a - b) | seen_unseen.csv |
| `rq4.resnet18.augmix.seen_gain_vs_standard.seed0` | augmix minus standard, accuracy on seen, resnet18, seed 0 | a - b | seen_unseen.csv |
| `rq4.resnet18.augmix.seen_gain_vs_standard.seed1` | augmix minus standard, accuracy on seen, resnet18, seed 1 | a - b | seen_unseen.csv |
| `rq4.resnet18.augmix.seen_gain_vs_standard.seed2` | augmix minus standard, accuracy on seen, resnet18, seed 2 | a - b | seen_unseen.csv |
| `rq4.resnet18.augmix.seen_gain_vs_standard.seed_noise` | Seed-noise label: augmix minus standard, accuracy on seen, resnet18 | \|mean difference\| < larger of the two seed stds | seen_unseen.csv |
| `rq4.resnet18.augmix.seen_minus_unseen_other_family_gain` | Gain vs standard on seen minus gain on unseen_other_family, augmix, resnet18 | mean over shared seeds of (a - b) | seen_unseen.csv |
| `rq4.resnet18.augmix.seen_minus_unseen_other_family_gain.seed0` | Gain vs standard on seen minus gain on unseen_other_family, augmix, resnet18, seed 0 | a - b | seen_unseen.csv |
| `rq4.resnet18.augmix.seen_minus_unseen_other_family_gain.seed1` | Gain vs standard on seen minus gain on unseen_other_family, augmix, resnet18, seed 1 | a - b | seen_unseen.csv |
| `rq4.resnet18.augmix.seen_minus_unseen_other_family_gain.seed2` | Gain vs standard on seen minus gain on unseen_other_family, augmix, resnet18, seed 2 | a - b | seen_unseen.csv |
| `rq4.resnet18.augmix.seen_minus_unseen_other_family_gain.seed_noise` | Seed-noise label: Gain vs standard on seen minus gain on unseen_other_family, augmix, resnet18 | \|mean difference\| < larger of the two seed stds | seen_unseen.csv |
| `rq4.resnet18.augmix.seen_minus_unseen_same_family_gain` | Gain vs standard on seen minus gain on unseen_same_family, augmix, resnet18 | mean over shared seeds of (a - b) | seen_unseen.csv |
| `rq4.resnet18.augmix.seen_minus_unseen_same_family_gain.seed0` | Gain vs standard on seen minus gain on unseen_same_family, augmix, resnet18, seed 0 | a - b | seen_unseen.csv |
| `rq4.resnet18.augmix.seen_minus_unseen_same_family_gain.seed1` | Gain vs standard on seen minus gain on unseen_same_family, augmix, resnet18, seed 1 | a - b | seen_unseen.csv |
| `rq4.resnet18.augmix.seen_minus_unseen_same_family_gain.seed2` | Gain vs standard on seen minus gain on unseen_same_family, augmix, resnet18, seed 2 | a - b | seen_unseen.csv |
| `rq4.resnet18.augmix.seen_minus_unseen_same_family_gain.seed_noise` | Seed-noise label: Gain vs standard on seen minus gain on unseen_same_family, augmix, resnet18 | \|mean difference\| < larger of the two seed stds | seen_unseen.csv |
| `rq4.resnet18.augmix.unseen_other_family_gain_vs_standard` | augmix minus standard, accuracy on unseen_other_family, resnet18 | mean over shared seeds of (a - b) | seen_unseen.csv |
| `rq4.resnet18.augmix.unseen_other_family_gain_vs_standard.seed0` | augmix minus standard, accuracy on unseen_other_family, resnet18, seed 0 | a - b | seen_unseen.csv |
| `rq4.resnet18.augmix.unseen_other_family_gain_vs_standard.seed1` | augmix minus standard, accuracy on unseen_other_family, resnet18, seed 1 | a - b | seen_unseen.csv |
| `rq4.resnet18.augmix.unseen_other_family_gain_vs_standard.seed2` | augmix minus standard, accuracy on unseen_other_family, resnet18, seed 2 | a - b | seen_unseen.csv |
| `rq4.resnet18.augmix.unseen_other_family_gain_vs_standard.seed_noise` | Seed-noise label: augmix minus standard, accuracy on unseen_other_family, resnet18 | \|mean difference\| < larger of the two seed stds | seen_unseen.csv |
| `rq4.resnet18.augmix.unseen_same_family_gain_vs_standard` | augmix minus standard, accuracy on unseen_same_family, resnet18 | mean over shared seeds of (a - b) | seen_unseen.csv |
| `rq4.resnet18.augmix.unseen_same_family_gain_vs_standard.seed0` | augmix minus standard, accuracy on unseen_same_family, resnet18, seed 0 | a - b | seen_unseen.csv |
| `rq4.resnet18.augmix.unseen_same_family_gain_vs_standard.seed1` | augmix minus standard, accuracy on unseen_same_family, resnet18, seed 1 | a - b | seen_unseen.csv |
| `rq4.resnet18.augmix.unseen_same_family_gain_vs_standard.seed2` | augmix minus standard, accuracy on unseen_same_family, resnet18, seed 2 | a - b | seen_unseen.csv |
| `rq4.resnet18.augmix.unseen_same_family_gain_vs_standard.seed_noise` | Seed-noise label: augmix minus standard, accuracy on unseen_same_family, resnet18 | \|mean difference\| < larger of the two seed stds | seen_unseen.csv |
| `rq4.resnet18.corruption_aware.seen_gain_vs_standard` | corruption_aware minus standard, accuracy on seen, resnet18 | mean over shared seeds of (a - b) | seen_unseen.csv |
| `rq4.resnet18.corruption_aware.seen_gain_vs_standard.seed0` | corruption_aware minus standard, accuracy on seen, resnet18, seed 0 | a - b | seen_unseen.csv |
| `rq4.resnet18.corruption_aware.seen_gain_vs_standard.seed1` | corruption_aware minus standard, accuracy on seen, resnet18, seed 1 | a - b | seen_unseen.csv |
| `rq4.resnet18.corruption_aware.seen_gain_vs_standard.seed2` | corruption_aware minus standard, accuracy on seen, resnet18, seed 2 | a - b | seen_unseen.csv |
| `rq4.resnet18.corruption_aware.seen_gain_vs_standard.seed_noise` | Seed-noise label: corruption_aware minus standard, accuracy on seen, resnet18 | \|mean difference\| < larger of the two seed stds | seen_unseen.csv |
| `rq4.resnet18.corruption_aware.seen_minus_unseen_other_family_gain` | Gain vs standard on seen minus gain on unseen_other_family, corruption_aware, resnet18 | mean over shared seeds of (a - b) | seen_unseen.csv |
| `rq4.resnet18.corruption_aware.seen_minus_unseen_other_family_gain.seed0` | Gain vs standard on seen minus gain on unseen_other_family, corruption_aware, resnet18, seed 0 | a - b | seen_unseen.csv |
| `rq4.resnet18.corruption_aware.seen_minus_unseen_other_family_gain.seed1` | Gain vs standard on seen minus gain on unseen_other_family, corruption_aware, resnet18, seed 1 | a - b | seen_unseen.csv |
| `rq4.resnet18.corruption_aware.seen_minus_unseen_other_family_gain.seed2` | Gain vs standard on seen minus gain on unseen_other_family, corruption_aware, resnet18, seed 2 | a - b | seen_unseen.csv |
| `rq4.resnet18.corruption_aware.seen_minus_unseen_other_family_gain.seed_noise` | Seed-noise label: Gain vs standard on seen minus gain on unseen_other_family, corruption_aware, resnet18 | \|mean difference\| < larger of the two seed stds | seen_unseen.csv |
| `rq4.resnet18.corruption_aware.seen_minus_unseen_same_family_gain` | Gain vs standard on seen minus gain on unseen_same_family, corruption_aware, resnet18 | mean over shared seeds of (a - b) | seen_unseen.csv |
| `rq4.resnet18.corruption_aware.seen_minus_unseen_same_family_gain.seed0` | Gain vs standard on seen minus gain on unseen_same_family, corruption_aware, resnet18, seed 0 | a - b | seen_unseen.csv |
| `rq4.resnet18.corruption_aware.seen_minus_unseen_same_family_gain.seed1` | Gain vs standard on seen minus gain on unseen_same_family, corruption_aware, resnet18, seed 1 | a - b | seen_unseen.csv |
| `rq4.resnet18.corruption_aware.seen_minus_unseen_same_family_gain.seed2` | Gain vs standard on seen minus gain on unseen_same_family, corruption_aware, resnet18, seed 2 | a - b | seen_unseen.csv |
| `rq4.resnet18.corruption_aware.seen_minus_unseen_same_family_gain.seed_noise` | Seed-noise label: Gain vs standard on seen minus gain on unseen_same_family, corruption_aware, resnet18 | \|mean difference\| < larger of the two seed stds | seen_unseen.csv |
| `rq4.resnet18.corruption_aware.unseen_other_family_gain_vs_standard` | corruption_aware minus standard, accuracy on unseen_other_family, resnet18 | mean over shared seeds of (a - b) | seen_unseen.csv |
| `rq4.resnet18.corruption_aware.unseen_other_family_gain_vs_standard.seed0` | corruption_aware minus standard, accuracy on unseen_other_family, resnet18, seed 0 | a - b | seen_unseen.csv |
| `rq4.resnet18.corruption_aware.unseen_other_family_gain_vs_standard.seed1` | corruption_aware minus standard, accuracy on unseen_other_family, resnet18, seed 1 | a - b | seen_unseen.csv |
| `rq4.resnet18.corruption_aware.unseen_other_family_gain_vs_standard.seed2` | corruption_aware minus standard, accuracy on unseen_other_family, resnet18, seed 2 | a - b | seen_unseen.csv |
| `rq4.resnet18.corruption_aware.unseen_other_family_gain_vs_standard.seed_noise` | Seed-noise label: corruption_aware minus standard, accuracy on unseen_other_family, resnet18 | \|mean difference\| < larger of the two seed stds | seen_unseen.csv |
| `rq4.resnet18.corruption_aware.unseen_same_family_gain_vs_standard` | corruption_aware minus standard, accuracy on unseen_same_family, resnet18 | mean over shared seeds of (a - b) | seen_unseen.csv |
| `rq4.resnet18.corruption_aware.unseen_same_family_gain_vs_standard.seed0` | corruption_aware minus standard, accuracy on unseen_same_family, resnet18, seed 0 | a - b | seen_unseen.csv |
| `rq4.resnet18.corruption_aware.unseen_same_family_gain_vs_standard.seed1` | corruption_aware minus standard, accuracy on unseen_same_family, resnet18, seed 1 | a - b | seen_unseen.csv |
| `rq4.resnet18.corruption_aware.unseen_same_family_gain_vs_standard.seed2` | corruption_aware minus standard, accuracy on unseen_same_family, resnet18, seed 2 | a - b | seen_unseen.csv |
| `rq4.resnet18.corruption_aware.unseen_same_family_gain_vs_standard.seed_noise` | Seed-noise label: corruption_aware minus standard, accuracy on unseen_same_family, resnet18 | \|mean difference\| < larger of the two seed stds | seen_unseen.csv |
| `rq5.cnn.augmix.blur.abs_degradation` | Absolute drop on blur, simple_cnn + augmix | seed mean of (clean - mean accuracy over its conditions) | per_family.csv |
| `rq5.cnn.augmix.brightness.abs_degradation` | Absolute drop on brightness, simple_cnn + augmix | seed mean of (clean - mean accuracy over its conditions) | per_corruption.csv |
| `rq5.cnn.augmix.contrast.abs_degradation` | Absolute drop on contrast, simple_cnn + augmix | seed mean of (clean - mean accuracy over its conditions) | per_corruption.csv |
| `rq5.cnn.augmix.corruption_mildest_gap` | Drop gap defocus_blur - brightness (corruption mildest gap), simple_cnn + augmix | mean over shared seeds of (a - b) | per_corruption.csv |
| `rq5.cnn.augmix.corruption_mildest_gap.seed0` | Drop gap defocus_blur - brightness (corruption mildest gap), simple_cnn + augmix, seed 0 | a - b | per_corruption.csv |
| `rq5.cnn.augmix.corruption_mildest_gap.seed1` | Drop gap defocus_blur - brightness (corruption mildest gap), simple_cnn + augmix, seed 1 | a - b | per_corruption.csv |
| `rq5.cnn.augmix.corruption_mildest_gap.seed2` | Drop gap defocus_blur - brightness (corruption mildest gap), simple_cnn + augmix, seed 2 | a - b | per_corruption.csv |
| `rq5.cnn.augmix.corruption_mildest_gap.seed_noise` | Seed-noise label: Drop gap defocus_blur - brightness (corruption mildest gap), simple_cnn + augmix | \|mean difference\| < larger of the two seed stds | per_corruption.csv |
| `rq5.cnn.augmix.corruption_second_gap` | Drop gap gaussian_noise - shot_noise (corruption second gap), simple_cnn + augmix | mean over shared seeds of (a - b) | per_corruption.csv |
| `rq5.cnn.augmix.corruption_second_gap.seed0` | Drop gap gaussian_noise - shot_noise (corruption second gap), simple_cnn + augmix, seed 0 | a - b | per_corruption.csv |
| `rq5.cnn.augmix.corruption_second_gap.seed1` | Drop gap gaussian_noise - shot_noise (corruption second gap), simple_cnn + augmix, seed 1 | a - b | per_corruption.csv |
| `rq5.cnn.augmix.corruption_second_gap.seed2` | Drop gap gaussian_noise - shot_noise (corruption second gap), simple_cnn + augmix, seed 2 | a - b | per_corruption.csv |
| `rq5.cnn.augmix.corruption_second_gap.seed_noise` | Seed-noise label: Drop gap gaussian_noise - shot_noise (corruption second gap), simple_cnn + augmix | \|mean difference\| < larger of the two seed stds | per_corruption.csv |
| `rq5.cnn.augmix.corruption_worst_gap` | Drop gap glass_blur - gaussian_noise (corruption worst gap), simple_cnn + augmix | mean over shared seeds of (a - b) | per_corruption.csv |
| `rq5.cnn.augmix.corruption_worst_gap.seed0` | Drop gap glass_blur - gaussian_noise (corruption worst gap), simple_cnn + augmix, seed 0 | a - b | per_corruption.csv |
| `rq5.cnn.augmix.corruption_worst_gap.seed1` | Drop gap glass_blur - gaussian_noise (corruption worst gap), simple_cnn + augmix, seed 1 | a - b | per_corruption.csv |
| `rq5.cnn.augmix.corruption_worst_gap.seed2` | Drop gap glass_blur - gaussian_noise (corruption worst gap), simple_cnn + augmix, seed 2 | a - b | per_corruption.csv |
| `rq5.cnn.augmix.corruption_worst_gap.seed_noise` | Seed-noise label: Drop gap glass_blur - gaussian_noise (corruption worst gap), simple_cnn + augmix | \|mean difference\| < larger of the two seed stds | per_corruption.csv |
| `rq5.cnn.augmix.defocus_blur.abs_degradation` | Absolute drop on defocus_blur, simple_cnn + augmix | seed mean of (clean - mean accuracy over its conditions) | per_corruption.csv |
| `rq5.cnn.augmix.digital.abs_degradation` | Absolute drop on digital, simple_cnn + augmix | seed mean of (clean - mean accuracy over its conditions) | per_family.csv |
| `rq5.cnn.augmix.elastic_transform.abs_degradation` | Absolute drop on elastic_transform, simple_cnn + augmix | seed mean of (clean - mean accuracy over its conditions) | per_corruption.csv |
| `rq5.cnn.augmix.family_mildest_gap` | Drop gap weather - digital (family mildest gap), simple_cnn + augmix | mean over shared seeds of (a - b) | per_family.csv |
| `rq5.cnn.augmix.family_mildest_gap.seed0` | Drop gap weather - digital (family mildest gap), simple_cnn + augmix, seed 0 | a - b | per_family.csv |
| `rq5.cnn.augmix.family_mildest_gap.seed1` | Drop gap weather - digital (family mildest gap), simple_cnn + augmix, seed 1 | a - b | per_family.csv |
| `rq5.cnn.augmix.family_mildest_gap.seed2` | Drop gap weather - digital (family mildest gap), simple_cnn + augmix, seed 2 | a - b | per_family.csv |
| `rq5.cnn.augmix.family_mildest_gap.seed_noise` | Seed-noise label: Drop gap weather - digital (family mildest gap), simple_cnn + augmix | \|mean difference\| < larger of the two seed stds | per_family.csv |
| `rq5.cnn.augmix.family_second_gap` | Drop gap blur - weather (family second gap), simple_cnn + augmix | mean over shared seeds of (a - b) | per_family.csv |
| `rq5.cnn.augmix.family_second_gap.seed0` | Drop gap blur - weather (family second gap), simple_cnn + augmix, seed 0 | a - b | per_family.csv |
| `rq5.cnn.augmix.family_second_gap.seed1` | Drop gap blur - weather (family second gap), simple_cnn + augmix, seed 1 | a - b | per_family.csv |
| `rq5.cnn.augmix.family_second_gap.seed2` | Drop gap blur - weather (family second gap), simple_cnn + augmix, seed 2 | a - b | per_family.csv |
| `rq5.cnn.augmix.family_second_gap.seed_noise` | Seed-noise label: Drop gap blur - weather (family second gap), simple_cnn + augmix | \|mean difference\| < larger of the two seed stds | per_family.csv |
| `rq5.cnn.augmix.fog.abs_degradation` | Absolute drop on fog, simple_cnn + augmix | seed mean of (clean - mean accuracy over its conditions) | per_corruption.csv |
| `rq5.cnn.augmix.frost.abs_degradation` | Absolute drop on frost, simple_cnn + augmix | seed mean of (clean - mean accuracy over its conditions) | per_corruption.csv |
| `rq5.cnn.augmix.gaussian_noise.abs_degradation` | Absolute drop on gaussian_noise, simple_cnn + augmix | seed mean of (clean - mean accuracy over its conditions) | per_corruption.csv |
| `rq5.cnn.augmix.glass_blur.abs_degradation` | Absolute drop on glass_blur, simple_cnn + augmix | seed mean of (clean - mean accuracy over its conditions) | per_corruption.csv |
| `rq5.cnn.augmix.impulse_noise.abs_degradation` | Absolute drop on impulse_noise, simple_cnn + augmix | seed mean of (clean - mean accuracy over its conditions) | per_corruption.csv |
| `rq5.cnn.augmix.jpeg_compression.abs_degradation` | Absolute drop on jpeg_compression, simple_cnn + augmix | seed mean of (clean - mean accuracy over its conditions) | per_corruption.csv |
| `rq5.cnn.augmix.motion_blur.abs_degradation` | Absolute drop on motion_blur, simple_cnn + augmix | seed mean of (clean - mean accuracy over its conditions) | per_corruption.csv |
| `rq5.cnn.augmix.noise.abs_degradation` | Absolute drop on noise, simple_cnn + augmix | seed mean of (clean - mean accuracy over its conditions) | per_family.csv |
| `rq5.cnn.augmix.pixelate.abs_degradation` | Absolute drop on pixelate, simple_cnn + augmix | seed mean of (clean - mean accuracy over its conditions) | per_corruption.csv |
| `rq5.cnn.augmix.shot_noise.abs_degradation` | Absolute drop on shot_noise, simple_cnn + augmix | seed mean of (clean - mean accuracy over its conditions) | per_corruption.csv |
| `rq5.cnn.augmix.snow.abs_degradation` | Absolute drop on snow, simple_cnn + augmix | seed mean of (clean - mean accuracy over its conditions) | per_corruption.csv |
| `rq5.cnn.augmix.weather.abs_degradation` | Absolute drop on weather, simple_cnn + augmix | seed mean of (clean - mean accuracy over its conditions) | per_family.csv |
| `rq5.cnn.augmix.worst_family_gap` | Worst family minus runner-up (noise - blur), simple_cnn + augmix | mean over shared seeds of (a - b) | per_family.csv |
| `rq5.cnn.augmix.worst_family_gap.seed0` | Worst family minus runner-up (noise - blur), simple_cnn + augmix, seed 0 | a - b | per_family.csv |
| `rq5.cnn.augmix.worst_family_gap.seed1` | Worst family minus runner-up (noise - blur), simple_cnn + augmix, seed 1 | a - b | per_family.csv |
| `rq5.cnn.augmix.worst_family_gap.seed2` | Worst family minus runner-up (noise - blur), simple_cnn + augmix, seed 2 | a - b | per_family.csv |
| `rq5.cnn.augmix.worst_family_gap.seed_noise` | Seed-noise label: Worst family minus runner-up (noise - blur), simple_cnn + augmix | \|mean difference\| < larger of the two seed stds | per_family.csv |
| `rq5.cnn.augmix.zoom_blur.abs_degradation` | Absolute drop on zoom_blur, simple_cnn + augmix | seed mean of (clean - mean accuracy over its conditions) | per_corruption.csv |
| `rq5.cnn.corruption_aware.blur.abs_degradation` | Absolute drop on blur, simple_cnn + corruption_aware | seed mean of (clean - mean accuracy over its conditions) | per_family.csv |
| `rq5.cnn.corruption_aware.brightness.abs_degradation` | Absolute drop on brightness, simple_cnn + corruption_aware | seed mean of (clean - mean accuracy over its conditions) | per_corruption.csv |
| `rq5.cnn.corruption_aware.contrast.abs_degradation` | Absolute drop on contrast, simple_cnn + corruption_aware | seed mean of (clean - mean accuracy over its conditions) | per_corruption.csv |
| `rq5.cnn.corruption_aware.corruption_mildest_gap` | Drop gap defocus_blur - brightness (corruption mildest gap), simple_cnn + corruption_aware | mean over shared seeds of (a - b) | per_corruption.csv |
| `rq5.cnn.corruption_aware.corruption_mildest_gap.seed0` | Drop gap defocus_blur - brightness (corruption mildest gap), simple_cnn + corruption_aware, seed 0 | a - b | per_corruption.csv |
| `rq5.cnn.corruption_aware.corruption_mildest_gap.seed1` | Drop gap defocus_blur - brightness (corruption mildest gap), simple_cnn + corruption_aware, seed 1 | a - b | per_corruption.csv |
| `rq5.cnn.corruption_aware.corruption_mildest_gap.seed2` | Drop gap defocus_blur - brightness (corruption mildest gap), simple_cnn + corruption_aware, seed 2 | a - b | per_corruption.csv |
| `rq5.cnn.corruption_aware.corruption_mildest_gap.seed_noise` | Seed-noise label: Drop gap defocus_blur - brightness (corruption mildest gap), simple_cnn + corruption_aware | \|mean difference\| < larger of the two seed stds | per_corruption.csv |
| `rq5.cnn.corruption_aware.corruption_second_gap` | Drop gap pixelate - snow (corruption second gap), simple_cnn + corruption_aware | mean over shared seeds of (a - b) | per_corruption.csv |
| `rq5.cnn.corruption_aware.corruption_second_gap.seed0` | Drop gap pixelate - snow (corruption second gap), simple_cnn + corruption_aware, seed 0 | a - b | per_corruption.csv |
| `rq5.cnn.corruption_aware.corruption_second_gap.seed1` | Drop gap pixelate - snow (corruption second gap), simple_cnn + corruption_aware, seed 1 | a - b | per_corruption.csv |
| `rq5.cnn.corruption_aware.corruption_second_gap.seed2` | Drop gap pixelate - snow (corruption second gap), simple_cnn + corruption_aware, seed 2 | a - b | per_corruption.csv |
| `rq5.cnn.corruption_aware.corruption_second_gap.seed_noise` | Seed-noise label: Drop gap pixelate - snow (corruption second gap), simple_cnn + corruption_aware | \|mean difference\| < larger of the two seed stds | per_corruption.csv |
| `rq5.cnn.corruption_aware.corruption_worst_gap` | Drop gap glass_blur - pixelate (corruption worst gap), simple_cnn + corruption_aware | mean over shared seeds of (a - b) | per_corruption.csv |
| `rq5.cnn.corruption_aware.corruption_worst_gap.seed0` | Drop gap glass_blur - pixelate (corruption worst gap), simple_cnn + corruption_aware, seed 0 | a - b | per_corruption.csv |
| `rq5.cnn.corruption_aware.corruption_worst_gap.seed1` | Drop gap glass_blur - pixelate (corruption worst gap), simple_cnn + corruption_aware, seed 1 | a - b | per_corruption.csv |
| `rq5.cnn.corruption_aware.corruption_worst_gap.seed2` | Drop gap glass_blur - pixelate (corruption worst gap), simple_cnn + corruption_aware, seed 2 | a - b | per_corruption.csv |
| `rq5.cnn.corruption_aware.corruption_worst_gap.seed_noise` | Seed-noise label: Drop gap glass_blur - pixelate (corruption worst gap), simple_cnn + corruption_aware | \|mean difference\| < larger of the two seed stds | per_corruption.csv |
| `rq5.cnn.corruption_aware.defocus_blur.abs_degradation` | Absolute drop on defocus_blur, simple_cnn + corruption_aware | seed mean of (clean - mean accuracy over its conditions) | per_corruption.csv |
| `rq5.cnn.corruption_aware.digital.abs_degradation` | Absolute drop on digital, simple_cnn + corruption_aware | seed mean of (clean - mean accuracy over its conditions) | per_family.csv |
| `rq5.cnn.corruption_aware.elastic_transform.abs_degradation` | Absolute drop on elastic_transform, simple_cnn + corruption_aware | seed mean of (clean - mean accuracy over its conditions) | per_corruption.csv |
| `rq5.cnn.corruption_aware.family_mildest_gap` | Drop gap noise - digital (family mildest gap), simple_cnn + corruption_aware | mean over shared seeds of (a - b) | per_family.csv |
| `rq5.cnn.corruption_aware.family_mildest_gap.seed0` | Drop gap noise - digital (family mildest gap), simple_cnn + corruption_aware, seed 0 | a - b | per_family.csv |
| `rq5.cnn.corruption_aware.family_mildest_gap.seed1` | Drop gap noise - digital (family mildest gap), simple_cnn + corruption_aware, seed 1 | a - b | per_family.csv |
| `rq5.cnn.corruption_aware.family_mildest_gap.seed2` | Drop gap noise - digital (family mildest gap), simple_cnn + corruption_aware, seed 2 | a - b | per_family.csv |
| `rq5.cnn.corruption_aware.family_mildest_gap.seed_noise` | Seed-noise label: Drop gap noise - digital (family mildest gap), simple_cnn + corruption_aware | \|mean difference\| < larger of the two seed stds | per_family.csv |
| `rq5.cnn.corruption_aware.family_second_gap` | Drop gap weather - noise (family second gap), simple_cnn + corruption_aware | mean over shared seeds of (a - b) | per_family.csv |
| `rq5.cnn.corruption_aware.family_second_gap.seed0` | Drop gap weather - noise (family second gap), simple_cnn + corruption_aware, seed 0 | a - b | per_family.csv |
| `rq5.cnn.corruption_aware.family_second_gap.seed1` | Drop gap weather - noise (family second gap), simple_cnn + corruption_aware, seed 1 | a - b | per_family.csv |
| `rq5.cnn.corruption_aware.family_second_gap.seed2` | Drop gap weather - noise (family second gap), simple_cnn + corruption_aware, seed 2 | a - b | per_family.csv |
| `rq5.cnn.corruption_aware.family_second_gap.seed_noise` | Seed-noise label: Drop gap weather - noise (family second gap), simple_cnn + corruption_aware | \|mean difference\| < larger of the two seed stds | per_family.csv |
| `rq5.cnn.corruption_aware.fog.abs_degradation` | Absolute drop on fog, simple_cnn + corruption_aware | seed mean of (clean - mean accuracy over its conditions) | per_corruption.csv |
| `rq5.cnn.corruption_aware.frost.abs_degradation` | Absolute drop on frost, simple_cnn + corruption_aware | seed mean of (clean - mean accuracy over its conditions) | per_corruption.csv |
| `rq5.cnn.corruption_aware.gaussian_noise.abs_degradation` | Absolute drop on gaussian_noise, simple_cnn + corruption_aware | seed mean of (clean - mean accuracy over its conditions) | per_corruption.csv |
| `rq5.cnn.corruption_aware.glass_blur.abs_degradation` | Absolute drop on glass_blur, simple_cnn + corruption_aware | seed mean of (clean - mean accuracy over its conditions) | per_corruption.csv |
| `rq5.cnn.corruption_aware.impulse_noise.abs_degradation` | Absolute drop on impulse_noise, simple_cnn + corruption_aware | seed mean of (clean - mean accuracy over its conditions) | per_corruption.csv |
| `rq5.cnn.corruption_aware.jpeg_compression.abs_degradation` | Absolute drop on jpeg_compression, simple_cnn + corruption_aware | seed mean of (clean - mean accuracy over its conditions) | per_corruption.csv |
| `rq5.cnn.corruption_aware.motion_blur.abs_degradation` | Absolute drop on motion_blur, simple_cnn + corruption_aware | seed mean of (clean - mean accuracy over its conditions) | per_corruption.csv |
| `rq5.cnn.corruption_aware.noise.abs_degradation` | Absolute drop on noise, simple_cnn + corruption_aware | seed mean of (clean - mean accuracy over its conditions) | per_family.csv |
| `rq5.cnn.corruption_aware.pixelate.abs_degradation` | Absolute drop on pixelate, simple_cnn + corruption_aware | seed mean of (clean - mean accuracy over its conditions) | per_corruption.csv |
| `rq5.cnn.corruption_aware.shot_noise.abs_degradation` | Absolute drop on shot_noise, simple_cnn + corruption_aware | seed mean of (clean - mean accuracy over its conditions) | per_corruption.csv |
| `rq5.cnn.corruption_aware.snow.abs_degradation` | Absolute drop on snow, simple_cnn + corruption_aware | seed mean of (clean - mean accuracy over its conditions) | per_corruption.csv |
| `rq5.cnn.corruption_aware.weather.abs_degradation` | Absolute drop on weather, simple_cnn + corruption_aware | seed mean of (clean - mean accuracy over its conditions) | per_family.csv |
| `rq5.cnn.corruption_aware.worst_family_gap` | Worst family minus runner-up (blur - weather), simple_cnn + corruption_aware | mean over shared seeds of (a - b) | per_family.csv |
| `rq5.cnn.corruption_aware.worst_family_gap.seed0` | Worst family minus runner-up (blur - weather), simple_cnn + corruption_aware, seed 0 | a - b | per_family.csv |
| `rq5.cnn.corruption_aware.worst_family_gap.seed1` | Worst family minus runner-up (blur - weather), simple_cnn + corruption_aware, seed 1 | a - b | per_family.csv |
| `rq5.cnn.corruption_aware.worst_family_gap.seed2` | Worst family minus runner-up (blur - weather), simple_cnn + corruption_aware, seed 2 | a - b | per_family.csv |
| `rq5.cnn.corruption_aware.worst_family_gap.seed_noise` | Seed-noise label: Worst family minus runner-up (blur - weather), simple_cnn + corruption_aware | \|mean difference\| < larger of the two seed stds | per_family.csv |
| `rq5.cnn.corruption_aware.zoom_blur.abs_degradation` | Absolute drop on zoom_blur, simple_cnn + corruption_aware | seed mean of (clean - mean accuracy over its conditions) | per_corruption.csv |
| `rq5.cnn.standard.blur.abs_degradation` | Absolute drop on blur, simple_cnn + standard | seed mean of (clean - mean accuracy over its conditions) | per_family.csv |
| `rq5.cnn.standard.brightness.abs_degradation` | Absolute drop on brightness, simple_cnn + standard | seed mean of (clean - mean accuracy over its conditions) | per_corruption.csv |
| `rq5.cnn.standard.contrast.abs_degradation` | Absolute drop on contrast, simple_cnn + standard | seed mean of (clean - mean accuracy over its conditions) | per_corruption.csv |
| `rq5.cnn.standard.corruption_mildest_gap` | Drop gap jpeg_compression - brightness (corruption mildest gap), simple_cnn + standard | mean over shared seeds of (a - b) | per_corruption.csv |
| `rq5.cnn.standard.corruption_mildest_gap.seed0` | Drop gap jpeg_compression - brightness (corruption mildest gap), simple_cnn + standard, seed 0 | a - b | per_corruption.csv |
| `rq5.cnn.standard.corruption_mildest_gap.seed1` | Drop gap jpeg_compression - brightness (corruption mildest gap), simple_cnn + standard, seed 1 | a - b | per_corruption.csv |
| `rq5.cnn.standard.corruption_mildest_gap.seed2` | Drop gap jpeg_compression - brightness (corruption mildest gap), simple_cnn + standard, seed 2 | a - b | per_corruption.csv |
| `rq5.cnn.standard.corruption_mildest_gap.seed_noise` | Seed-noise label: Drop gap jpeg_compression - brightness (corruption mildest gap), simple_cnn + standard | \|mean difference\| < larger of the two seed stds | per_corruption.csv |
| `rq5.cnn.standard.corruption_second_gap` | Drop gap glass_blur - shot_noise (corruption second gap), simple_cnn + standard | mean over shared seeds of (a - b) | per_corruption.csv |
| `rq5.cnn.standard.corruption_second_gap.seed0` | Drop gap glass_blur - shot_noise (corruption second gap), simple_cnn + standard, seed 0 | a - b | per_corruption.csv |
| `rq5.cnn.standard.corruption_second_gap.seed1` | Drop gap glass_blur - shot_noise (corruption second gap), simple_cnn + standard, seed 1 | a - b | per_corruption.csv |
| `rq5.cnn.standard.corruption_second_gap.seed2` | Drop gap glass_blur - shot_noise (corruption second gap), simple_cnn + standard, seed 2 | a - b | per_corruption.csv |
| `rq5.cnn.standard.corruption_second_gap.seed_noise` | Seed-noise label: Drop gap glass_blur - shot_noise (corruption second gap), simple_cnn + standard | \|mean difference\| < larger of the two seed stds | per_corruption.csv |
| `rq5.cnn.standard.corruption_worst_gap` | Drop gap gaussian_noise - glass_blur (corruption worst gap), simple_cnn + standard | mean over shared seeds of (a - b) | per_corruption.csv |
| `rq5.cnn.standard.corruption_worst_gap.seed0` | Drop gap gaussian_noise - glass_blur (corruption worst gap), simple_cnn + standard, seed 0 | a - b | per_corruption.csv |
| `rq5.cnn.standard.corruption_worst_gap.seed1` | Drop gap gaussian_noise - glass_blur (corruption worst gap), simple_cnn + standard, seed 1 | a - b | per_corruption.csv |
| `rq5.cnn.standard.corruption_worst_gap.seed2` | Drop gap gaussian_noise - glass_blur (corruption worst gap), simple_cnn + standard, seed 2 | a - b | per_corruption.csv |
| `rq5.cnn.standard.corruption_worst_gap.seed_noise` | Seed-noise label: Drop gap gaussian_noise - glass_blur (corruption worst gap), simple_cnn + standard | \|mean difference\| < larger of the two seed stds | per_corruption.csv |
| `rq5.cnn.standard.defocus_blur.abs_degradation` | Absolute drop on defocus_blur, simple_cnn + standard | seed mean of (clean - mean accuracy over its conditions) | per_corruption.csv |
| `rq5.cnn.standard.digital.abs_degradation` | Absolute drop on digital, simple_cnn + standard | seed mean of (clean - mean accuracy over its conditions) | per_family.csv |
| `rq5.cnn.standard.elastic_transform.abs_degradation` | Absolute drop on elastic_transform, simple_cnn + standard | seed mean of (clean - mean accuracy over its conditions) | per_corruption.csv |
| `rq5.cnn.standard.family_mildest_gap` | Drop gap weather - digital (family mildest gap), simple_cnn + standard | mean over shared seeds of (a - b) | per_family.csv |
| `rq5.cnn.standard.family_mildest_gap.seed0` | Drop gap weather - digital (family mildest gap), simple_cnn + standard, seed 0 | a - b | per_family.csv |
| `rq5.cnn.standard.family_mildest_gap.seed1` | Drop gap weather - digital (family mildest gap), simple_cnn + standard, seed 1 | a - b | per_family.csv |
| `rq5.cnn.standard.family_mildest_gap.seed2` | Drop gap weather - digital (family mildest gap), simple_cnn + standard, seed 2 | a - b | per_family.csv |
| `rq5.cnn.standard.family_mildest_gap.seed_noise` | Seed-noise label: Drop gap weather - digital (family mildest gap), simple_cnn + standard | \|mean difference\| < larger of the two seed stds | per_family.csv |
| `rq5.cnn.standard.family_second_gap` | Drop gap blur - weather (family second gap), simple_cnn + standard | mean over shared seeds of (a - b) | per_family.csv |
| `rq5.cnn.standard.family_second_gap.seed0` | Drop gap blur - weather (family second gap), simple_cnn + standard, seed 0 | a - b | per_family.csv |
| `rq5.cnn.standard.family_second_gap.seed1` | Drop gap blur - weather (family second gap), simple_cnn + standard, seed 1 | a - b | per_family.csv |
| `rq5.cnn.standard.family_second_gap.seed2` | Drop gap blur - weather (family second gap), simple_cnn + standard, seed 2 | a - b | per_family.csv |
| `rq5.cnn.standard.family_second_gap.seed_noise` | Seed-noise label: Drop gap blur - weather (family second gap), simple_cnn + standard | \|mean difference\| < larger of the two seed stds | per_family.csv |
| `rq5.cnn.standard.fog.abs_degradation` | Absolute drop on fog, simple_cnn + standard | seed mean of (clean - mean accuracy over its conditions) | per_corruption.csv |
| `rq5.cnn.standard.frost.abs_degradation` | Absolute drop on frost, simple_cnn + standard | seed mean of (clean - mean accuracy over its conditions) | per_corruption.csv |
| `rq5.cnn.standard.gaussian_noise.abs_degradation` | Absolute drop on gaussian_noise, simple_cnn + standard | seed mean of (clean - mean accuracy over its conditions) | per_corruption.csv |
| `rq5.cnn.standard.glass_blur.abs_degradation` | Absolute drop on glass_blur, simple_cnn + standard | seed mean of (clean - mean accuracy over its conditions) | per_corruption.csv |
| `rq5.cnn.standard.impulse_noise.abs_degradation` | Absolute drop on impulse_noise, simple_cnn + standard | seed mean of (clean - mean accuracy over its conditions) | per_corruption.csv |
| `rq5.cnn.standard.jpeg_compression.abs_degradation` | Absolute drop on jpeg_compression, simple_cnn + standard | seed mean of (clean - mean accuracy over its conditions) | per_corruption.csv |
| `rq5.cnn.standard.motion_blur.abs_degradation` | Absolute drop on motion_blur, simple_cnn + standard | seed mean of (clean - mean accuracy over its conditions) | per_corruption.csv |
| `rq5.cnn.standard.noise.abs_degradation` | Absolute drop on noise, simple_cnn + standard | seed mean of (clean - mean accuracy over its conditions) | per_family.csv |
| `rq5.cnn.standard.pixelate.abs_degradation` | Absolute drop on pixelate, simple_cnn + standard | seed mean of (clean - mean accuracy over its conditions) | per_corruption.csv |
| `rq5.cnn.standard.shot_noise.abs_degradation` | Absolute drop on shot_noise, simple_cnn + standard | seed mean of (clean - mean accuracy over its conditions) | per_corruption.csv |
| `rq5.cnn.standard.snow.abs_degradation` | Absolute drop on snow, simple_cnn + standard | seed mean of (clean - mean accuracy over its conditions) | per_corruption.csv |
| `rq5.cnn.standard.weather.abs_degradation` | Absolute drop on weather, simple_cnn + standard | seed mean of (clean - mean accuracy over its conditions) | per_family.csv |
| `rq5.cnn.standard.worst_family_gap` | Worst family minus runner-up (noise - blur), simple_cnn + standard | mean over shared seeds of (a - b) | per_family.csv |
| `rq5.cnn.standard.worst_family_gap.seed0` | Worst family minus runner-up (noise - blur), simple_cnn + standard, seed 0 | a - b | per_family.csv |
| `rq5.cnn.standard.worst_family_gap.seed1` | Worst family minus runner-up (noise - blur), simple_cnn + standard, seed 1 | a - b | per_family.csv |
| `rq5.cnn.standard.worst_family_gap.seed2` | Worst family minus runner-up (noise - blur), simple_cnn + standard, seed 2 | a - b | per_family.csv |
| `rq5.cnn.standard.worst_family_gap.seed_noise` | Seed-noise label: Worst family minus runner-up (noise - blur), simple_cnn + standard | \|mean difference\| < larger of the two seed stds | per_family.csv |
| `rq5.cnn.standard.zoom_blur.abs_degradation` | Absolute drop on zoom_blur, simple_cnn + standard | seed mean of (clean - mean accuracy over its conditions) | per_corruption.csv |
| `rq5.mildest_corruption.cnn.augmix` | The mildest corruption by mean drop, simple_cnn + augmix | argmin of the mean absolute degradation | per_corruption.csv |
| `rq5.mildest_corruption.cnn.corruption_aware` | The mildest corruption by mean drop, simple_cnn + corruption_aware | argmin of the mean absolute degradation | per_corruption.csv |
| `rq5.mildest_corruption.cnn.standard` | The mildest corruption by mean drop, simple_cnn + standard | argmin of the mean absolute degradation | per_corruption.csv |
| `rq5.mildest_corruption.resnet18.augmix` | The mildest corruption by mean drop, resnet18 + augmix | argmin of the mean absolute degradation | per_corruption.csv |
| `rq5.mildest_corruption.resnet18.corruption_aware` | The mildest corruption by mean drop, resnet18 + corruption_aware | argmin of the mean absolute degradation | per_corruption.csv |
| `rq5.mildest_corruption.resnet18.standard` | The mildest corruption by mean drop, resnet18 + standard | argmin of the mean absolute degradation | per_corruption.csv |
| `rq5.mildest_family.cnn.augmix` | The mildest family by mean drop, simple_cnn + augmix | argmin of the mean absolute degradation | per_family.csv |
| `rq5.mildest_family.cnn.corruption_aware` | The mildest family by mean drop, simple_cnn + corruption_aware | argmin of the mean absolute degradation | per_family.csv |
| `rq5.mildest_family.cnn.standard` | The mildest family by mean drop, simple_cnn + standard | argmin of the mean absolute degradation | per_family.csv |
| `rq5.mildest_family.resnet18.augmix` | The mildest family by mean drop, resnet18 + augmix | argmin of the mean absolute degradation | per_family.csv |
| `rq5.mildest_family.resnet18.corruption_aware` | The mildest family by mean drop, resnet18 + corruption_aware | argmin of the mean absolute degradation | per_family.csv |
| `rq5.mildest_family.resnet18.standard` | The mildest family by mean drop, resnet18 + standard | argmin of the mean absolute degradation | per_family.csv |
| `rq5.resnet18.augmix.blur.abs_degradation` | Absolute drop on blur, resnet18 + augmix | seed mean of (clean - mean accuracy over its conditions) | per_family.csv |
| `rq5.resnet18.augmix.brightness.abs_degradation` | Absolute drop on brightness, resnet18 + augmix | seed mean of (clean - mean accuracy over its conditions) | per_corruption.csv |
| `rq5.resnet18.augmix.contrast.abs_degradation` | Absolute drop on contrast, resnet18 + augmix | seed mean of (clean - mean accuracy over its conditions) | per_corruption.csv |
| `rq5.resnet18.augmix.corruption_mildest_gap` | Drop gap defocus_blur - brightness (corruption mildest gap), resnet18 + augmix | mean over shared seeds of (a - b) | per_corruption.csv |
| `rq5.resnet18.augmix.corruption_mildest_gap.seed0` | Drop gap defocus_blur - brightness (corruption mildest gap), resnet18 + augmix, seed 0 | a - b | per_corruption.csv |
| `rq5.resnet18.augmix.corruption_mildest_gap.seed1` | Drop gap defocus_blur - brightness (corruption mildest gap), resnet18 + augmix, seed 1 | a - b | per_corruption.csv |
| `rq5.resnet18.augmix.corruption_mildest_gap.seed2` | Drop gap defocus_blur - brightness (corruption mildest gap), resnet18 + augmix, seed 2 | a - b | per_corruption.csv |
| `rq5.resnet18.augmix.corruption_mildest_gap.seed_noise` | Seed-noise label: Drop gap defocus_blur - brightness (corruption mildest gap), resnet18 + augmix | \|mean difference\| < larger of the two seed stds | per_corruption.csv |
| `rq5.resnet18.augmix.corruption_second_gap` | Drop gap gaussian_noise - impulse_noise (corruption second gap), resnet18 + augmix | mean over shared seeds of (a - b) | per_corruption.csv |
| `rq5.resnet18.augmix.corruption_second_gap.seed0` | Drop gap gaussian_noise - impulse_noise (corruption second gap), resnet18 + augmix, seed 0 | a - b | per_corruption.csv |
| `rq5.resnet18.augmix.corruption_second_gap.seed1` | Drop gap gaussian_noise - impulse_noise (corruption second gap), resnet18 + augmix, seed 1 | a - b | per_corruption.csv |
| `rq5.resnet18.augmix.corruption_second_gap.seed2` | Drop gap gaussian_noise - impulse_noise (corruption second gap), resnet18 + augmix, seed 2 | a - b | per_corruption.csv |
| `rq5.resnet18.augmix.corruption_second_gap.seed_noise` | Seed-noise label: Drop gap gaussian_noise - impulse_noise (corruption second gap), resnet18 + augmix | \|mean difference\| < larger of the two seed stds | per_corruption.csv |
| `rq5.resnet18.augmix.corruption_worst_gap` | Drop gap glass_blur - gaussian_noise (corruption worst gap), resnet18 + augmix | mean over shared seeds of (a - b) | per_corruption.csv |
| `rq5.resnet18.augmix.corruption_worst_gap.seed0` | Drop gap glass_blur - gaussian_noise (corruption worst gap), resnet18 + augmix, seed 0 | a - b | per_corruption.csv |
| `rq5.resnet18.augmix.corruption_worst_gap.seed1` | Drop gap glass_blur - gaussian_noise (corruption worst gap), resnet18 + augmix, seed 1 | a - b | per_corruption.csv |
| `rq5.resnet18.augmix.corruption_worst_gap.seed2` | Drop gap glass_blur - gaussian_noise (corruption worst gap), resnet18 + augmix, seed 2 | a - b | per_corruption.csv |
| `rq5.resnet18.augmix.corruption_worst_gap.seed_noise` | Seed-noise label: Drop gap glass_blur - gaussian_noise (corruption worst gap), resnet18 + augmix | \|mean difference\| < larger of the two seed stds | per_corruption.csv |
| `rq5.resnet18.augmix.defocus_blur.abs_degradation` | Absolute drop on defocus_blur, resnet18 + augmix | seed mean of (clean - mean accuracy over its conditions) | per_corruption.csv |
| `rq5.resnet18.augmix.digital.abs_degradation` | Absolute drop on digital, resnet18 + augmix | seed mean of (clean - mean accuracy over its conditions) | per_family.csv |
| `rq5.resnet18.augmix.elastic_transform.abs_degradation` | Absolute drop on elastic_transform, resnet18 + augmix | seed mean of (clean - mean accuracy over its conditions) | per_corruption.csv |
| `rq5.resnet18.augmix.family_mildest_gap` | Drop gap digital - weather (family mildest gap), resnet18 + augmix | mean over shared seeds of (a - b) | per_family.csv |
| `rq5.resnet18.augmix.family_mildest_gap.seed0` | Drop gap digital - weather (family mildest gap), resnet18 + augmix, seed 0 | a - b | per_family.csv |
| `rq5.resnet18.augmix.family_mildest_gap.seed1` | Drop gap digital - weather (family mildest gap), resnet18 + augmix, seed 1 | a - b | per_family.csv |
| `rq5.resnet18.augmix.family_mildest_gap.seed2` | Drop gap digital - weather (family mildest gap), resnet18 + augmix, seed 2 | a - b | per_family.csv |
| `rq5.resnet18.augmix.family_mildest_gap.seed_noise` | Seed-noise label: Drop gap digital - weather (family mildest gap), resnet18 + augmix | \|mean difference\| < larger of the two seed stds | per_family.csv |
| `rq5.resnet18.augmix.family_second_gap` | Drop gap blur - digital (family second gap), resnet18 + augmix | mean over shared seeds of (a - b) | per_family.csv |
| `rq5.resnet18.augmix.family_second_gap.seed0` | Drop gap blur - digital (family second gap), resnet18 + augmix, seed 0 | a - b | per_family.csv |
| `rq5.resnet18.augmix.family_second_gap.seed1` | Drop gap blur - digital (family second gap), resnet18 + augmix, seed 1 | a - b | per_family.csv |
| `rq5.resnet18.augmix.family_second_gap.seed2` | Drop gap blur - digital (family second gap), resnet18 + augmix, seed 2 | a - b | per_family.csv |
| `rq5.resnet18.augmix.family_second_gap.seed_noise` | Seed-noise label: Drop gap blur - digital (family second gap), resnet18 + augmix | \|mean difference\| < larger of the two seed stds | per_family.csv |
| `rq5.resnet18.augmix.fog.abs_degradation` | Absolute drop on fog, resnet18 + augmix | seed mean of (clean - mean accuracy over its conditions) | per_corruption.csv |
| `rq5.resnet18.augmix.frost.abs_degradation` | Absolute drop on frost, resnet18 + augmix | seed mean of (clean - mean accuracy over its conditions) | per_corruption.csv |
| `rq5.resnet18.augmix.gaussian_noise.abs_degradation` | Absolute drop on gaussian_noise, resnet18 + augmix | seed mean of (clean - mean accuracy over its conditions) | per_corruption.csv |
| `rq5.resnet18.augmix.glass_blur.abs_degradation` | Absolute drop on glass_blur, resnet18 + augmix | seed mean of (clean - mean accuracy over its conditions) | per_corruption.csv |
| `rq5.resnet18.augmix.impulse_noise.abs_degradation` | Absolute drop on impulse_noise, resnet18 + augmix | seed mean of (clean - mean accuracy over its conditions) | per_corruption.csv |
| `rq5.resnet18.augmix.jpeg_compression.abs_degradation` | Absolute drop on jpeg_compression, resnet18 + augmix | seed mean of (clean - mean accuracy over its conditions) | per_corruption.csv |
| `rq5.resnet18.augmix.motion_blur.abs_degradation` | Absolute drop on motion_blur, resnet18 + augmix | seed mean of (clean - mean accuracy over its conditions) | per_corruption.csv |
| `rq5.resnet18.augmix.noise.abs_degradation` | Absolute drop on noise, resnet18 + augmix | seed mean of (clean - mean accuracy over its conditions) | per_family.csv |
| `rq5.resnet18.augmix.pixelate.abs_degradation` | Absolute drop on pixelate, resnet18 + augmix | seed mean of (clean - mean accuracy over its conditions) | per_corruption.csv |
| `rq5.resnet18.augmix.shot_noise.abs_degradation` | Absolute drop on shot_noise, resnet18 + augmix | seed mean of (clean - mean accuracy over its conditions) | per_corruption.csv |
| `rq5.resnet18.augmix.snow.abs_degradation` | Absolute drop on snow, resnet18 + augmix | seed mean of (clean - mean accuracy over its conditions) | per_corruption.csv |
| `rq5.resnet18.augmix.weather.abs_degradation` | Absolute drop on weather, resnet18 + augmix | seed mean of (clean - mean accuracy over its conditions) | per_family.csv |
| `rq5.resnet18.augmix.worst_family_gap` | Worst family minus runner-up (noise - blur), resnet18 + augmix | mean over shared seeds of (a - b) | per_family.csv |
| `rq5.resnet18.augmix.worst_family_gap.seed0` | Worst family minus runner-up (noise - blur), resnet18 + augmix, seed 0 | a - b | per_family.csv |
| `rq5.resnet18.augmix.worst_family_gap.seed1` | Worst family minus runner-up (noise - blur), resnet18 + augmix, seed 1 | a - b | per_family.csv |
| `rq5.resnet18.augmix.worst_family_gap.seed2` | Worst family minus runner-up (noise - blur), resnet18 + augmix, seed 2 | a - b | per_family.csv |
| `rq5.resnet18.augmix.worst_family_gap.seed_noise` | Seed-noise label: Worst family minus runner-up (noise - blur), resnet18 + augmix | \|mean difference\| < larger of the two seed stds | per_family.csv |
| `rq5.resnet18.augmix.zoom_blur.abs_degradation` | Absolute drop on zoom_blur, resnet18 + augmix | seed mean of (clean - mean accuracy over its conditions) | per_corruption.csv |
| `rq5.resnet18.corruption_aware.blur.abs_degradation` | Absolute drop on blur, resnet18 + corruption_aware | seed mean of (clean - mean accuracy over its conditions) | per_family.csv |
| `rq5.resnet18.corruption_aware.brightness.abs_degradation` | Absolute drop on brightness, resnet18 + corruption_aware | seed mean of (clean - mean accuracy over its conditions) | per_corruption.csv |
| `rq5.resnet18.corruption_aware.contrast.abs_degradation` | Absolute drop on contrast, resnet18 + corruption_aware | seed mean of (clean - mean accuracy over its conditions) | per_corruption.csv |
| `rq5.resnet18.corruption_aware.corruption_mildest_gap` | Drop gap brightness - contrast (corruption mildest gap), resnet18 + corruption_aware | mean over shared seeds of (a - b) | per_corruption.csv |
| `rq5.resnet18.corruption_aware.corruption_mildest_gap.seed0` | Drop gap brightness - contrast (corruption mildest gap), resnet18 + corruption_aware, seed 0 | a - b | per_corruption.csv |
| `rq5.resnet18.corruption_aware.corruption_mildest_gap.seed1` | Drop gap brightness - contrast (corruption mildest gap), resnet18 + corruption_aware, seed 1 | a - b | per_corruption.csv |
| `rq5.resnet18.corruption_aware.corruption_mildest_gap.seed2` | Drop gap brightness - contrast (corruption mildest gap), resnet18 + corruption_aware, seed 2 | a - b | per_corruption.csv |
| `rq5.resnet18.corruption_aware.corruption_mildest_gap.seed_noise` | Seed-noise label: Drop gap brightness - contrast (corruption mildest gap), resnet18 + corruption_aware | \|mean difference\| < larger of the two seed stds | per_corruption.csv |
| `rq5.resnet18.corruption_aware.corruption_second_gap` | Drop gap pixelate - motion_blur (corruption second gap), resnet18 + corruption_aware | mean over shared seeds of (a - b) | per_corruption.csv |
| `rq5.resnet18.corruption_aware.corruption_second_gap.seed0` | Drop gap pixelate - motion_blur (corruption second gap), resnet18 + corruption_aware, seed 0 | a - b | per_corruption.csv |
| `rq5.resnet18.corruption_aware.corruption_second_gap.seed1` | Drop gap pixelate - motion_blur (corruption second gap), resnet18 + corruption_aware, seed 1 | a - b | per_corruption.csv |
| `rq5.resnet18.corruption_aware.corruption_second_gap.seed2` | Drop gap pixelate - motion_blur (corruption second gap), resnet18 + corruption_aware, seed 2 | a - b | per_corruption.csv |
| `rq5.resnet18.corruption_aware.corruption_second_gap.seed_noise` | Seed-noise label: Drop gap pixelate - motion_blur (corruption second gap), resnet18 + corruption_aware | \|mean difference\| < larger of the two seed stds | per_corruption.csv |
| `rq5.resnet18.corruption_aware.corruption_worst_gap` | Drop gap glass_blur - pixelate (corruption worst gap), resnet18 + corruption_aware | mean over shared seeds of (a - b) | per_corruption.csv |
| `rq5.resnet18.corruption_aware.corruption_worst_gap.seed0` | Drop gap glass_blur - pixelate (corruption worst gap), resnet18 + corruption_aware, seed 0 | a - b | per_corruption.csv |
| `rq5.resnet18.corruption_aware.corruption_worst_gap.seed1` | Drop gap glass_blur - pixelate (corruption worst gap), resnet18 + corruption_aware, seed 1 | a - b | per_corruption.csv |
| `rq5.resnet18.corruption_aware.corruption_worst_gap.seed2` | Drop gap glass_blur - pixelate (corruption worst gap), resnet18 + corruption_aware, seed 2 | a - b | per_corruption.csv |
| `rq5.resnet18.corruption_aware.corruption_worst_gap.seed_noise` | Seed-noise label: Drop gap glass_blur - pixelate (corruption worst gap), resnet18 + corruption_aware | \|mean difference\| < larger of the two seed stds | per_corruption.csv |
| `rq5.resnet18.corruption_aware.defocus_blur.abs_degradation` | Absolute drop on defocus_blur, resnet18 + corruption_aware | seed mean of (clean - mean accuracy over its conditions) | per_corruption.csv |
| `rq5.resnet18.corruption_aware.digital.abs_degradation` | Absolute drop on digital, resnet18 + corruption_aware | seed mean of (clean - mean accuracy over its conditions) | per_family.csv |
| `rq5.resnet18.corruption_aware.elastic_transform.abs_degradation` | Absolute drop on elastic_transform, resnet18 + corruption_aware | seed mean of (clean - mean accuracy over its conditions) | per_corruption.csv |
| `rq5.resnet18.corruption_aware.family_mildest_gap` | Drop gap noise - digital (family mildest gap), resnet18 + corruption_aware | mean over shared seeds of (a - b) | per_family.csv |
| `rq5.resnet18.corruption_aware.family_mildest_gap.seed0` | Drop gap noise - digital (family mildest gap), resnet18 + corruption_aware, seed 0 | a - b | per_family.csv |
| `rq5.resnet18.corruption_aware.family_mildest_gap.seed1` | Drop gap noise - digital (family mildest gap), resnet18 + corruption_aware, seed 1 | a - b | per_family.csv |
| `rq5.resnet18.corruption_aware.family_mildest_gap.seed2` | Drop gap noise - digital (family mildest gap), resnet18 + corruption_aware, seed 2 | a - b | per_family.csv |
| `rq5.resnet18.corruption_aware.family_mildest_gap.seed_noise` | Seed-noise label: Drop gap noise - digital (family mildest gap), resnet18 + corruption_aware | \|mean difference\| < larger of the two seed stds | per_family.csv |
| `rq5.resnet18.corruption_aware.family_second_gap` | Drop gap weather - noise (family second gap), resnet18 + corruption_aware | mean over shared seeds of (a - b) | per_family.csv |
| `rq5.resnet18.corruption_aware.family_second_gap.seed0` | Drop gap weather - noise (family second gap), resnet18 + corruption_aware, seed 0 | a - b | per_family.csv |
| `rq5.resnet18.corruption_aware.family_second_gap.seed1` | Drop gap weather - noise (family second gap), resnet18 + corruption_aware, seed 1 | a - b | per_family.csv |
| `rq5.resnet18.corruption_aware.family_second_gap.seed2` | Drop gap weather - noise (family second gap), resnet18 + corruption_aware, seed 2 | a - b | per_family.csv |
| `rq5.resnet18.corruption_aware.family_second_gap.seed_noise` | Seed-noise label: Drop gap weather - noise (family second gap), resnet18 + corruption_aware | \|mean difference\| < larger of the two seed stds | per_family.csv |
| `rq5.resnet18.corruption_aware.fog.abs_degradation` | Absolute drop on fog, resnet18 + corruption_aware | seed mean of (clean - mean accuracy over its conditions) | per_corruption.csv |
| `rq5.resnet18.corruption_aware.frost.abs_degradation` | Absolute drop on frost, resnet18 + corruption_aware | seed mean of (clean - mean accuracy over its conditions) | per_corruption.csv |
| `rq5.resnet18.corruption_aware.gaussian_noise.abs_degradation` | Absolute drop on gaussian_noise, resnet18 + corruption_aware | seed mean of (clean - mean accuracy over its conditions) | per_corruption.csv |
| `rq5.resnet18.corruption_aware.glass_blur.abs_degradation` | Absolute drop on glass_blur, resnet18 + corruption_aware | seed mean of (clean - mean accuracy over its conditions) | per_corruption.csv |
| `rq5.resnet18.corruption_aware.impulse_noise.abs_degradation` | Absolute drop on impulse_noise, resnet18 + corruption_aware | seed mean of (clean - mean accuracy over its conditions) | per_corruption.csv |
| `rq5.resnet18.corruption_aware.jpeg_compression.abs_degradation` | Absolute drop on jpeg_compression, resnet18 + corruption_aware | seed mean of (clean - mean accuracy over its conditions) | per_corruption.csv |
| `rq5.resnet18.corruption_aware.motion_blur.abs_degradation` | Absolute drop on motion_blur, resnet18 + corruption_aware | seed mean of (clean - mean accuracy over its conditions) | per_corruption.csv |
| `rq5.resnet18.corruption_aware.noise.abs_degradation` | Absolute drop on noise, resnet18 + corruption_aware | seed mean of (clean - mean accuracy over its conditions) | per_family.csv |
| `rq5.resnet18.corruption_aware.pixelate.abs_degradation` | Absolute drop on pixelate, resnet18 + corruption_aware | seed mean of (clean - mean accuracy over its conditions) | per_corruption.csv |
| `rq5.resnet18.corruption_aware.shot_noise.abs_degradation` | Absolute drop on shot_noise, resnet18 + corruption_aware | seed mean of (clean - mean accuracy over its conditions) | per_corruption.csv |
| `rq5.resnet18.corruption_aware.snow.abs_degradation` | Absolute drop on snow, resnet18 + corruption_aware | seed mean of (clean - mean accuracy over its conditions) | per_corruption.csv |
| `rq5.resnet18.corruption_aware.weather.abs_degradation` | Absolute drop on weather, resnet18 + corruption_aware | seed mean of (clean - mean accuracy over its conditions) | per_family.csv |
| `rq5.resnet18.corruption_aware.worst_family_gap` | Worst family minus runner-up (blur - weather), resnet18 + corruption_aware | mean over shared seeds of (a - b) | per_family.csv |
| `rq5.resnet18.corruption_aware.worst_family_gap.seed0` | Worst family minus runner-up (blur - weather), resnet18 + corruption_aware, seed 0 | a - b | per_family.csv |
| `rq5.resnet18.corruption_aware.worst_family_gap.seed1` | Worst family minus runner-up (blur - weather), resnet18 + corruption_aware, seed 1 | a - b | per_family.csv |
| `rq5.resnet18.corruption_aware.worst_family_gap.seed2` | Worst family minus runner-up (blur - weather), resnet18 + corruption_aware, seed 2 | a - b | per_family.csv |
| `rq5.resnet18.corruption_aware.worst_family_gap.seed_noise` | Seed-noise label: Worst family minus runner-up (blur - weather), resnet18 + corruption_aware | \|mean difference\| < larger of the two seed stds | per_family.csv |
| `rq5.resnet18.corruption_aware.zoom_blur.abs_degradation` | Absolute drop on zoom_blur, resnet18 + corruption_aware | seed mean of (clean - mean accuracy over its conditions) | per_corruption.csv |
| `rq5.resnet18.standard.blur.abs_degradation` | Absolute drop on blur, resnet18 + standard | seed mean of (clean - mean accuracy over its conditions) | per_family.csv |
| `rq5.resnet18.standard.brightness.abs_degradation` | Absolute drop on brightness, resnet18 + standard | seed mean of (clean - mean accuracy over its conditions) | per_corruption.csv |
| `rq5.resnet18.standard.contrast.abs_degradation` | Absolute drop on contrast, resnet18 + standard | seed mean of (clean - mean accuracy over its conditions) | per_corruption.csv |
| `rq5.resnet18.standard.corruption_mildest_gap` | Drop gap fog - brightness (corruption mildest gap), resnet18 + standard | mean over shared seeds of (a - b) | per_corruption.csv |
| `rq5.resnet18.standard.corruption_mildest_gap.seed0` | Drop gap fog - brightness (corruption mildest gap), resnet18 + standard, seed 0 | a - b | per_corruption.csv |
| `rq5.resnet18.standard.corruption_mildest_gap.seed1` | Drop gap fog - brightness (corruption mildest gap), resnet18 + standard, seed 1 | a - b | per_corruption.csv |
| `rq5.resnet18.standard.corruption_mildest_gap.seed2` | Drop gap fog - brightness (corruption mildest gap), resnet18 + standard, seed 2 | a - b | per_corruption.csv |
| `rq5.resnet18.standard.corruption_mildest_gap.seed_noise` | Seed-noise label: Drop gap fog - brightness (corruption mildest gap), resnet18 + standard | \|mean difference\| < larger of the two seed stds | per_corruption.csv |
| `rq5.resnet18.standard.corruption_second_gap` | Drop gap glass_blur - impulse_noise (corruption second gap), resnet18 + standard | mean over shared seeds of (a - b) | per_corruption.csv |
| `rq5.resnet18.standard.corruption_second_gap.seed0` | Drop gap glass_blur - impulse_noise (corruption second gap), resnet18 + standard, seed 0 | a - b | per_corruption.csv |
| `rq5.resnet18.standard.corruption_second_gap.seed1` | Drop gap glass_blur - impulse_noise (corruption second gap), resnet18 + standard, seed 1 | a - b | per_corruption.csv |
| `rq5.resnet18.standard.corruption_second_gap.seed2` | Drop gap glass_blur - impulse_noise (corruption second gap), resnet18 + standard, seed 2 | a - b | per_corruption.csv |
| `rq5.resnet18.standard.corruption_second_gap.seed_noise` | Seed-noise label: Drop gap glass_blur - impulse_noise (corruption second gap), resnet18 + standard | \|mean difference\| < larger of the two seed stds | per_corruption.csv |
| `rq5.resnet18.standard.corruption_worst_gap` | Drop gap gaussian_noise - glass_blur (corruption worst gap), resnet18 + standard | mean over shared seeds of (a - b) | per_corruption.csv |
| `rq5.resnet18.standard.corruption_worst_gap.seed0` | Drop gap gaussian_noise - glass_blur (corruption worst gap), resnet18 + standard, seed 0 | a - b | per_corruption.csv |
| `rq5.resnet18.standard.corruption_worst_gap.seed1` | Drop gap gaussian_noise - glass_blur (corruption worst gap), resnet18 + standard, seed 1 | a - b | per_corruption.csv |
| `rq5.resnet18.standard.corruption_worst_gap.seed2` | Drop gap gaussian_noise - glass_blur (corruption worst gap), resnet18 + standard, seed 2 | a - b | per_corruption.csv |
| `rq5.resnet18.standard.corruption_worst_gap.seed_noise` | Seed-noise label: Drop gap gaussian_noise - glass_blur (corruption worst gap), resnet18 + standard | \|mean difference\| < larger of the two seed stds | per_corruption.csv |
| `rq5.resnet18.standard.defocus_blur.abs_degradation` | Absolute drop on defocus_blur, resnet18 + standard | seed mean of (clean - mean accuracy over its conditions) | per_corruption.csv |
| `rq5.resnet18.standard.digital.abs_degradation` | Absolute drop on digital, resnet18 + standard | seed mean of (clean - mean accuracy over its conditions) | per_family.csv |
| `rq5.resnet18.standard.elastic_transform.abs_degradation` | Absolute drop on elastic_transform, resnet18 + standard | seed mean of (clean - mean accuracy over its conditions) | per_corruption.csv |
| `rq5.resnet18.standard.family_mildest_gap` | Drop gap weather - digital (family mildest gap), resnet18 + standard | mean over shared seeds of (a - b) | per_family.csv |
| `rq5.resnet18.standard.family_mildest_gap.seed0` | Drop gap weather - digital (family mildest gap), resnet18 + standard, seed 0 | a - b | per_family.csv |
| `rq5.resnet18.standard.family_mildest_gap.seed1` | Drop gap weather - digital (family mildest gap), resnet18 + standard, seed 1 | a - b | per_family.csv |
| `rq5.resnet18.standard.family_mildest_gap.seed2` | Drop gap weather - digital (family mildest gap), resnet18 + standard, seed 2 | a - b | per_family.csv |
| `rq5.resnet18.standard.family_mildest_gap.seed_noise` | Seed-noise label: Drop gap weather - digital (family mildest gap), resnet18 + standard | \|mean difference\| < larger of the two seed stds | per_family.csv |
| `rq5.resnet18.standard.family_second_gap` | Drop gap blur - weather (family second gap), resnet18 + standard | mean over shared seeds of (a - b) | per_family.csv |
| `rq5.resnet18.standard.family_second_gap.seed0` | Drop gap blur - weather (family second gap), resnet18 + standard, seed 0 | a - b | per_family.csv |
| `rq5.resnet18.standard.family_second_gap.seed1` | Drop gap blur - weather (family second gap), resnet18 + standard, seed 1 | a - b | per_family.csv |
| `rq5.resnet18.standard.family_second_gap.seed2` | Drop gap blur - weather (family second gap), resnet18 + standard, seed 2 | a - b | per_family.csv |
| `rq5.resnet18.standard.family_second_gap.seed_noise` | Seed-noise label: Drop gap blur - weather (family second gap), resnet18 + standard | \|mean difference\| < larger of the two seed stds | per_family.csv |
| `rq5.resnet18.standard.fog.abs_degradation` | Absolute drop on fog, resnet18 + standard | seed mean of (clean - mean accuracy over its conditions) | per_corruption.csv |
| `rq5.resnet18.standard.frost.abs_degradation` | Absolute drop on frost, resnet18 + standard | seed mean of (clean - mean accuracy over its conditions) | per_corruption.csv |
| `rq5.resnet18.standard.gaussian_noise.abs_degradation` | Absolute drop on gaussian_noise, resnet18 + standard | seed mean of (clean - mean accuracy over its conditions) | per_corruption.csv |
| `rq5.resnet18.standard.glass_blur.abs_degradation` | Absolute drop on glass_blur, resnet18 + standard | seed mean of (clean - mean accuracy over its conditions) | per_corruption.csv |
| `rq5.resnet18.standard.impulse_noise.abs_degradation` | Absolute drop on impulse_noise, resnet18 + standard | seed mean of (clean - mean accuracy over its conditions) | per_corruption.csv |
| `rq5.resnet18.standard.jpeg_compression.abs_degradation` | Absolute drop on jpeg_compression, resnet18 + standard | seed mean of (clean - mean accuracy over its conditions) | per_corruption.csv |
| `rq5.resnet18.standard.motion_blur.abs_degradation` | Absolute drop on motion_blur, resnet18 + standard | seed mean of (clean - mean accuracy over its conditions) | per_corruption.csv |
| `rq5.resnet18.standard.noise.abs_degradation` | Absolute drop on noise, resnet18 + standard | seed mean of (clean - mean accuracy over its conditions) | per_family.csv |
| `rq5.resnet18.standard.pixelate.abs_degradation` | Absolute drop on pixelate, resnet18 + standard | seed mean of (clean - mean accuracy over its conditions) | per_corruption.csv |
| `rq5.resnet18.standard.shot_noise.abs_degradation` | Absolute drop on shot_noise, resnet18 + standard | seed mean of (clean - mean accuracy over its conditions) | per_corruption.csv |
| `rq5.resnet18.standard.snow.abs_degradation` | Absolute drop on snow, resnet18 + standard | seed mean of (clean - mean accuracy over its conditions) | per_corruption.csv |
| `rq5.resnet18.standard.weather.abs_degradation` | Absolute drop on weather, resnet18 + standard | seed mean of (clean - mean accuracy over its conditions) | per_family.csv |
| `rq5.resnet18.standard.worst_family_gap` | Worst family minus runner-up (noise - blur), resnet18 + standard | mean over shared seeds of (a - b) | per_family.csv |
| `rq5.resnet18.standard.worst_family_gap.seed0` | Worst family minus runner-up (noise - blur), resnet18 + standard, seed 0 | a - b | per_family.csv |
| `rq5.resnet18.standard.worst_family_gap.seed1` | Worst family minus runner-up (noise - blur), resnet18 + standard, seed 1 | a - b | per_family.csv |
| `rq5.resnet18.standard.worst_family_gap.seed2` | Worst family minus runner-up (noise - blur), resnet18 + standard, seed 2 | a - b | per_family.csv |
| `rq5.resnet18.standard.worst_family_gap.seed_noise` | Seed-noise label: Worst family minus runner-up (noise - blur), resnet18 + standard | \|mean difference\| < larger of the two seed stds | per_family.csv |
| `rq5.resnet18.standard.zoom_blur.abs_degradation` | Absolute drop on zoom_blur, resnet18 + standard | seed mean of (clean - mean accuracy over its conditions) | per_corruption.csv |
| `rq5.runner_up_corruption.cnn.augmix` | The runner up corruption by mean drop, simple_cnn + augmix | position 1 of the corruptions sorted by mean drop, largest first | per_corruption.csv |
| `rq5.runner_up_corruption.cnn.corruption_aware` | The runner up corruption by mean drop, simple_cnn + corruption_aware | position 1 of the corruptions sorted by mean drop, largest first | per_corruption.csv |
| `rq5.runner_up_corruption.cnn.standard` | The runner up corruption by mean drop, simple_cnn + standard | position 1 of the corruptions sorted by mean drop, largest first | per_corruption.csv |
| `rq5.runner_up_corruption.resnet18.augmix` | The runner up corruption by mean drop, resnet18 + augmix | position 1 of the corruptions sorted by mean drop, largest first | per_corruption.csv |
| `rq5.runner_up_corruption.resnet18.corruption_aware` | The runner up corruption by mean drop, resnet18 + corruption_aware | position 1 of the corruptions sorted by mean drop, largest first | per_corruption.csv |
| `rq5.runner_up_corruption.resnet18.standard` | The runner up corruption by mean drop, resnet18 + standard | position 1 of the corruptions sorted by mean drop, largest first | per_corruption.csv |
| `rq5.runner_up_family.cnn.augmix` | The runner up family by mean drop, simple_cnn + augmix | position 1 of the familys sorted by mean drop, largest first | per_family.csv |
| `rq5.runner_up_family.cnn.corruption_aware` | The runner up family by mean drop, simple_cnn + corruption_aware | position 1 of the familys sorted by mean drop, largest first | per_family.csv |
| `rq5.runner_up_family.cnn.standard` | The runner up family by mean drop, simple_cnn + standard | position 1 of the familys sorted by mean drop, largest first | per_family.csv |
| `rq5.runner_up_family.resnet18.augmix` | The runner up family by mean drop, resnet18 + augmix | position 1 of the familys sorted by mean drop, largest first | per_family.csv |
| `rq5.runner_up_family.resnet18.corruption_aware` | The runner up family by mean drop, resnet18 + corruption_aware | position 1 of the familys sorted by mean drop, largest first | per_family.csv |
| `rq5.runner_up_family.resnet18.standard` | The runner up family by mean drop, resnet18 + standard | position 1 of the familys sorted by mean drop, largest first | per_family.csv |
| `rq5.second_mildest_corruption.cnn.augmix` | The second mildest corruption by mean drop, simple_cnn + augmix | position -2 of the corruptions sorted by mean drop, largest first | per_corruption.csv |
| `rq5.second_mildest_corruption.cnn.corruption_aware` | The second mildest corruption by mean drop, simple_cnn + corruption_aware | position -2 of the corruptions sorted by mean drop, largest first | per_corruption.csv |
| `rq5.second_mildest_corruption.cnn.standard` | The second mildest corruption by mean drop, simple_cnn + standard | position -2 of the corruptions sorted by mean drop, largest first | per_corruption.csv |
| `rq5.second_mildest_corruption.resnet18.augmix` | The second mildest corruption by mean drop, resnet18 + augmix | position -2 of the corruptions sorted by mean drop, largest first | per_corruption.csv |
| `rq5.second_mildest_corruption.resnet18.corruption_aware` | The second mildest corruption by mean drop, resnet18 + corruption_aware | position -2 of the corruptions sorted by mean drop, largest first | per_corruption.csv |
| `rq5.second_mildest_corruption.resnet18.standard` | The second mildest corruption by mean drop, resnet18 + standard | position -2 of the corruptions sorted by mean drop, largest first | per_corruption.csv |
| `rq5.second_mildest_family.cnn.augmix` | The second mildest family by mean drop, simple_cnn + augmix | position -2 of the familys sorted by mean drop, largest first | per_family.csv |
| `rq5.second_mildest_family.cnn.corruption_aware` | The second mildest family by mean drop, simple_cnn + corruption_aware | position -2 of the familys sorted by mean drop, largest first | per_family.csv |
| `rq5.second_mildest_family.cnn.standard` | The second mildest family by mean drop, simple_cnn + standard | position -2 of the familys sorted by mean drop, largest first | per_family.csv |
| `rq5.second_mildest_family.resnet18.augmix` | The second mildest family by mean drop, resnet18 + augmix | position -2 of the familys sorted by mean drop, largest first | per_family.csv |
| `rq5.second_mildest_family.resnet18.corruption_aware` | The second mildest family by mean drop, resnet18 + corruption_aware | position -2 of the familys sorted by mean drop, largest first | per_family.csv |
| `rq5.second_mildest_family.resnet18.standard` | The second mildest family by mean drop, resnet18 + standard | position -2 of the familys sorted by mean drop, largest first | per_family.csv |
| `rq5.third_corruption.cnn.augmix` | The third corruption by mean drop, simple_cnn + augmix | position 2 of the corruptions sorted by mean drop, largest first | per_corruption.csv |
| `rq5.third_corruption.cnn.corruption_aware` | The third corruption by mean drop, simple_cnn + corruption_aware | position 2 of the corruptions sorted by mean drop, largest first | per_corruption.csv |
| `rq5.third_corruption.cnn.standard` | The third corruption by mean drop, simple_cnn + standard | position 2 of the corruptions sorted by mean drop, largest first | per_corruption.csv |
| `rq5.third_corruption.resnet18.augmix` | The third corruption by mean drop, resnet18 + augmix | position 2 of the corruptions sorted by mean drop, largest first | per_corruption.csv |
| `rq5.third_corruption.resnet18.corruption_aware` | The third corruption by mean drop, resnet18 + corruption_aware | position 2 of the corruptions sorted by mean drop, largest first | per_corruption.csv |
| `rq5.third_corruption.resnet18.standard` | The third corruption by mean drop, resnet18 + standard | position 2 of the corruptions sorted by mean drop, largest first | per_corruption.csv |
| `rq5.third_family.cnn.augmix` | The third family by mean drop, simple_cnn + augmix | position 2 of the familys sorted by mean drop, largest first | per_family.csv |
| `rq5.third_family.cnn.corruption_aware` | The third family by mean drop, simple_cnn + corruption_aware | position 2 of the familys sorted by mean drop, largest first | per_family.csv |
| `rq5.third_family.cnn.standard` | The third family by mean drop, simple_cnn + standard | position 2 of the familys sorted by mean drop, largest first | per_family.csv |
| `rq5.third_family.resnet18.augmix` | The third family by mean drop, resnet18 + augmix | position 2 of the familys sorted by mean drop, largest first | per_family.csv |
| `rq5.third_family.resnet18.corruption_aware` | The third family by mean drop, resnet18 + corruption_aware | position 2 of the familys sorted by mean drop, largest first | per_family.csv |
| `rq5.third_family.resnet18.standard` | The third family by mean drop, resnet18 + standard | position 2 of the familys sorted by mean drop, largest first | per_family.csv |
| `rq5.worst_corruption.cnn.augmix` | The worst corruption by mean drop, simple_cnn + augmix | argmax of the mean absolute degradation | per_corruption.csv |
| `rq5.worst_corruption.cnn.corruption_aware` | The worst corruption by mean drop, simple_cnn + corruption_aware | argmax of the mean absolute degradation | per_corruption.csv |
| `rq5.worst_corruption.cnn.standard` | The worst corruption by mean drop, simple_cnn + standard | argmax of the mean absolute degradation | per_corruption.csv |
| `rq5.worst_corruption.resnet18.augmix` | The worst corruption by mean drop, resnet18 + augmix | argmax of the mean absolute degradation | per_corruption.csv |
| `rq5.worst_corruption.resnet18.corruption_aware` | The worst corruption by mean drop, resnet18 + corruption_aware | argmax of the mean absolute degradation | per_corruption.csv |
| `rq5.worst_corruption.resnet18.standard` | The worst corruption by mean drop, resnet18 + standard | argmax of the mean absolute degradation | per_corruption.csv |
| `rq5.worst_family.cnn.augmix` | The worst family by mean drop, simple_cnn + augmix | argmax of the mean absolute degradation | per_family.csv |
| `rq5.worst_family.cnn.corruption_aware` | The worst family by mean drop, simple_cnn + corruption_aware | argmax of the mean absolute degradation | per_family.csv |
| `rq5.worst_family.cnn.standard` | The worst family by mean drop, simple_cnn + standard | argmax of the mean absolute degradation | per_family.csv |
| `rq5.worst_family.resnet18.augmix` | The worst family by mean drop, resnet18 + augmix | argmax of the mean absolute degradation | per_family.csv |
| `rq5.worst_family.resnet18.corruption_aware` | The worst family by mean drop, resnet18 + corruption_aware | argmax of the mean absolute degradation | per_family.csv |
| `rq5.worst_family.resnet18.standard` | The worst family by mean drop, resnet18 + standard | argmax of the mean absolute degradation | per_family.csv |
| `setup.device_names` | Distinct device_name values with their run counts | value counts | results/runs.csv |
| `setup.git_commits` | Distinct commits (7 characters) with their run counts | value counts | results/runs.csv |
| `setup.n_dirty_runs` | Runs with uncommitted changes | count of git_dirty | results/runs.csv |
| `setup.n_params.cnn` | Trainable parameters of simple_cnn | n_params | results/runs.csv |
| `setup.n_params.resnet18` | Trainable parameters of resnet18 | n_params | results/runs.csv |
| `setup.n_runs` | Runs in the registry | rows of the 18 | results/runs.csv |
| `setup.python_versions` | Distinct python_version values with their run counts | value counts | results/runs.csv |
| `setup.torch_versions` | Distinct torch_version values with their run counts | value counts | results/runs.csv |
| `setup.torchvision_versions` | Distinct torchvision_version values with their run counts | value counts | results/runs.csv |
