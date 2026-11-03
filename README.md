# CIFAR-10 Robustness Study

How much accuracy do image classifiers (a simple CNN and ResNet-18) lose on corrupted images, and
do data augmentation (AugMix) and corruption-aware training reduce that loss? We train on CIFAR-10,
evaluate on clean CIFAR-10 and CIFAR-10-C (15 corruptions x 5 severities), and report accuracy,
macro-F1, degradation and mCE, with everything reproducible on free Google Colab GPUs.

**Status: in development.** See [docs/PROJECT_OVERVIEW.md](docs/PROJECT_OVERVIEW.md) for the plan.
