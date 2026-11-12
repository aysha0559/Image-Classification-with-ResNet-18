---
title: CIFAR-10 Robustness Study
emoji: 🔬
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
short_description: CIFAR-10-C robustness results and a live corruption demo
license: mit
---

# CIFAR-10 robustness study: results viewer and demo

How image classifiers fail under common corruptions, and how robustness-oriented training reduces
that failure. A Simple CNN and ResNet-18 trained on CIFAR-10 with standard augmentation, AugMix
and corruption-aware training, 3 seeds each, tested on CIFAR-10-C.

- **Results:** the main tables and the 8 figures of the report, from the study's committed result
  files.
- **Try it:** pick a CIFAR-10 test image (with its real CIFAR-10-C versions) or upload an image
  (with the study's own 4 corruptions), and compare the top-3 predictions of the 6 seed-0 models.
  Uploaded images are kept in memory only and never stored.

The models are downloaded at start from the model repo named in the Space variable
`HF_MODEL_REPO`. This file is the Space's `README.md` (`app/README.md` in the repository); the
Space is built from the repository's `Dockerfile`.

Data: CIFAR-10 (Krizhevsky, 2009) and CIFAR-10-C (Hendrycks & Dietterich, 2019; CC BY 4.0).
Code: MIT licence.
