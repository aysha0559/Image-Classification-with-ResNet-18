# Project Overview: Robustness of Image Classifiers to Real-World Image Corruptions

> **Working title:** *How Well Do Image Classifiers Survive Real-World Image Corruptions? A Study of CNN and ResNet-18 Robustness on CIFAR-10-C, Comparing Standard and Corruption-Aware Training*
>
> **Status:** Planning. This document explains the project before any code is written.
> **Last updated:** 2026-10-31

---

## Table of Contents

1. [The Project in One Paragraph](#1-the-project-in-one-paragraph)
2. [The Problem We Are Studying](#2-the-problem-we-are-studying)
3. [Domain and Key Concepts](#3-domain-and-key-concepts)
4. [Research Questions and Hypotheses](#4-research-questions-and-hypotheses)
5. [What We Are Building (Components)](#5-what-we-are-building-components)
6. [Datasets](#6-datasets)
7. [Models](#7-models)
8. [Training Strategies (The Core Comparison)](#8-training-strategies-the-core-comparison)
9. [Evaluation Metrics](#9-evaluation-metrics)
10. [Experimental Design](#10-experimental-design)
11. [Scope: In and Out](#11-scope-in-and-out)
12. [Tools, Environment, and Compute Budget](#12-tools-environment-and-compute-budget)
13. [Proposed Repository Structure](#13-proposed-repository-structure)
14. [Deliverables](#14-deliverables)
15. [Roadmap and Milestones](#15-roadmap-and-milestones)
16. [Benefits to Society and Business](#16-benefits-to-society-and-business)
17. [How to Present This for a Scholarship](#17-how-to-present-this-for-a-scholarship)
18. [Risks and Pitfalls](#18-risks-and-pitfalls)
19. [Possible Extensions](#19-possible-extensions)
20. [Glossary](#20-glossary)
21. [References](#21-references)

---

## 1. The Project in One Paragraph

Image-recognition models score very well on clean benchmark photos, but real cameras produce blurry, noisy, foggy, dark, or heavily compressed images. This project **measures how much accuracy two image classifiers (a simple CNN and ResNet-18) lose when images are corrupted**, and **tests whether better training methods (data augmentation and corruption-aware training) reduce that loss**. We train on **CIFAR-10**, test on clean CIFAR-10 and on **CIFAR-10-C** (the same test images with 15 corruption types at 5 severity levels), and report **accuracy, macro F1-score, and performance degradation**. The result is a **reproducible GitHub repository** containing code, charts, and a short research report. All of it can run on free **Google Colab** GPUs.

---

## 2. The Problem We Are Studying

### 2.1 The gap between the lab and the real world

Most deep-learning models are trained and tested on images from the **same distribution**: similar cameras, lighting, and quality. Under those conditions a model like ResNet-18 can reach **about 93–95% accuracy** on CIFAR-10.

Deployed models see different conditions:

| Real-world situation | Corruption it produces |
|---|---|
| A cheap or old phone camera | Sensor noise, low resolution |
| A car driving at night or in rain | Low brightness, noise, motion blur, spatter |
| A drone or CCTV camera in bad weather | Fog, snow, frost |
| A moving camera or subject | Motion blur, zoom blur |
| Images sent over WhatsApp or a slow network | JPEG compression, pixelation |
| A camera that is out of focus | Defocus blur |

On such images, accuracy can drop by **20–30 percentage points or more**, even though a human would still recognise the object easily. This is called **distribution shift**, and models that fail under it are **not robust**.

### 2.2 Why this matters

A model that is 95% accurate in the lab and 60% accurate in fog is unsafe in a self-driving car, unreliable in a medical device, and frustrating in a consumer app. Before deploying a model, we need to know **how much performance it loses under realistic corruptions** and **what cheap training changes reduce that loss**. This project answers both questions at small scale.

---

## 3. Domain and Key Concepts

**Primary domain:** Computer Vision → Image Classification
**Research sub-field:** Robustness and Reliability of Machine Learning (also called *Trustworthy AI* or *Robust ML*)
**Related areas:** Distribution shift, data augmentation, AI safety, model evaluation

### Key concepts in plain language

| Concept | Meaning |
|---|---|
| **Image classification** | Given an image, predict which category it belongs to (e.g., "cat", "truck"). |
| **CNN (Convolutional Neural Network)** | A neural network that learns visual patterns (edges, textures, shapes) with sliding filters. It is the standard architecture for images. |
| **ResNet (Residual Network)** | A deeper CNN with *skip connections*, which let the network train well at greater depth. ResNet-18 has 18 layers. |
| **Clean vs. corrupted data** | *Clean* means the original, unaltered images. *Corrupted* means the same images with blur, noise, weather effects, and similar damage added. |
| **Distribution shift** | Test data looks different from training data. Corruption is one kind of shift. |
| **Robustness** | How well a model keeps its performance under shift. |
| **Data augmentation** | Randomly altering training images (flips, crops, colour changes, mixing) so the model sees more variety. |
| **Corruption-aware training** | Deliberately adding realistic corruptions (noise, blur, and so on) to training images so the model learns to handle them. |
| **Severity** | How strong a corruption is. CIFAR-10-C uses levels 1 (mild) to 5 (extreme). |
| **Performance degradation** | How much a metric drops from clean to corrupted data. |

---

## 4. Research Questions and Hypotheses

A clear set of research questions is what turns a coding exercise into a **research project**.

| # | Research Question | Hypothesis |
|---|---|---|
| **RQ1** | How much do a simple CNN and ResNet-18 degrade on corrupted images compared with clean images? | Both degrade substantially. Degradation grows with severity. |
| **RQ2** | Does a stronger architecture (ResNet-18) also mean more robustness, or only higher clean accuracy? | ResNet-18 has higher clean *and* corrupted accuracy, but its **relative** degradation may not be much smaller. Clean accuracy ≠ robustness. |
| **RQ3** | Do data augmentation (e.g., AugMix) and corruption-aware training reduce degradation compared with standard training? | Yes. Both reduce degradation, with the largest gains on corruption types similar to those seen in training. |
| **RQ4** | Does corruption-aware training generalise to corruption types **not** seen during training? | Partly. Gains are large on *seen* corruptions and smaller on *unseen* ones. This is a key, honest finding. |
| **RQ5** | Which corruption families (noise, blur, weather, digital) hurt the most? | Noise and blur cause the largest drops. Brightness causes the smallest. |

---

## 5. What We Are Building (Components)

This is an **experimental research system**, not an app with a user interface. It has six parts:

```
┌──────────────────────────────────────────────────────────────────────┐
│                         EXPERIMENT PIPELINE                          │
│                                                                      │
│  1. DATA            2. MODELS          3. TRAINING                   │
│  CIFAR-10 loader    Simple CNN         Standard                      │
│  CIFAR-10-C loader  ResNet-18          + Data augmentation (AugMix)  │
│  Corruption fns                        + Corruption-aware training   │
│        │                 │                    │                      │
│        └─────────────────┴────────┬───────────┘                      │
│                                   ▼                                  │
│  4. EVALUATION  → clean test + 15 corruptions × 5 severities         │
│                   accuracy, macro-F1, degradation, mCE               │
│                                   ▼                                  │
│  5. ANALYSIS    → tables, charts, per-corruption breakdown           │
│                                   ▼                                  │
│  6. REPORTING   → research report + README + reproducible repo       │
└──────────────────────────────────────────────────────────────────────┘
```

1. **Data module:** downloads and loads CIFAR-10 and CIFAR-10-C, applies normalisation and augmentations, and creates train/validation splits.
2. **Model module:** defines the simple CNN baseline and a CIFAR-adapted ResNet-18.
3. **Training module:** a single training loop configured by a YAML file (model, strategy, epochs, seed). It saves checkpoints and logs.
4. **Evaluation module:** evaluates any checkpoint on clean data and on every (corruption, severity) pair, and writes results to CSV.
5. **Analysis module:** aggregates results across seeds and produces charts and tables.
6. **Reporting:** a research report (PDF/Markdown) and a polished README.

---

## 6. Datasets

### 6.1 CIFAR-10 (clean data)

- **60,000** colour images, **32×32 pixels**, in **10 classes**: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.
- **50,000 training** and **10,000 test** images. The classes are perfectly balanced (6,000 per class).
- Created by Krizhevsky (2009). Built into `torchvision.datasets.CIFAR10`, so it downloads automatically.
- Plan: take a **validation split** of 5,000 images from the training set for model selection. The test set is used only for final reporting.

### 6.2 CIFAR-10-C (corrupted data)

- Created by **Hendrycks & Dietterich (ICLR 2019)**, *"Benchmarking Neural Network Robustness to Common Corruptions and Perturbations."*
- It contains the **10,000 CIFAR-10 test images**, each with corruptions applied at **5 severity levels**.
- **15 standard corruptions** in **4 families** (plus 4 extra "held-out" corruptions):

| Family | Corruptions |
|---|---|
| **Noise** | Gaussian noise, Shot noise, Impulse noise |
| **Blur** | Defocus blur, Glass blur, Motion blur, Zoom blur |
| **Weather** | Snow, Frost, Fog |
| **Digital** | Brightness, Contrast, Elastic transform, Pixelate, JPEG compression |
| *Extra (optional)* | Speckle noise, Gaussian blur, Spatter, Saturate |

- **Size:** about 2.9 GB, distributed as `.npy` files on Zenodo. Each corruption file holds 50,000 images (10,000 × 5 severities, in order), and a shared `labels.npy` holds the labels.
- **Important:** CIFAR-10-C is a **test-only** benchmark. It must **never** be used for training or tuning (see [Risks](#18-risks-and-pitfalls)).

---

## 7. Models

### 7.1 Simple CNN (baseline)

A small, deliberately simple network that gives us a reference point.

```
Input 3×32×32
→ [Conv 3×3, 32] → BatchNorm → ReLU → [Conv 3×3, 32] → BN → ReLU → MaxPool → Dropout
→ [Conv 3×3, 64] → BN → ReLU → [Conv 3×3, 64] → BN → ReLU → MaxPool → Dropout
→ [Conv 3×3, 128] → BN → ReLU → MaxPool → Dropout
→ Flatten → Linear(→256) → ReLU → Dropout → Linear(→10)
```

- About 0.5–1 million parameters. Expected clean accuracy is **about 80–87%**.
- Purpose: show how a basic model behaves and give a baseline for normalised metrics.

### 7.2 ResNet-18 (main model)

- The standard ResNet-18 (He et al., 2016), **adapted for 32×32 images**: the first layer is a 3×3 convolution with stride 1, and the initial max-pool is removed. The ImageNet version shrinks small images too aggressively.
- About **11 million parameters**. Expected clean accuracy is **about 92–95%**, depending on the number of epochs.
- **Trained from scratch** (no ImageNet pre-training), so the comparison stays fair and controlled.

---

## 8. Training Strategies (The Core Comparison)

Every model is trained with each strategy below. Everything else (optimiser, epochs, learning-rate schedule) stays **identical**, so any difference comes from the strategy.

| ID | Strategy | What the model sees during training | Role |
|---|---|---|---|
| **S1** | **Standard training** | Clean images + basic augmentation (random crop with padding, horizontal flip) | The baseline. This is "normal" practice. |
| **S2** | **Data augmentation (AugMix)** | S1 + **AugMix**, which mixes several randomly augmented versions of each image. Optionally includes the Jensen-Shannon consistency loss. | Tests a published, general-purpose robustness method. `torchvision.transforms.AugMix` provides it. |
| **S3** | **Corruption-aware training** | S1 + random synthetic corruptions (noise, blur, contrast, and so on) applied to a fraction of training images at random severities | Tests whether directly exposing the model to corruptions helps. |
| *S0 (optional ablation)* | *No augmentation* | Clean images only | Shows how much even basic augmentation matters. |

### Making corruption-aware training scientifically honest

If we train on *exactly* the corruptions we test on, a good score proves little. The model has simply seen the test conditions. To keep the comparison fair, **split the corruptions into two groups**:

- **Seen corruptions:** used in S3 training, e.g., Gaussian noise, defocus blur, contrast, JPEG compression. They are **generated by our own code on training images**, never taken from CIFAR-10-C.
- **Unseen (held-out) corruptions:** never used in training, e.g., snow, fog, frost, motion blur, elastic transform, pixelate.

We then report results **separately for seen and unseen corruptions**. This answers RQ4 and is the kind of careful design that reviewers notice.

---

## 9. Evaluation Metrics

### 9.1 Accuracy
Fraction of images classified correctly. Simple and intuitive.

### 9.2 Macro F1-score
The F1-score (harmonic mean of precision and recall) is computed **for each class separately** and then averaged equally across the 10 classes.
*Why include it:* CIFAR-10 is balanced, but corruptions can push a model to over-predict certain classes, e.g., everything foggy becomes "airplane" or "ship". Macro F1 exposes this per-class collapse, which accuracy can hide.

### 9.3 Performance degradation
How much performance drops from clean to corrupted data.

- **Absolute degradation:** `Δ = Metric_clean − Metric_corrupted` (in percentage points)
- **Relative degradation:** `Δ_rel = (Metric_clean − Metric_corrupted) / Metric_clean × 100%`

Report it **per corruption**, **per severity**, **per family**, and **on average**.

### 9.4 Mean Corruption Error (mCE), recommended extra
The standard metric from the CIFAR-10-C paper:

```
CE_c(model) = Σ_s Error(model, c, s)  /  Σ_s Error(baseline, c, s)
mCE         = average of CE_c over all 15 corruptions
```

The original paper uses AlexNet as the baseline. Here we use **our simple CNN with standard training (S1)** as the reference, so mCE = 100% for the baseline and lower is better. Including mCE makes the results comparable in style to published work.

### 9.5 Relative mCE, optional
Measures degradation *relative to each model's own clean error*. It separates "robust" from "just more accurate."

---

## 10. Experimental Design

### 10.1 Experiment matrix

| | S1 Standard | S2 AugMix | S3 Corruption-aware |
|---|---|---|---|
| **Simple CNN** | Run | Run | Run |
| **ResNet-18** | Run | Run | Run |

- **6 configurations × 3 random seeds = 18 training runs.** Seeds such as 0, 1, and 2 let us report the **mean ± standard deviation**, which shows the results are not luck.
- Each trained model is evaluated on **1 clean set + 15 corruptions × 5 severities = 76 test conditions**.

### 10.2 Controlled settings (identical for all runs)

| Setting | Proposed value |
|---|---|
| Optimiser | SGD, momentum 0.9, Nesterov |
| Learning rate | 0.1, cosine annealing schedule |
| Weight decay | 5e-4 |
| Batch size | 128 |
| Epochs | 30–50 for ResNet-18 (Colab budget); 30 for the CNN |
| Mixed precision | Enabled (faster on Colab GPUs) |
| Seeds | Fixed for Python, NumPy, and PyTorch; cuDNN deterministic where feasible |

### 10.3 Planned outputs (charts and tables)

1. **Main results table:** clean accuracy, clean macro-F1, mean corrupted accuracy, mean corrupted macro-F1, mean degradation, mCE, for all 6 configurations (mean ± std).
2. **Accuracy vs. severity line chart:** one line per configuration, severity 0 (clean) to 5.
3. **Per-corruption heatmap:** rows are corruptions, columns are configurations, and cells show accuracy or degradation.
4. **Per-family bar chart:** noise, blur, weather, and digital degradation for each strategy.
5. **Seen vs. unseen corruption chart** for S3 (answers RQ4).
6. **Confusion matrices** (clean vs. heavy fog or noise) to show *which* classes collapse.
7. **Sample image grid:** one image shown under every corruption and severity, which is useful for the report and the README.
8. **Training curves:** loss and accuracy per epoch.

---

## 11. Scope: In and Out

### In scope
- CIFAR-10 training; CIFAR-10 + CIFAR-10-C evaluation.
- Two architectures: simple CNN and ResNet-18 (from scratch).
- Three training strategies (S1–S3), plus an optional S0 ablation.
- Metrics: accuracy, macro F1, degradation (absolute and relative), and mCE.
- Multiple seeds, a reproducible configuration, charts, and a written report.
- Runs on free Google Colab.

### Out of scope (and why)
| Not included | Reason |
|---|---|
| Adversarial attacks (FGSM, PGD) | A different kind of robustness (worst-case, malicious). Good future work. |
| ImageNet / ImageNet-C | Needs far more compute than Colab provides. |
| Pre-trained / transfer-learning models | Adds a confounding factor. Can be an extension. |
| Deploying an app or API | This is a research study, not a product. |
| New architectures or novel methods | The contribution is a **careful, reproducible empirical comparison**, which is valuable on its own. |

---

## 12. Tools, Environment, and Compute Budget

### Tech stack
| Purpose | Tool |
|---|---|
| Language | Python 3.10+ |
| Deep learning | PyTorch, torchvision |
| Metrics | scikit-learn (`f1_score`, `confusion_matrix`) |
| Data handling | NumPy, pandas |
| Charts | matplotlib, seaborn |
| Configuration | YAML config files |
| Experiment tracking | CSV logs (simple). Optionally TensorBoard or Weights & Biases. |
| Compute | Google Colab (free T4 GPU) |
| Version control | Git + GitHub |
| Report | Markdown → PDF, or LaTeX (Overleaf) |

### Compute estimate (rough; to be confirmed in the first runs)
| Task | Approx. time on Colab T4 |
|---|---|
| Simple CNN, 30 epochs | ~5–10 min |
| ResNet-18, 30–50 epochs (with mixed precision) | ~20–45 min |
| AugMix runs | ~1.5–3× slower than standard |
| Evaluation of one model on all of CIFAR-10-C (750k images) | ~3–10 min |
| **Full study (18 runs + evaluation)** | **~12–20 GPU-hours in total**, spread over several Colab sessions |

### Colab practicalities
- Free Colab sessions **disconnect** (roughly after 1 to several hours of use or when idle). Save **checkpoints and results to Google Drive** after every epoch so runs can resume.
- Download CIFAR-10-C **once** to Google Drive and copy it to the Colab local disk at the start of each session. Reading from Drive directly is slow.
- Keep all logic in `.py` files in the repo. Colab notebooks should only **call** them, which keeps the work reproducible and the GitHub repo clean.

---

## 13. Proposed Repository Structure

```
cifar10-robustness-study/
├── README.md                  # Project summary, key results, how to reproduce
├── docs/
│   ├── PROJECT_OVERVIEW.md    # (this file) the planning document
│   └── agent-prompts/         # build + validation prompts per phase
├── requirements.txt           # Pinned dependency versions
├── LICENSE                    # e.g., MIT
├── configs/                   # One YAML per experiment
│   ├── cnn_standard.yaml
│   ├── cnn_augmix.yaml
│   ├── cnn_corruption_aware.yaml
│   ├── resnet18_standard.yaml
│   ├── resnet18_augmix.yaml
│   └── resnet18_corruption_aware.yaml
├── src/
│   ├── data/
│   │   ├── cifar10.py         # Clean data loaders + splits
│   │   ├── cifar10c.py        # CIFAR-10-C loader (corruption, severity)
│   │   └── corruptions.py     # Our own corruption functions for S3 training
│   ├── models/
│   │   ├── simple_cnn.py
│   │   └── resnet.py          # CIFAR-adapted ResNet-18
│   ├── train.py               # Training entry point
│   ├── evaluate.py            # Clean + CIFAR-10-C evaluation
│   ├── metrics.py             # Accuracy, macro-F1, degradation, mCE
│   └── utils.py               # Seeding, checkpointing, logging
├── scripts/
│   ├── download_cifar10c.sh
│   ├── run_all.sh             # Reproduces every experiment
│   └── make_figures.py        # Builds all charts from results CSVs
├── notebooks/
│   ├── 00_colab_setup.ipynb   # Mount Drive, install, download data
│   ├── 01_explore_data.ipynb  # Visualise corruptions
│   └── 02_analysis.ipynb      # Results exploration
├── results/
│   ├── raw/                   # Per-run CSVs (metrics per corruption/severity)
│   ├── tables/                # Aggregated tables
│   └── figures/               # Final charts (PNG/PDF)
├── report/
│   └── report.md / report.pdf # The research report
└── tests/                     # Small sanity tests (shapes, metric correctness)
```

---

## 14. Deliverables

1. **GitHub repository** with clean, documented, reproducible code (one command or notebook to reproduce).
2. **Results:** CSV files and summary tables for all 18 runs × 76 test conditions.
3. **Charts:** the 8 figures listed in [Section 10.3](#103-planned-outputs-charts-and-tables).
4. **Research report** (about 6–10 pages) with this structure:
   - Abstract
   - 1. Introduction and motivation
   - 2. Related work (CIFAR-10-C, AugMix, robustness literature)
   - 3. Methodology (data, models, strategies, metrics)
   - 4. Experimental setup
   - 5. Results (answers to RQ1–RQ5)
   - 6. Discussion (real-world implications, limitations)
   - 7. Conclusion and future work
   - References
5. **Polished README** with a results summary, a key chart, and reproduction steps. Reviewers often read *only* this.
6. *(Optional)* A short blog post, poster, or 2-minute video summary for the scholarship application.

---

## 15. Roadmap and Milestones

| Phase | Goal | Output | Est. time |
|---|---|---|---|
| **0. Learning** | Understand CNNs, ResNets, PyTorch basics, and the CIFAR-10-C paper | Notes, familiarity | 3–7 days |
| **1. Setup** | Repo, environment, Colab notebook, data download to Drive | Working skeleton | 1–2 days |
| **2. Data** | Loaders for CIFAR-10 and CIFAR-10-C; visualise corruptions | Sample image grid | 1–2 days |
| **3. Baseline** | Simple CNN + standard training + clean evaluation | First trained model | 2–3 days |
| **4. Evaluation** | Full CIFAR-10-C evaluation + metrics module | First degradation numbers | 2–3 days |
| **5. ResNet-18** | Add ResNet-18; repeat steps 3–4 | Second model's results | 2–3 days |
| **6. Strategies** | Implement AugMix (S2) + corruption-aware training (S3) | All 6 configurations run | 4–6 days |
| **7. Seeds** | Run all configurations × 3 seeds | Final raw results | 3–5 days (GPU time) |
| **8. Analysis** | Aggregation, charts, tables | All figures | 2–4 days |
| **9. Report** | Write report + README | Final deliverables | 4–7 days |
| **Total** | | | **~4–6 weeks** part-time |

**Tip:** Build the full pipeline end to end for **one** configuration first (CNN + S1). Add the others only once that works.

---

## 16. Benefits to Society and Business

This study is small (32×32 images), but it addresses a **real, high-stakes problem**: AI systems that work in the lab and fail in the real world. The methods and findings carry over to larger systems.

### 16.1 Society

| Area | How robustness research helps |
|---|---|
| **Road safety / autonomous vehicles** | Cars must recognise pedestrians and signs in fog, rain, snow, glare, and darkness. Corruption robustness is a direct safety requirement. |
| **Healthcare** | Medical images (X-ray, ultrasound, dermatology photos) vary with equipment quality, motion, and noise. Robust models reduce misdiagnosis risk, especially in clinics with older equipment. |
| **Equity and access in developing regions** | Many people use low-cost phones with noisy cameras, and images are heavily compressed when shared. Robust models make AI tools (crop-disease detection, document scanning, health apps) **work for everyone**, not only users with expensive hardware. |
| **Agriculture** | Drone and phone-based crop and pest detection must handle varying light, dust, motion blur, and weather. |
| **Disaster response and environment** | Satellite and drone imagery is often degraded by clouds, smoke, or haze, which matters for flood mapping, wildfire monitoring, and search and rescue. |
| **Accessibility** | Apps that describe the world to visually impaired users receive blurry, poorly lit photos and must still be reliable. |
| **Public trust in AI** | Measuring and reporting robustness honestly is a core part of **trustworthy, responsible AI**. |

### 16.2 Business and industry

| Area | Value |
|---|---|
| **Manufacturing quality control** | Visual inspection systems face changing lighting, dust on lenses, and vibration blur. Robust models mean fewer false alarms and fewer missed defects. |
| **Retail and e-commerce** | Product recognition from user-uploaded photos of varying quality. |
| **Security and surveillance** | CCTV at night, in rain, or at low resolution. |
| **Insurance and fintech** | Damage assessment and document/ID verification from customer phone photos. |
| **Model risk and MLOps** | A **robustness evaluation pipeline** like this one is a practical tool companies can add to model testing *before deployment*, catching failures early and cheaply. |
| **Regulatory compliance** | Regulations such as the **EU AI Act** require high-risk AI systems to be accurate and **robust**. Benchmarks like this are how robustness is demonstrated. |
| **Cost savings** | The project shows that **cheap training changes** (augmentation) can improve reliability without bigger models or more data collection. |

### 16.3 Research and education value
- A **clean, reproducible benchmark** that other students can reuse and extend.
- It shows how to run a **fair experimental comparison** (controlled variables, multiple seeds, held-out corruptions).
- It reproduces and verifies published findings, and **reproducibility** is itself a recognised research contribution.

---

## 17. How to Present This for a Scholarship

Selection committees look for **curiosity, rigour, impact, and communication**. Frame the project around those:

1. **Lead with the real-world problem, not the code.**
   > *"AI models that recognise images with 95% accuracy can drop below 70% when the photo is foggy or blurry. I investigated why, and whether simple training changes can fix it."*
2. **Show scientific thinking:** research questions, hypotheses, controlled experiments, multiple seeds, a seen/unseen corruption split, and honest limitations.
3. **Show impact:** link the findings to safety, healthcare, and **equitable AI for low-cost devices**, which is a strong personal and social angle.
4. **Show results visually:** one striking chart (accuracy vs. severity) plus the sample corruption grid.
5. **Show reproducibility:** "Anyone can reproduce every result with one command on free Colab."
6. **Be honest about limitations:** small images, one dataset, no adversarial robustness. Honesty reads as maturity.
7. **Point to the future:** what she would study next with the scholarship (e.g., robustness in medical imaging, larger datasets, test-time adaptation).

**Suggested one-line summary for the application:**
> *An empirical study of how image classifiers fail under real-world image corruptions, and how robustness-oriented training reduces that failure, with fully reproducible PyTorch code on CIFAR-10-C.*

---

## 18. Risks and Pitfalls

| Risk | Consequence | Mitigation |
|---|---|---|
| **Training or tuning on CIFAR-10-C** | Data leakage; results become meaningless | Use CIFAR-10-C *only* for final evaluation. Generate training corruptions with our own code. |
| **Same corruptions in training and test (S3)** | Overstated robustness | Report seen and unseen corruptions separately. |
| **Unfair comparison** (different epochs or hyperparameters per strategy) | Conclusions not valid | Identical settings for all runs. Only the strategy changes. |
| **Single seed** | Results may be luck | 3 seeds, report mean ± std. |
| **Colab disconnections** | Lost training progress | Checkpoint to Google Drive every epoch; resumable training. |
| **CIFAR-10-C download/storage** (~2.9 GB) | Slow sessions | Store on Drive once; copy to local disk per session; load corruptions lazily. |
| **AugMix slowness** | Exceeds GPU budget | Fewer epochs, or skip the JSD loss (plain AugMix augmentation). |
| **Scope creep** | Project never finishes | Stick to Section 11. Put extra ideas in "Future work". |
| **Non-reproducibility** | Weakens credibility | Pinned `requirements.txt`, config files, fixed seeds, and a `run_all` script. |

---

## 19. Possible Extensions

Only if time allows, or as "future work" in the report:

- **Pre-trained ResNet-18** (ImageNet weights) vs. from scratch: does pre-training improve robustness?
- **More architectures:** Wide-ResNet, MobileNet (edge devices), or a small Vision Transformer.
- **Test-time adaptation** (e.g., updating BatchNorm statistics on corrupted data).
- **Calibration analysis:** are models *overconfident* on corrupted images? (Expected Calibration Error)
- **Adversarial robustness** (FGSM/PGD) compared with corruption robustness.
- **CIFAR-10.1 / CIFAR-10.2:** natural distribution shift, not synthetic corruptions.
- **Domain transfer:** apply the same pipeline to a medical or agricultural image dataset.
- **Interactive demo** (Gradio/Streamlit): upload an image, add corruptions with a slider, and watch predictions change. Good for presentations.

---

## 20. Glossary

| Term | Definition |
|---|---|
| **Epoch** | One full pass of the training data through the model. |
| **Batch size** | Number of images processed together in one training step. |
| **Learning rate** | How large each model update is during training. |
| **Overfitting** | The model memorises training data and performs worse on new data. |
| **Validation set** | Data held out from training and used to make decisions (e.g., when to stop). |
| **Seed** | A number that fixes randomness so results can be reproduced. |
| **BatchNorm** | A layer that normalises activations to stabilise training. |
| **Precision / Recall** | Of the predicted "cat" images, how many really were cats (precision). Of the real cats, how many were found (recall). |
| **F1-score** | Harmonic mean of precision and recall. |
| **Macro average** | Compute a metric per class, then average with equal weight per class. |
| **mCE** | Mean Corruption Error, the standard summary metric for CIFAR-10-C / ImageNet-C. |
| **AugMix** | An augmentation method that mixes several augmented versions of an image to improve robustness. |
| **JSD loss** | Jensen-Shannon divergence consistency loss. In AugMix it encourages similar predictions across augmented versions of the same image. |
| **Mixed precision (AMP)** | Training with 16-bit numbers where safe, which is faster and uses less memory. |

---

## 21. References

1. Krizhevsky, A. (2009). *Learning Multiple Layers of Features from Tiny Images.* (CIFAR-10)
2. He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition.* CVPR. (ResNet)
3. Hendrycks, D., & Dietterich, T. (2019). *Benchmarking Neural Network Robustness to Common Corruptions and Perturbations.* ICLR. (CIFAR-10-C, mCE)
4. Hendrycks, D., Mu, N., Cubuk, E. D., Zoph, B., Gilmer, J., & Lakshminarayanan, B. (2020). *AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty.* ICLR.
5. Geirhos, R., et al. (2018). *Generalisation in Humans and Deep Neural Networks.* NeurIPS. (Models trained on specific noise types fail to generalise to others; relevant to RQ4.)
6. Recht, B., Roelofs, R., Schmidt, L., & Shankar, V. (2019). *Do ImageNet Classifiers Generalize to ImageNet?* ICML. (Distribution shift)
7. Schneider, S., et al. (2020). *Improving Robustness Against Common Corruptions by Covariate Shift Adaptation.* NeurIPS. (BatchNorm adaptation, a possible extension)

**Useful links**
- CIFAR-10: https://www.cs.toronto.edu/~kriz/cifar.html
- CIFAR-10-C (Zenodo): https://zenodo.org/records/2535967
- Robustness benchmark code: https://github.com/hendrycks/robustness
- AugMix code: https://github.com/google-research/augmix
- PyTorch tutorials: https://pytorch.org/tutorials/

---

### Next Step

Once this overview is understood and agreed, the build starts with **Phase 1 (Setup)**: create the repository skeleton from Section 13, set up the Colab notebook, and download both datasets to Google Drive.
