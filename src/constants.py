"""Fixed names used across the project (CONTRACTS C3, DECISIONS D6)."""

from __future__ import annotations

CLASS_NAMES: tuple[str, ...] = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)
NUM_CLASSES: int = len(CLASS_NAMES)

MODELS: tuple[str, ...] = ("simple_cnn", "resnet18")
MODEL_SHORT_NAMES: dict[str, str] = {"simple_cnn": "cnn", "resnet18": "resnet18"}
STRATEGIES: tuple[str, ...] = ("none", "standard", "augmix", "corruption_aware")
SEEDS: tuple[int, ...] = (0, 1, 2)

# The 15 standard CIFAR-10-C corruptions, in CIFAR-10-C file-stem form.
CORRUPTIONS: tuple[str, ...] = (
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "defocus_blur",
    "glass_blur",
    "motion_blur",
    "zoom_blur",
    "snow",
    "frost",
    "fog",
    "brightness",
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression",
)
# The 4 extra corruptions (not evaluated by default, never in mCE; DECISIONS D7).
EXTRA_CORRUPTIONS: tuple[str, ...] = ("speckle_noise", "gaussian_blur", "spatter", "saturate")
SEVERITIES: tuple[int, ...] = (1, 2, 3, 4, 5)

FAMILIES: dict[str, tuple[str, ...]] = {
    "noise": ("gaussian_noise", "shot_noise", "impulse_noise"),
    "blur": ("defocus_blur", "glass_blur", "motion_blur", "zoom_blur"),
    "weather": ("snow", "frost", "fog"),
    "digital": ("brightness", "contrast", "elastic_transform", "pixelate", "jpeg_compression"),
    "extra": EXTRA_CORRUPTIONS,
}
CORRUPTION_TO_FAMILY: dict[str, str] = {c: fam for fam, cs in FAMILIES.items() for c in cs}

# DECISIONS D6: seen corruptions (S3 trains on our own versions) and unseen ones.
SEEN_CORRUPTIONS: tuple[str, ...] = (
    "gaussian_noise",
    "defocus_blur",
    "contrast",
    "jpeg_compression",
)
UNSEEN_CORRUPTIONS: tuple[str, ...] = (
    "shot_noise",
    "impulse_noise",
    "glass_blur",
    "motion_blur",
    "zoom_blur",
    "snow",
    "frost",
    "fog",
    "brightness",
    "elastic_transform",
    "pixelate",
)

# DECISIONS D6: the three analysis groups for RQ4.
ANALYSIS_GROUPS: dict[str, tuple[str, ...]] = {
    "seen": SEEN_CORRUPTIONS,
    "unseen_same_family": (
        "shot_noise",
        "impulse_noise",
        "glass_blur",
        "motion_blur",
        "zoom_blur",
        "brightness",
        "elastic_transform",
        "pixelate",
    ),
    "unseen_other_family": ("snow", "frost", "fog"),
}
