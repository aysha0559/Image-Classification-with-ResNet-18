"""Checks that src/constants.py matches CONTRACTS C3 and DECISIONS D6."""

from src import constants as C

C3_CORRUPTIONS = (
    "gaussian_noise shot_noise impulse_noise defocus_blur glass_blur motion_blur zoom_blur "
    "snow frost fog brightness contrast elastic_transform pixelate jpeg_compression"
).split()
C3_CLASSES = "airplane automobile bird cat deer dog frog horse ship truck".split()


def test_counts_and_exact_names():
    assert list(C.CORRUPTIONS) == C3_CORRUPTIONS
    assert len(C.CORRUPTIONS) == 15
    assert list(C.EXTRA_CORRUPTIONS) == ["speckle_noise", "gaussian_blur", "spatter", "saturate"]
    assert list(C.CLASS_NAMES) == C3_CLASSES
    assert C.NUM_CLASSES == 10


def test_every_corruption_in_exactly_one_family():
    for c in (*C.CORRUPTIONS, *C.EXTRA_CORRUPTIONS):
        assert sum(c in members for members in C.FAMILIES.values()) == 1, c
    assert set(C.FAMILIES) == {"noise", "blur", "weather", "digital", "extra"}
    assert set(C.FAMILIES["extra"]) == set(C.EXTRA_CORRUPTIONS)


def test_seen_unseen_partition():
    seen, unseen = set(C.SEEN_CORRUPTIONS), set(C.UNSEEN_CORRUPTIONS)
    assert seen == {"gaussian_noise", "defocus_blur", "contrast", "jpeg_compression"}
    assert not seen & unseen
    assert seen | unseen == set(C.CORRUPTIONS)


def test_analysis_groups_partition():
    groups = C.ANALYSIS_GROUPS
    assert set(groups["seen"]) == set(C.SEEN_CORRUPTIONS)
    assert set(groups["unseen_other_family"]) == {"snow", "frost", "fog"}
    assert set(groups["unseen_same_family"]) | set(groups["unseen_other_family"]) == set(
        C.UNSEEN_CORRUPTIONS
    )
    all_members = [c for g in groups.values() for c in g]
    assert sorted(all_members) == sorted(C.CORRUPTIONS)
