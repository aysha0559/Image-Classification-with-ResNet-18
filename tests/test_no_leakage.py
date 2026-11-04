"""Guard against CIFAR-10-C leaking into training code (RULES section 3)."""

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FORBIDDEN = ("cifar10c", "cifar-10-c")  # compared against lower-cased text
TRAINING_CODE = [
    REPO / "src" / "train.py",
    REPO / "src" / "data" / "transforms.py",
    REPO / "src" / "data" / "corruptions.py",
    *sorted((REPO / "src" / "models").rglob("*.py")),
]
# RULES section 3: the only modules allowed to import src.data.cifar10c (plus the Phase 11
# demo asset builder under app/, which is outside src/ and scripts/).
ALLOWED_IMPORTERS = {
    REPO / "src" / "evaluate.py",
    REPO / "scripts" / "make_sample_grid.py",
    REPO / "src" / "data" / "cifar10c.py",
}


def test_training_code_never_mentions_cifar10c():
    assert len(TRAINING_CODE) >= 6
    hits = []
    for path in TRAINING_CODE:
        text = path.read_text().lower()
        hits += [f"{path.relative_to(REPO)}: {w}" for w in FORBIDDEN if w in text]
    assert not hits, f"CIFAR-10-C referenced in training code: {hits}"


def _imports_cifar10c(path: Path) -> bool:
    """True if the module imports src.data.cifar10c (any import form)."""
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(a.name.endswith("cifar10c") for a in node.names):
                return True
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module.endswith("cifar10c") or any(a.name == "cifar10c" for a in node.names):
                return True
    return False


def test_only_allowed_modules_import_cifar10c():
    files = sorted((REPO / "src").rglob("*.py")) + sorted((REPO / "scripts").rglob("*.py"))
    offenders = [
        str(p.relative_to(REPO))
        for p in files
        if p not in ALLOWED_IMPORTERS and _imports_cifar10c(p)
    ]
    assert not offenders, (
        f"only evaluate.py and make_sample_grid.py may import cifar10c: {offenders}"
    )
