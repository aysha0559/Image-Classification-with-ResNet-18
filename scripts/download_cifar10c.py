"""Download, verify and extract CIFAR-10-C (python -m scripts.download_cifar10c).

Steps:
1. Read the archive's name, size and checksum from Zenodo's record metadata (record 2535967).
2. Download `CIFAR-10-C.tar` into `DATA_DIR` (resumable, progress bar). Skipped if the
   archive already there verifies, or if the extracted files already match the checksums JSON.
3. Verify the archive checksum (one fresh re-download if it fails; a second failure stops).
4. Extract to `DATA_DIR/CIFAR-10-C/`, check 19 corruption files + `labels.npy` with the
   expected shapes and dtypes, and sha256 every `.npy`.
5. Write `splits/cifar10c_checksums.json`, or, if it already exists, check the files match it.

`--verify-only` skips the network and only re-checks the extracted files against the JSON
(used on Colab after copying the folder from Drive).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.constants import CORRUPTIONS, EXTRA_CORRUPTIONS
from src.utils import REPO_ROOT, env_path

ZENODO_RECORD = "2535967"
ZENODO_API = f"https://zenodo.org/api/records/{ZENODO_RECORD}"
ARCHIVE_NAME = "CIFAR-10-C.tar"
FOLDER_NAME = "CIFAR-10-C"
CHECKSUMS_PATH = REPO_ROOT / "splits" / "cifar10c_checksums.json"
EXPECTED_FILES: tuple[str, ...] = tuple(f"{c}.npy" for c in CORRUPTIONS + EXTRA_CORRUPTIONS) + (
    "labels.npy",
)
N_ROWS = 50_000
CHUNK = 1 << 20
USER_AGENT = "cifar10-robustness-study/0.1 (dataset download script)"


def fetch_archive_metadata() -> dict:
    """Return {"name", "size", "checksum_type", "checksum", "url"} for the archive from Zenodo."""
    req = urllib.request.Request(ZENODO_API, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        record = json.load(resp)
    for f in record["files"]:
        if f["key"] == ARCHIVE_NAME:
            algo, value = f["checksum"].split(":", 1)
            return {
                "name": f["key"],
                "size": int(f["size"]),
                "checksum_type": algo,
                "checksum": value,
                "url": f["links"]["self"],
            }
    raise RuntimeError(f"{ARCHIVE_NAME} not listed in Zenodo record {ZENODO_RECORD}")


def file_hash(path: Path, algo: str, desc: str | None = None) -> str:
    """Return the hex digest of `path` with hashlib algorithm `algo`, with a progress bar."""
    h = hashlib.new(algo)
    size = path.stat().st_size
    with (
        open(path, "rb") as f,
        tqdm(
            total=size, unit="B", unit_scale=True, desc=desc or f"{algo} {path.name}", leave=False
        ) as bar,
    ):
        while chunk := f.read(CHUNK):
            h.update(chunk)
            bar.update(len(chunk))
    return h.hexdigest()


def archive_verifies(path: Path, meta: dict) -> bool:
    """True if `path` exists with Zenodo's size and checksum. Prints both checksums."""
    if not path.is_file() or path.stat().st_size != meta["size"]:
        return False
    got = file_hash(path, meta["checksum_type"])
    print(f"archive {meta['checksum_type']}: zenodo {meta['checksum']}  local {got}")
    return got == meta["checksum"]


def download(url: str, dest: Path, total: int) -> None:
    """Download `url` to `dest`, resuming from `dest.part` if a partial file exists."""
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    if have > total:
        part.unlink()
        have = 0
    if have < total:
        headers = {"User-Agent": USER_AGENT}
        if have:
            headers["Range"] = f"bytes={have}-"
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=60) as resp:
            if have and resp.status != 206:  # server ignored the range: start over
                have = 0
            mode = "ab" if have else "wb"
            with (
                open(part, mode) as f,
                tqdm(total=total, initial=have, unit="B", unit_scale=True, desc=dest.name) as bar,
            ):
                while chunk := resp.read(CHUNK):
                    f.write(chunk)
                    bar.update(len(chunk))
    if part.stat().st_size != total:
        raise RuntimeError(f"{part}: got {part.stat().st_size} bytes, expected {total}")
    part.replace(dest)


def extract(archive: Path, data_dir: Path) -> None:
    """Extract the tar archive into `data_dir` (safe 'data' filter when available)."""
    kwargs = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
    with tarfile.open(archive) as tar:
        names = tar.getnames()
        bad = [n for n in names if not (n == FOLDER_NAME or n.startswith(FOLDER_NAME + "/"))]
        if bad:
            raise RuntimeError(f"unexpected paths in {archive.name}: {bad[:5]}")
        print(f"extracting {len(names)} entries to {data_dir / FOLDER_NAME}")
        tar.extractall(data_dir, **kwargs)


def check_structure(folder: Path) -> None:
    """Raise unless `folder` holds exactly the 20 expected `.npy` files with the right shapes."""
    present = sorted(p.name for p in folder.glob("*.npy"))
    missing = sorted(set(EXPECTED_FILES) - set(present))
    extra = sorted(set(present) - set(EXPECTED_FILES))
    if missing or extra:
        raise RuntimeError(f"{folder}: missing {missing}, unexpected {extra}")
    for name in EXPECTED_FILES:
        arr = np.load(folder / name, mmap_mode="r")
        want = (N_ROWS,) if name == "labels.npy" else (N_ROWS, 32, 32, 3)
        if arr.shape != want or (name != "labels.npy" and arr.dtype != np.uint8):
            raise RuntimeError(f"{folder / name}: got {arr.dtype} {arr.shape}, expected {want}")
    print(f"structure ok: {len(EXPECTED_FILES)} files, corruptions uint8 {(N_ROWS, 32, 32, 3)}")


def hash_folder(folder: Path) -> dict[str, str]:
    """Return {file name: sha256} for the expected `.npy` files."""
    return {name: file_hash(folder / name, "sha256") for name in EXPECTED_FILES}


def compare_hashes(got: dict[str, str], want: dict[str, str]) -> None:
    """Raise if the two {name: sha256} maps differ."""
    bad = sorted(n for n in set(got) | set(want) if got.get(n) != want.get(n))
    if bad:
        raise RuntimeError(f"sha256 mismatch against {CHECKSUMS_PATH.name}: {bad}")
    print(f"all {len(got)} files match {CHECKSUMS_PATH.relative_to(REPO_ROOT)}")


def load_checksums() -> dict | None:
    """Return the committed checksums JSON, or None if it does not exist yet."""
    if not CHECKSUMS_PATH.is_file():
        return None
    return json.loads(CHECKSUMS_PATH.read_text())


def verify_only(data_dir: Path) -> None:
    """Check the extracted files against the committed JSON (no network)."""
    saved = load_checksums()
    if saved is None:
        raise RuntimeError(f"{CHECKSUMS_PATH} not found; run without --verify-only first")
    folder = data_dir / FOLDER_NAME
    check_structure(folder)
    compare_hashes(hash_folder(folder), saved["files"])


def get_verified_archive(data_dir: Path, meta: dict) -> Path:
    """Download (if needed) and verify the archive; one fresh retry on a checksum failure."""
    archive = data_dir / ARCHIVE_NAME
    for attempt in (1, 2):
        if archive_verifies(archive, meta):
            print("archive checksum matches Zenodo")
            return archive
        if archive.exists():
            print(f"archive did not verify (attempt {attempt}); removing it")
            archive.unlink()
            archive.with_name(archive.name + ".part").unlink(missing_ok=True)
            if attempt == 2:
                break
        download(meta["url"], archive, meta["size"])
    raise RuntimeError("archive checksum failed twice; stop and check the Zenodo record")


def full_run(data_dir: Path) -> None:
    """Download, verify, extract, hash, and write or check the checksums JSON."""
    data_dir.mkdir(parents=True, exist_ok=True)
    saved = load_checksums()
    folder = data_dir / FOLDER_NAME
    if saved is not None and folder.is_dir():
        try:
            verify_only(data_dir)
            print("extracted files already verified; nothing to download")
            return
        except RuntimeError as err:
            print(f"existing folder does not verify ({err}); re-downloading")
    meta = fetch_archive_metadata()
    print(f"zenodo: {meta['name']} {meta['size']} bytes {meta['checksum_type']}:{meta['checksum']}")
    if saved is not None and {k: saved["archive"][k] for k in ("name", "size", "checksum")} != {
        k: meta[k] for k in ("name", "size", "checksum")
    }:
        raise RuntimeError(f"Zenodo metadata differs from {CHECKSUMS_PATH.name}: {meta}")
    archive = get_verified_archive(data_dir, meta)
    extract(archive, data_dir)
    check_structure(folder)
    hashes = hash_folder(folder)
    if saved is not None:
        compare_hashes(hashes, saved["files"])
        return
    out = {
        "source": f"zenodo {ZENODO_RECORD}",
        "archive": {k: meta[k] for k in ("name", "size", "checksum_type", "checksum")},
        "files": hashes,
    }
    CHECKSUMS_PATH.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {CHECKSUMS_PATH.relative_to(REPO_ROOT)} ({len(hashes)} file hashes)")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default=None, help="default: $DATA_DIR or ./data")
    parser.add_argument(
        "--verify-only", action="store_true", help="only check extracted files against the JSON"
    )
    args = parser.parse_args(argv)
    data_dir = Path(args.data_dir) if args.data_dir else env_path("DATA_DIR")
    try:
        verify_only(data_dir) if args.verify_only else full_run(data_dir)
    except (RuntimeError, OSError) as err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
