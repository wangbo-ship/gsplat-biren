#!/usr/bin/env /usr/bin/python3
"""Download and extract common 3DGS datasets or an arbitrary archive URL."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

from gsplat._logging import get_logger, log_error, log_info


LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class DatasetSource:
    name: str
    urls: tuple[str, ...]
    directory: str


DATASETS: dict[str, DatasetSource] = {
    "mipnerf360": DatasetSource(
        name="mipnerf360",
        urls=("http://storage.googleapis.com/gresearch/refraw360/360_v2.zip",),
        directory="360_v2",
    ),
    "mipnerf360-extra": DatasetSource(
        name="mipnerf360-extra",
        urls=(
            "https://storage.googleapis.com/gresearch/refraw360/360_extra_scenes.zip",
        ),
        directory="360_v2",
    ),
    "bilarf": DatasetSource(
        name="bilarf",
        urls=(
            "https://huggingface.co/datasets/Yuehao/bilarf_data/resolve/main/bilarf_data.zip",
        ),
        directory="bilarf",
    ),
    "zipnerf": DatasetSource(
        name="zipnerf",
        urls=(
            "https://storage.googleapis.com/gresearch/refraw360/zipnerf/berlin.zip",
            "https://storage.googleapis.com/gresearch/refraw360/zipnerf/london.zip",
            "https://storage.googleapis.com/gresearch/refraw360/zipnerf/nyc.zip",
            "https://storage.googleapis.com/gresearch/refraw360/zipnerf/alameda.zip",
        ),
        directory="zipnerf",
    ),
    "zipnerf-undistorted": DatasetSource(
        name="zipnerf-undistorted",
        urls=(
            "https://storage.googleapis.com/gresearch/refraw360/zipnerf-undistorted/berlin.zip",
            "https://storage.googleapis.com/gresearch/refraw360/zipnerf-undistorted/london.zip",
            "https://storage.googleapis.com/gresearch/refraw360/zipnerf-undistorted/nyc.zip",
            "https://storage.googleapis.com/gresearch/refraw360/zipnerf-undistorted/alameda.zip",
        ),
        directory="zipnerf_undistorted",
    ),
}


def _parse_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def _add_bool_arg(
    parser: argparse.ArgumentParser,
    name: str,
    *,
    default: bool,
    help: str,
) -> None:
    parser.add_argument(
        name,
        type=_parse_bool,
        default=default,
        metavar="true|false",
        help=f"{help} Defaults to {str(default).lower()}.",
    )


def _archive_name(url: str) -> str:
    name = Path(urlparse(url).path).name
    if not name:
        raise ValueError(f"cannot infer archive name from URL: {url}")
    return name


def _download(url: str, path: Path, *, skip_existing: bool) -> None:
    if skip_existing and path.is_file():
        log_info(LOGGER, f"Using existing archive: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    command = ["curl", "-L", "--fail", "-o", str(path), url]
    subprocess.run(command, check=True)


def _extract(archive: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    suffixes = "".join(archive.suffixes[-2:])
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(output_dir)
    elif archive.suffix in {".tgz", ".gz", ".tar"} or suffixes in {".tar.gz", ".tar.xz", ".tar.bz2"}:
        with tarfile.open(archive) as handle:
            handle.extractall(output_dir)
    else:
        raise ValueError(f"unsupported archive format: {archive}")


def _iter_sources(args: argparse.Namespace) -> Iterable[tuple[str, tuple[str, ...]]]:
    if args.url:
        directory = args.name or Path(_archive_name(args.url)).stem
        yield directory, (args.url,)
        return
    source = DATASETS[args.dataset]
    yield source.directory, source.urls


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset",
        nargs="?",
        choices=sorted(DATASETS),
        default="mipnerf360",
        help="Known dataset to download when --url is not provided.",
    )
    parser.add_argument(
        "--url",
        help="Download an arbitrary archive URL instead of a known dataset.",
    )
    parser.add_argument(
        "--name",
        help="Output directory name for --url downloads. Defaults to archive stem.",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path(".agents/dataset"),
        help="Directory where archives are extracted.",
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        help="Temporary archive directory. Defaults to .agents/temporary/downloads.",
    )
    _add_bool_arg(
        parser,
        "--keep-archive",
        default=False,
        help="Keep downloaded archives after successful extraction.",
    )
    _add_bool_arg(
        parser,
        "--skip-existing",
        default=False,
        help="Reuse already downloaded archives in the download directory.",
    )
    _add_bool_arg(
        parser,
        "--list",
        default=False,
        help="List known dataset names and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list:
        for name in sorted(DATASETS):
            source = DATASETS[name]
            log_info(LOGGER, f"{name}\t{source.directory}\t{len(source.urls)} archive(s)")
        return

    download_dir = args.download_dir or Path(".agents/temporary/downloads")
    curl = shutil.which("curl")
    if curl is None:
        raise RuntimeError("curl is required for dataset downloads")

    for directory, urls in _iter_sources(args):
        extract_dir = args.save_dir / directory
        for url in urls:
            archive = download_dir / _archive_name(url)
            log_info(LOGGER, f"Downloading {url}")
            _download(url, archive, skip_existing=args.skip_existing)
            log_info(LOGGER, f"Extracting {archive} -> {extract_dir}")
            _extract(archive, extract_dir)
            if not args.keep_archive:
                archive.unlink(missing_ok=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - CLI error path
        log_error(LOGGER, f"error: {exc}")
        raise SystemExit(1) from exc
