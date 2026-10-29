# SPDX-License-Identifier: Apache-2.0

"""SUPA-oriented package metadata for the gsplat-biren fork."""

from __future__ import annotations

from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).resolve().parent

version_scope: dict[str, str] = {}
exec((ROOT / "gsplat" / "version.py").read_text(encoding="utf-8"), version_scope)


def read_readme() -> str:
    readme = ROOT / "README.md"
    if readme.is_file():
        return readme.read_text(encoding="utf-8")
    return "Biren SUPA adaptation of gsplat."


setup(
    name="gsplat-biren",
    version=version_scope["__version__"],
    description="Biren SUPA adaptation of gsplat for basic 3D Gaussian Splatting.",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    keywords="gaussian splatting 3dgs biren supa brpytorch",
    python_requires=">=3.10",
    packages=find_packages(),
    package_data={
        "gsplat.supa": ["*.su", "*.cpp"],
    },
    include_package_data=True,
    install_requires=[
        "numpy==1.26.4",
        "ninja==1.13.0",
        "pycolmap==4.1.0",
        "typing_extensions==4.15.0",
        "Pillow==12.2.0",
    ],
    extras_require={
        "dev": [
            "pytest==9.0.3",
        ],
        "trainer": [
            "imageio==2.37.3",
        ],
    },
    project_urls={
        "Upstream gsplat": "https://github.com/nerfstudio-project/gsplat",
    },
)
