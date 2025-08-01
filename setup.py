"""Build the package."""

import os
import re
from io import open

from setuptools import find_packages, setup

here = os.path.abspath(os.path.dirname(__file__))
PACKAGE_NAME = "gem"

with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    readme = f.read()


def _read_version():
    VERSION_FILE = f"{PACKAGE_NAME}/__about__.py"
    ver_str_line = open(VERSION_FILE, "rt").read()
    version_re = r"^__version__ = ['\"]([^'\"]*)['\"]"
    mo = re.search(version_re, ver_str_line, re.M)
    if mo:
        version = mo.group(1)
    else:
        raise RuntimeError(f"Unable to find version string in {VERSION_FILE}.")
    return version


setup(
    name=PACKAGE_NAME,
    version=_read_version(),
    author="Zichen Liu, Anya Sims, Keyu Duan, Changyu Chen",
    author_email="lkevinzc@gmail.com, anya.sims@stats.ox.ac.uk, k.duan@u.nus.edu, abelco.cy@gmail.com",
    description="A Gym for Generalist LLMs.",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/axon-rl/gem",
    license="Apache-2.0",
    packages=find_packages(exclude=["examples*", "test*"]),
    include_package_data=True,
    python_requires=">=3.10, <=3.12",
    setup_requires=["setuptools_scm>=7.0"],
    zip_safe=False,
)
