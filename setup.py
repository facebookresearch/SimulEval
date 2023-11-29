# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import find_packages, setup

with open("README.md", "r") as readme_file:
    long_description = readme_file.read()

setup(
    python_requires=">3.7.0",
    name="simuleval",
    version="1.1.3",
    author="Xutai Ma",
    description="SimulEval: A Flexible Toolkit for Automated Machine Translation Evaluation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    homepage="https://github.com/facebookresearch/SimulEval.git",
    documentation="https://simuleval.readthedocs.io/en/v1.1.0/quick_start.html",
    license="LICENSE",
    entry_points={
        "console_scripts": [
            "simuleval = simuleval.cli:main",
        ],
    },
    install_requires=[
        "pytest",
        "pytest-cov",
        "sacrebleu>=2.3.1",
        "tornado",
        "soundfile",
        "pandas",
        "requests",
        "pytest-flake8",
        "textgrid",
        "tqdm==4.64.1",
        "pyyaml",
        "bitarray==2.6.0",
        "yt-dlp",
        "pydub",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
    ],
    keywords=[
        "SimulEval",
        "Machine Translation",
        "Evaluation",
        "Metrics",
        "BLEU",
        "TER",
        "METEOR",
        "chrF",
        "RIBES",
        "WMD",
        "Embedding Average",
        "Embedding Extrema",
        "Embedding Greedy",
        "Embedding Average",
        "SimulEval",
        "SimulEval_Testing_Package_1",
        "facebookresearch",
        "facebook",
        "Meta-Evaluation",
    ],
    packages=find_packages(
        exclude=[
            "examples",
            "examples.*",
            "docs",
            "docs.*",
        ]
    ),
    setup_requires=["setuptools_scm"],
    use_scm_version=True,
)
