# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import setuptools

setuptools.setup(
    python_requires=">3.7.0",
    name="simuleval",
    version="1.1.0",
    author="Xutai Ma",
    entry_points={
        "console_scripts": [
            "simuleval = simuleval.cli:main",
            "simuleval-scorer = simuleval.cli:score",
        ],
    },
    install_requires=[
        "pytest",
        "pytest-cov",
        "sacrebleu==1.5.1",
        "tornado",
        "soundfile",
        "pandas",
        "requests",
        "pytest-flake8",
        "textgrid",
        "tqdm",
        "pyyaml",
        "bitarray==2.6.0",
    ],
    package=setuptools.find_packages(
        exclude=[
            "examples",
            "examples.*",
            "docs",
            "docs.*",
        ]
    ),
)
