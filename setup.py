#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="slm",
    version="0.0.1",
    description="Structure Language Models",
    author="",
    author_email="",
    url="https://github.com/lujiarui/esmdiff",
    install_requires=["lightning", "hydra-core"],
    packages=find_packages(),
    # use this to customize global commands available in the terminal after installing the package
    entry_points={
        "console_scripts": [
            "trainit = slm.train:main",
        ]
    },
)
