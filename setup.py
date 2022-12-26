#!/usr/bin/env python
# encoding: utf-8
from setuptools import setup

setup(
    name='macro_marl',
    version='0.0.1',
    description='macro_marl - Macro-action-based multi-agent reinforcement learning',
    packages=['macro_marl'],
    package_dir={'': 'src'},

    # TODO setup dependencies correctly!

    scripts=[
        'scripts/value_based_main.py',
        'scripts/pg_based_main.py',
    ],

    license='MIT',
)
