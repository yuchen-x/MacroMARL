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
        'scripts/ma_iac_rnn_V.py',
        'scripts/ma_cac_rnn_V.py',
        'scripts/ma_niacc_rnn_V.py',
        'scripts/ma_niacc_rnn_sV.py',
        'scripts/ma_iaicc_rnn_V.py',
        'scripts/ma_iaicc_rnn_sV.py',
        'scripts/ma_hddrqn.py',
        'scripts/ma_cen_condi_ddrqn.py',
        'scripts/ma_dec_cen_hddrqn_sep.py',
        'scripts/ma_dec_cen_hddrqn.py',
    ],

    license='MIT',
)
