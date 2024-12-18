#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat May 11 22:38:11 2019

@author: kevinrussell
"""

from setuptools import setup
import setuptools

__version__ = "0.1.0"


def parse_requirements(filename):
    """load requirements from a pip requirements file"""
    lineiter = (line.strip() for line in open(filename))
    return [line for line in lineiter if line and not line.startswith("#")]


install_reqs = parse_requirements("requirements.txt")

setup(
    name="engforge",
    version=__version__,
    description="The Engineer's Framework",
    url="https://github.com/SoundsSerious/engforge",
    author="Kevin russell",
    # packages=["ottermatics"],
    packages=setuptools.find_packages(),
    author_email="kevin@ottermatics.com",
    license="MIT",
    entry_points={
        "console_scripts": [
            # command = package.module:function
            # "condaenvset=engforge.common:main_cli",
            # "ollymakes=engforge.locations:main_cli",
            # "engforgedrive=engforge.gdocs:main_cli",
            # TODO: inspect folders / list / display components + solver options
            # TODO: test adhoc modules with
            # TODO: run analysis adhoc on components with cli options
        ]
    },
    install_requires=install_reqs,
    include_package_data=True,
    zip_safe=False,
)
