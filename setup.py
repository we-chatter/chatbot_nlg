#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author:cch
# datetime:2020/12/7 下午5:09

import os
from setuptools import setup, find_packages, Extension
import sys
import os

sys.path.append(os.getcwd())

__author__ = [
    '"chenchanghao"<chenchanghao@navinfo.com>'
]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# def readme():
#     # with codecs.open(os.path.join(SCRIPT_DIR, "README.md"),"r","utf-8") as file:
#     #     return file.read()
#     return open(os.path.join(SCRIPT_DIR, "README.md"),
#                 'r+', encoding="utf-8").read()


# Maybe here use >= or ~= is more suitable.
with open("requirements.txt", "r") as file:
    install_requires = file.readlines()

setup(name='gpt_service',
      version='1.0.0',
      description="gpt2 service",
      # long_description=readme(),

      keywords="gpt2 service",
      author="Simulation Team",
      license="MIT",
      packages=find_packages(),
      # platforms='python 3.5',
      url="None",
      python_requires='~=3.6',
      entry_points={
          'console_scripts':
          ['gpt2_server=chatbot.server.server:main']
      },
      install_requires=install_requires,
      zip_safe=False
      )


