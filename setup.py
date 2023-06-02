#! /usr/bin/env python
# -*- coding: utf-8 -*-
try:
    from setuptools import setup,find_packages
except ImportError:
    from distutils.core import setup
import setuptools

setup(
    name='wiseRL',  # 包的名字
    author='李科',  # 作者
    version='0.1.0',  # 版本号
    license='MIT',

    description='wiseRL 一个基于ray的分布式是强化学习框架，方便用户简单将强化学习应用到分布式上',
    author_email = 'like@wiseworker.com.cn',
    url='https://github.com/wiseworker/wiseRL',
    # 依赖包
    install_requires=[
        'ray',
        "gym",
        'gym[box2d]',
        'envpool'
    ],
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)