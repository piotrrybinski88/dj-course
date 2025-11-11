#!/usr/bin/env python3
"""
Setup script for Tokenizer CLI application.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="tokenizer-cli",
    version="1.0.0",
    author="Your Name",
    description="CLI application for BPE tokenizer training and encoding",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/tokenizer-cli",
    py_modules=["cli"],
    install_requires=[
        "tokenizers>=0.13.0",
        "click>=8.0.0",
    ],
    entry_points={
        "console_scripts": [
            "tokenizer-cli=cli:cli",
        ],
    },
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
