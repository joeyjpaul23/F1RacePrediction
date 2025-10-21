#!/usr/bin/env python3
"""
Setup script for F1 Race Predictor - Streamlit Application
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="f1-race-predictor",
    version="1.0.0",
    author="F1 Predictor Team",
    author_email="your.email@example.com",
    description="A machine learning-powered Streamlit application for predicting Formula 1 race results",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/F1Predictor",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "f1-predictor=streamlit_app:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yml", "*.yaml"],
    },
    keywords="f1, formula1, machine-learning, prediction, racing, sports-analytics, streamlit",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/F1Predictor/issues",
        "Source": "https://github.com/yourusername/F1Predictor",
        "Documentation": "https://github.com/yourusername/F1Predictor#readme",
    },
) 