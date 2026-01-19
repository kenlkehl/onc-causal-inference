"""Setup configuration for Propensity Score Matching package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

setup(
    name="psm-clinical-text",
    version="0.2.0",
    author="Ken Kehl",
    author_email="kenneth_kehl@dfci.harvard.edu",
    description="Propensity score matching for causal inference from clinical text",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kenlkehl/causal-dragonnet-text",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "sentence-transformers>=2.2.0",
        "pandas>=1.5.0",
        "numpy>=1.23.0",
        "scikit-learn>=1.2.0",
        "scipy>=1.10.0",
        "tqdm>=4.65.0",
        "pyarrow",
        "joblib>=1.2.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "psm=cdt.cli:main",  # New propensity matching CLI
            "cdt=cdt.cli:main",  # Legacy alias for backward compatibility
        ],
    },
)
