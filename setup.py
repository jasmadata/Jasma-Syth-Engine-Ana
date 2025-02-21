from setuptools import setup, find_packages

setup(
    name="data-synthesizer",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "pyyaml>=6.0.0",
    ],
    extras_require={
        "dev": [
            "black>=22.3.0",
            "isort>=5.10.1",
            "mypy>=0.950",
            "pylint>=2.14.0",
        ],
        "test": [
            "pytest>=7.1.0",
            "pytest-cov>=3.0.0",
        ],
    },
    python_requires=">=3.8",
    author="@jasma-core",
    description="jasma syth engine is a synthetic data generation framework that allows you to generate realistic and diverse data for testing, development, and training purposes.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: beta-version",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
    ],
) 