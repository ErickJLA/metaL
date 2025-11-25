from setuptools import setup, find_packages

setup(
    name="ecometa",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "matplotlib",
        "seaborn",
        "statsmodels",
        "ipywidgets",
        "patsy"
    ],
    author="Your Name",
    description="A Python library for ecological meta-analysis",
)