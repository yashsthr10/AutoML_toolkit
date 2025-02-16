from setuptools import setup, find_packages

setup(
    name="automl_toolkit",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "scikit-learn",
        "pandas",
        "matplotlib",
        "seaborn"
    ],
)
