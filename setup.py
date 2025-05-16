from setuptools import setup, find_packages

setup(
    name="automl_toolkit",
    version="1.0",
    packages=find_packages(),
    install_requires=[
        "scikit-learn",
        "pandas",
        "matplotlib",
        "imblearn",
        "xgboost",   
        "optuna",
        "numpy",
        "pickle",
        "json",
        "scipy"
    ],
)
