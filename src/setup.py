# setup.py

from setuptools import setup, find_packages

setup(
    name="wikiart_clip",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch",
        "torchvision",
        "open_clip",
        "pandas",
        "tqdm",
        "Pillow",
        "numpy",
    ],
)