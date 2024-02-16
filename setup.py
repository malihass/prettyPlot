import os
from setuptools import setup

here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, "prettyPlot", "version.py"), encoding="utf-8") as f:
    version = f.read()
version = version.split('=')[-1].strip().strip('"').strip("'")

setup(
    name='prettyPlot',
    version=version,
    description="Plotting tools for journal quality figures",
    url="https://github.com/malihass/prettyPlot",
    author="Malik Hassanaly",
    license="BSD 3-Clause",
    package_dir={"prettyPlot": "prettyPlot"},
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires='>=3.10',
    install_requires=["numpy","matplotlib>=3.7.3", "imageio"],
)
