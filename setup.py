import os
from setuptools import setup

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, "requirements.txt")) as f:
    install_requires = f.readlines()

setup(
    name='paperPlot',
    version='0.0.1',
    description="Plotting tools for journal quality figures",
    url="https://github.com/malihass/paperPlot",
    license="BSD 3-Clause",
    package_dir={"paperPlot": "paperPlot"},
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires='>=3.10',
    install_requires=install_requires,
)
