import os
from setuptools import setup, find_packages


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="proteovae",
    version="0.0.1",
    author="Nate Nethercott",
    author_email="natenethercott@gmail.com",
    description=("package for implementing guided variational autoencoders"),
    license="MIT",
    # keywords = "example documentation tutorial",
    url="https://github.com/nnethercott/proteovae",
    long_description=read('README.md'),
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
    ],
    package_dir={"": "proteovae"},
    packages=find_packages(where="proteovae"),

    install_requires=[
        'numpy==1.22.4',
        'pydantic>=1.10.0',
        'scikit_learn>=1.2.0',
        'torch==2.0.0',
        'tqdm==4.65.0',
    ],

)
