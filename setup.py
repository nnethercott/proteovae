import os
from setuptools import setup, find_packages


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

# python3 setup.py sdist bdist_wheel


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
    package_dir={"": "src"},
    packages=find_packages(where="src"),

    install_requires=[
        'numpy==1.22.4',
        'pydantic==1.10.7',
        'scikit_learn==1.2.2',
        'torch==2.0.0',
        'tqdm==4.65.0',
    ],
    extras_require={
        'tutorial': ['torchvision==0.15.1', 'jupyter', 'matplotlib==3.7.1'],
    }
)
