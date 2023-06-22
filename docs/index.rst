.. proteovae documentation master file, created by
   sphinx-quickstart on Wed Jun 21 14:11:41 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to proteovae's documentation!
=====================================
This package aims at providing a consistent set of tools for implementing multi-task objective variational autoencoders.  
It was originally developed for uses in genomics data (hence the name *proteo[mics]*-vae).

**News** 📢

The first version of this package is now live ❤️🇮🇹🧑‍🔬! in subsequent realeases options for specifying different losses in the reconstruction 
objective will be available, as well as a cleaner embedding method for VAE objects. 

.. toctree::
   :maxdepth: 1
   :caption: proteovae:

   proteovae.models
   proteovae.trainers
   proteovae.disentanglement

Setup
~~~~~~~~~~~~~

To install the latest stable release of this library run the following using ``pip``

.. code-block:: bash

   $ pip install proteovae

For local installations try git cloning and installing the project repo 

.. code-block:: bash

   $ git clone git@github.com:nnethercott/proteovae.git
   $ pip install -e .


If you clone the proteovae repository you will access to the following:

- ``docs``: The folder in which the documentation can be retrieved.
- ``examples``: A list of ``ipynb`` tutorials and script describing some use cases for proteovae.
- ``src/proteovae``: The main library which can be installed with ``pip``.



