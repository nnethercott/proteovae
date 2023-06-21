
<p align="center">
    <a>
	    <img src='https://img.shields.io/badge/python-3.10-blueviolet' alt='Python' />
	</a>
	<a href='https://pythae.readthedocs.io/en/latest/?badge=latest'>
    	<img src='https://readthedocs.org/projects/pythae/badge/?version=latest' alt='Documentation Status' />
	</a>
	<a href='https://opensource.org/license/mit/'>
	    <img src='https://img.shields.io/github/license/nnethercott/proteovae?color=blue' />
	</a><br>
	</a>
</p>

</p>
<p align="center">
  <a href="https://proteovae.readthedocs.io/en">Documentation</a>
</p>

# proteovae 

This library implements a convenient set of modules for designing and implementing several different variational autoencoder frameworks. So far support is provided for [VAEs](https://arxiv.org/abs/1312.6114), [$\beta$-VAE](https://openreview.net/forum?id=Sy2fzU9gl), and the here-presented "guided" VAE. `proteovae` also provides a few different model trainers to facilitate the training process (although you can use vanilla PyTorch or [Lightning](https://www.pytorchlightning.ai/index.html) as you please).  For a much more comprehensive suite of VAE implementations I would point you in the direction of [pythae](https://github.com/clementchadebec/benchmark_VAE/tree/main).

**News** 📢

Version 0.0.1 now on PyPI! ❤️🇮🇹🧑‍🔬

## Quick Access
- [proteovae](#proteovae)
  - [Quick Access](#quick-access)
- [Installation](#installation)
  - [Defining Custom Architectures](#defining-custom-architectures)
  - [Model Training](#model-training)

# Installation 
To install the latest stable release of this library run the following using ``pip`` 
```bash
$ pip install proteovae
``` 


## Defining Custom Architectures
In addition to the models provided `proteovae.models.base` module you can also write your own encoder and decoder architectures for the VAE you're fitting! 

## Model Training 
To train a model we 