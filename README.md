
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

# proteovae <!-- omit from toc -->

This library implements a convenient set of modules for designing and implementing several different variational autoencoder frameworks. So far support is provided for the vanilla [VAE](https://arxiv.org/abs/1312.6114), [beta-VAE](https://openreview.net/forum?id=Sy2fzU9gl), and the here-presented guided VAE (GVAE). 

`proteovae` also provides a few different model trainers to facilitate the training process, although you can also use standard PyTorch or [Lightning](https://www.pytorchlightning.ai/index.html) as you please.  This package was developed as a tool to explore genomics data (hence *proteo[mics]*-vae), for a much more comprehensive suite of VAE implementations I would point you in the direction of [pythae](https://github.com/clementchadebec/benchmark_VAE/tree/main).

**News** 📢

Version 0.0.1 now on PyPI! ❤️🇮🇹🧑‍🔬

## Quick Access
- [Installation](#installation)
- [Defining Custom Architectures](#defining-custom-architectures)
- [Model Training](#model-training)
- [Tutorials](#tutorials)

# Installation 
To install the latest stable release of this library run the following using ``pip`` 
```bash
$ pip install proteovae
``` 


# Defining Custom Architectures
In addition to the models provided `proteovae.models.base` module you can also write your own encoder and decoder architectures for the VAE you're fitting.  

```python  
>>> from proteovae.models.base import Encoder, Guide, Decoder
>>> from proteovae.models import GuidedConfig, GuidedVAE
>>> import torch 
>>> from torch import nn 
...
>>> input_dim = 64
>>> latent_dim = 10
>>> guided_dim = 1
>>> n_classes = 2 # dummy 
...
>>> config = GuidedConfig(
...     input_dim = input_dim,
...     latent_dim = latent_dim, 
...     guided_dim = guided_dim
... )
...
>>> #using proteovae.models objects 
>>> enc = Encoder(
...         input_dim=input_dim, 
...         latent_dim=latent_dim, 
...         hidden_dims = [32,16,]
... )
>>> dec = Decoder(
...         output_dim = input_dim, 
...         latent_dim = latent_dim, 
...         hidden_dims = [16,32,]
... )
...
>>> gvae1 = GuidedVAE(
...         model_config = config,
...         encoder = enc,
...         decoder = dec, 
...         guide = Guide(dim_in = guided_dim, dim_out = n_classes)
)
...
>>> #or with generic torch objects 
>>> class CustomDecoder(nn.Module):
...     def __init__(self, **kwargs):
...         super().__init__(**kwargs)
...         self.fwd_block = nn.Sequential(
...         nn.Linear(latent_dim, 2*latent_dim),
...         nn.Tanh(),
...         nn.Linear(2*latent_dim, input_dim),
...     )
...     def forward(self, x):
...         return self.fwd_block(x)
...
>>> custom_dec = CustomDecoder()
>>> gvae2 = GuidedVAE(
...         model_config = config,
...         encoder = enc,
...         decoder = custom_dec, 
...         guide = Guide(dim_in = guided_dim, dim_out = n_classes)
```

# Model Training 
Two different `proteovae.trainers` Trainers (`BaseTrainer` and `ScheduledTrainer`) are provided to bundle up a lot of the annoying aspects of defining training loops in PyTorch. As shown below, their implementation is fairly straightfoward 

```python
>>> from proteovae.trainers import ScheduledTrainer 
>>> from torch import optim 
...
>>> #define any proteovae.models object and torch data loaders 
>>> model = # ... 
>>> train_loader = # ... 
>>> val_loader = # ... 
...
>>> #define optimizer and lr_scheduler 
>>> n_epochs = 5 
>>> optimizer = optim.Adam(model.parameters(), 
...                        lr=1e-03)
>>> scheduler = optim.lr_scheduler.LinearLR(optimizer, 
...                                         start_factor=1.0, 
...                                         end_factor=0.33, 
...                                         total_iters=n_epochs*len(train_loader))
>>> #trainer init 
>>> trainer = ScheduledTrainer(model, optimizer, scheduler)
...
>>> #train
>>> trainer.train(train_loader, n_epochs, val_data = val_data)

```

# Tutorials
- [mnist_proteovae.ipynb](https://github.com/nnethercott/proteovae/tree/main/examples/notebooks/) shows you how to implement your own architectures for traditional vae tasks<br>[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nnethercott/proteovae/blob/main/examples/notebooks/mnist_proteovae.ipynb) 
  
- [nonlinear_pca.ipynb](https://github.com/nnethercott/proteovae/tree/main/examples/notebooks/) motivates a generic use case for the GVAE framework<br>[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nnethercott/proteovae/blob/main/examples/notebooks/nonlinear_pca.ipynb) 
