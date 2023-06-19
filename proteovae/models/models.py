import torch
import torch.distributions as dist
from torch import nn
import torch.nn.functional as F
from .utils import ModelOutput

# ADD DOCSTRINGS HERE


class BaseVAE(nn.Module):
    def __init__(self, config, encoder, decoder):
        super().__init__()
        self.model_config = config
        self.input_dim = self.model_config.input_dim
        self.latent_dim = self.model_config.latent_dim
        self.device = self.model_config.device

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        raise NotImplementedError

    def loss_function(self, x):
        raise NotImplementedError

    def val_function(self, x):
        raise NotImplementedError


class GuidedVAE(BaseVAE):
    def __init__(self, guide, **kwargs):
        super().__init__(**kwargs)

        # Tuning ELBO
        self.beta = self.model_config.beta
        self.eta = self.model_config.eta
        self.gamma = self.model_config.gamma

        self.elbo_scheduler = self.model_config.elbo_scheduler

        # Guide
        self.guide = guide
        self.guided_dim = self.model_config.guided_dim

        # Latent prior
        self.pz = dist.normal.Normal(torch.zeros(1, self.latent_dim, device=self.device),
                                     torch.ones(self.latent_dim, device=self.device))

    def forward(self, x):
        encoded = self.encoder(x)
        mu, log_var = encoded['cont']

        # sampling
        z = mu + torch.exp(0.5*log_var) * \
            (torch.zeros_like(mu, device=self.device).normal_())
        decoded = self.decoder(z)

        # posterior
        qz_x = dist.normal.Normal(mu, torch.exp(0.5*log_var))

        return qz_x, decoded

    def loss_function(self, data):
        X, Y = data

        # Model Feedforward
        qz_x, decoded = self.forward(X)

        # Reconstruction
        # recon = F.mse_loss(decoded, X, reduction='sum') #don't think this works well for non-image data
        recon = torch.square(decoded-X).sum(dim=1).mean()

        # KL-Divergence
        kl = dist.kl.kl_divergence(qz_x, self.pz).sum(-1).mean()

        mu = qz_x.loc.clone()
        # mu = qz_x.rsample() #stochastic
        var = qz_x.scale.clone()**2

        # DIP-VAE I-II
        varEz_x = torch.cov(mu.t())
        Evarz_x = torch.diag(var.mean(dim=0))
        cov_z = varEz_x + Evarz_x

        # diag
        dipvae_i = torch.square(torch.ones(
            self.latent_dim, device=self.device) - torch.diag(cov_z)).sum()
        # off-diag
        dipvae_ii = torch.square(cov_z.flatten()[1:].view(
            self.latent_dim-1, self.latent_dim+1)[:, :-1].flatten()).sum()

        dipvae = 10*dipvae_i + dipvae_ii

        # Guided
        g = mu[:, -self.guided_dim:]
        guided_logits = self.guide(g)
        guided_preds = guided_logits.argmax(dim=1)
        guided = F.cross_entropy(guided_logits, Y.type(torch.int64))

        # Final loss
        loss = recon + self.beta*kl + self.eta*guided + self.gamma*(dipvae)
        acc = (guided_preds == Y).sum().float()/(guided_preds.size(0))

        losses = ModelOutput(
            loss=loss,
            recon=recon,
            kl=kl,
            dipvae=dipvae,
            acc=acc
        )

        return losses

    def val_function(self, data):
        """
        to be applied over validation set during training. I'm interested in seeing validation accuracy
        of the guide branch 
        """
        X, Y = data

        qz_x, _ = self.forward(X)
        mu = qz_x.loc.clone()

        g = mu[:, -self.guided_dim:]
        guided_logits = self.guide(g)
        guided_preds = guided_logits.argmax(dim=1)

        val_acc = (guided_preds == Y).sum().float()/guided_preds.size(0)

        return ModelOutput(val_acc=val_acc)

    def _elbo_scheduler_update(self, e):
        self.beta = self.model_config.beta*self.elbo_scheduler['beta'](e)
        self.eta = self.model_config.eta*self.elbo_scheduler['eta'](e)
        self.gamma = self.model_config.gamma*self.elbo_scheduler['gamma'](e)


class JointVAE(GuidedVAE):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)  # delegate

        self.disc_dim = self.model_config.disc_dim
        assert (self.guided_dim == self.disc_dim,
                f'ERROR: assuming guided dims == discrete dim necessarily')

        # discrete prior
        self.pc = dist.categorical.Categorical(
            probs=(1/self.disc_dim)*torch.ones(self.disc_dim, device=self.device))

    def forward(self, x):
        enc = self.encoder(x)
        mu, log_var = enc['cont']
        alpha_logits = enc['disc']

        # sampling
        z_cont = mu + torch.exp(0.5*log_var) * \
            (torch.zeros(mu.shape, device=self.device).normal_())
        z_disc = self.sample_concrete(alpha_logits, tau=0.1)
        z = torch.cat((z_cont, z_disc), dim=1)

        decoded = self.decoder(z)

        # posteriors
        qz_x = dist.normal.Normal(mu, torch.exp(0.5*log_var))
        qc_x = dist.categorical.Categorical(logits=alpha_logits)

        return (qz_x, qc_x), decoded

    def sample_concrete(self, logits, tau=0.1):
        u = torch.zeros_like(logits, device=self.device).uniform_(0, 1)
        g = -torch.log(-torch.log(u))
        return F.softmax((logits+g)/tau, dim=1)

    def loss_function(self, data):
        X, Y = data

        # Model Feedforward
        (qz_x, qc_x), decoded = self.forward(X)

        # Reconstruction mse
        recon = torch.square(decoded-X).sum(dim=1).mean()

        # KL-Divergence [ decomposition ]
        kl_z = dist.kl.kl_divergence(qz_x, self.pz).sum(-1).mean()
        kl_c = dist.kl.kl_divergence(qc_x, self.pc).mean()
        kl = kl_z + kl_c

        mu = qz_x.loc.clone()
        var = qz_x.scale.clone()**2

        # DIP-VAE I-II
        varEz_x = torch.cov(mu.t())
        Evarz_x = torch.diag(var.mean(dim=0))
        cov_z = varEz_x + Evarz_x

        dipvae_i = torch.square(torch.ones(
            self.latent_dim, device=self.device) - torch.diag(cov_z)).sum()
        dipvae_ii = torch.square(cov_z.flatten()[1:].view(
            self.latent_dim-1, self.latent_dim+1)[:, :-1].flatten()).sum()
        dipvae = 10*dipvae_i + dipvae_ii

        # Guided
        g = self.sample_concrete(qc_x.logits, tau=0.1)
        guided_logits = self.guide(g)
        guided_preds = guided_logits.argmax(dim=1)
        guided = F.cross_entropy(guided_logits, Y.type(torch.int64))

        # Final loss
        loss = recon + self.beta*kl + self.eta*guided + self.gamma*(dipvae)
        acc = (guided_preds == Y).sum().float()/(guided_preds.size(0))

        losses = ModelOutput(
            loss=loss,
            recon=recon,
            kl=kl,
            dipvae=dipvae,
            acc=acc
        )

        return losses

    def val_function(self, data):
        """
        to be applied over validation set during training. I'm interested in seeing validation accuracy
        of the guide branch 
        """
        X, Y = data

        (qz_x, qc_x), decoded = self.forward(X)

        g = self.sample_concrete(qc_x.logits, tau=0.1)
        guided_logits = self.guide(g)
        guided_preds = guided_logits.argmax(dim=1)

        val_acc = (guided_preds == Y).sum().float()/guided_preds.size(0)

        return ModelOutput(val_acc=val_acc)
