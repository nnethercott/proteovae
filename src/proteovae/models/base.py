import torch
from torch import nn
from typing import List


class Encoder(nn.Module):
    r"""Generic Encoder implementation inheriting from :class:`torch.nn.Module`

    .. code-block::

        >>> from proteovae.models.base import Encoder 
        ...
        >>> input_dim = 512
        >>> latent_dim = 16
        >>> hidden_dims1 = [256,128,64,]
        >>> hidden_dims2 = [256,]
        ...
        >>> enc1 = Encoder(input_dim, latent_dim, hidden_dims1)
        >>> enc2 = Encoder(input_dim, latent_dim, hidden_dims2)
    """

    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: List):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims

        # variable length hidden layers
        encoder_block = [
            nn.Linear(self.input_dim, self.hidden_dims[0]), nn.ReLU(),]
        for i in range(len(hidden_dims)-1):
            encoder_block += [nn.Linear(self.hidden_dims[i],
                                        self.hidden_dims[i+1]), nn.ReLU()]

        self.linear_block = nn.Sequential(
            *encoder_block
        )

        # loc and scale
        self.fc_mu = nn.Linear(self.hidden_dims[-1], self.latent_dim)
        self.fc_logvar = nn.Linear(self.hidden_dims[-1], self.latent_dim)

    def forward(self, x):
        r"""
        Parameters:
            x (~torch.Tensor): data batch of dimension [B x N] 

        Returns:
            dict(str, [~torch.Tensor, ~torch.Tensor]): location-scale parameters of posterior distribution

        """
        x = self.linear_block(x)
        mu = self.fc_mu(x)
        log_var = self.fc_logvar(x)

        output = {'cont': (mu, log_var)}

        return output


class Decoder(nn.Module):
    r"""Generic Encoder implementation inheriting from :class:`torch.nn.Module`

    """

    def __init__(self, output_dim, latent_dim, hidden_dims):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims

        decoder_block = [
            nn.Linear(self.latent_dim, self.hidden_dims[0]), nn.ReLU()]
        for i in range(len(hidden_dims)-1):
            decoder_block += [nn.Linear(self.hidden_dims[i],
                                        self.hidden_dims[i+1]), nn.ReLU()]

        # no relu since INT transform
        decoder_block += [nn.Linear(self.hidden_dims[-1], self.output_dim)]

        self.linear_block = nn.Sequential(
            *decoder_block
        )

    def forward(self, x):
        r"""
        Maps embeddings to reconstructions

        Parameters: 
            x (~torch.Tensor): tensor of batch embeddings [B x D]

        Returns:
            recon (~torch.Tensor): tensor of reconstructions [B x N]
        """
        recon = self.linear_block(x)
        return recon


class Guide(nn.Module):
    r"""Shallow logistic regression implementation. 
    """

    def __init__(self, dim_in, dim_out):
        super(Guide, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(dim_in, dim_out),
        )

    def forward(self, x):
        """
        Returns:
            ~torch.Tensor: logits of class probabilites of shape [B x C], with B batch size and C num classes
        """
        return self.classifier(x)


class JointEncoder(Encoder):
    r"""
    Implementation following https://arxiv.org/abs/1804.00104 allowing 
    for categorical latent dimensions in addition to the standard continuous ones.
    Subclasses :class:`~Encoder`. 
    """

    def __init__(self, input_dim, latent_dim, hidden_dims, disc_dim):
        super().__init__(input_dim, latent_dim, hidden_dims)

        # discrete
        self.disc_dim = disc_dim
        self.fc_alpha_logits = nn.Linear(self.hidden_dims[2], self.disc_dim)

    def forward(self, x):
        r"""
        Parameters:
            x (~torch.Tensor): data batch of dimension [B x N] 

        Returns:
            dict(str, [~torch.Tensor, ~torch.Tensor]): location-scale parameters of categorical and normal posterior distributions

        .. code-block::

            >>> from proteovae.models.base import JointDecoder
            ...
            >>> joint_decoder = JointDecoder(input_dim = 512, 
            ...                              latent_dim = 16, 
            ...                              hidden_dims = [256,128,], 
            ...                              discrete_dim =2)
            ...
            >>> post_params = joint_decoder(torch.rand(64, 512))
            >>> cont = post_params['cont']
            >>> disc = post_params['disc']
        """

        x = self.linear_block(x)
        mu = self.fc_mu(x)
        log_var = self.fc_logvar(x)

        alpha_logits = self.fc_alpha_logits(x)

        return {'cont': (mu, log_var), 'disc': alpha_logits}
