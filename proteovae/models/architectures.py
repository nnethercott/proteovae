import torch
from torch import nn
from typing import List

# MAKE SURE TO ADD THE TYPES IN ALL OF THE FUNCTIONS AND STUFF SO IT LOOKS NICE IN SPHINX


class Encoder(nn.Module):
    """
    doc string for the encoder 

    """

    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: List):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        # build variable length encoder block
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
        x = self.linear_block(x)
        mu = self.fc_mu(x)
        log_var = self.fc_logvar(x)

        return {'cont': (mu, log_var)}


class Decoder(nn.Module):
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
        recon = self.linear_block(x)
        return recon


class Guide(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(Guide, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(dim_in, dim_out),
        )

    def forward(self, x):
        return self.classifier(x)


class JointEncoder(Encoder):
    def __init__(self, input_dim, latent_dim, hidden_dims, disc_dim):
        super().__init__(input_dim, latent_dim, hidden_dims)

        # discrete
        self.disc_dim = disc_dim
        self.fc_alpha_logits = nn.Linear(self.hidden_dims[2], self.disc_dim)

    def forward(self, x):
        x = self.linear_block(x)
        mu = self.fc_mu(x)
        log_var = self.fc_logvar(x)

        alpha_logits = self.fc_alpha_logits(x)

        return {'cont': (mu, log_var), 'disc': alpha_logits}
