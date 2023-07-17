from collections import OrderedDict
from typing import Union, Optional, Callable, List
from pydantic import BaseModel, PositiveInt, PositiveFloat, Field
from enum import Enum
import torch
from typing import Callable

# ~borrowing~ conventions from https://github.com/clementchadebec/benchmark_VAE

# some potentially useful threads for Pydantic related issues
# https://stackoverflow.com/questions/72630488/valueerror-mutable-default-class-dict-for-field-headers-is-not-allowed-use
# https://stackoverflow.com/questions/68766527/how-to-set-a-default-value-for-enum-class-for-pydantic-fastapi


class ModelOutput(OrderedDict):
    """
    Wrapper providing a legible `__repr__` for printing various model outputs to console
    """

    def __repr__(self):
        """
        iterates over self.items to produce a " | " delimited str to print to console 
        """
        ps = ""
        for i, (k, v) in enumerate(self.items()):
            ps += f'{k}: {v:>4f}'

            if i != len(self) - 1:
                ps += " | "

        return ps


class TorchDeviceChoices(str, Enum):
    """
    Enum class constraining the possible device choices 
    """
    cpu = "cpu"
    cuda = "cuda"


class BaseConfig(BaseModel):
    """
    Base config file all VAEs require to pass high-level model params
    """
    input_dim: Union[PositiveInt, None] = None
    latent_dim: PositiveInt = 10
    device: TorchDeviceChoices = TorchDeviceChoices.cpu
    beta: float = 1.0
    recon_loss: Callable = lambda x, y: torch.square(y-x).sum(dim=1).mean()


class GuidedConfig(BaseConfig):
    """
    Config file for GuidedVAE 
    """
    guided_dim: PositiveInt
    eta: float = 10.0
    gamma: float = 1000.0
    elbo_scheduler: dict = Field(default_factory=lambda: {
                                 'beta': lambda x: 1.0, 'eta': lambda x: 1.0, 'gamma': lambda x: 1.0})


class MultiGuidedConfig(BaseConfig):
    """
    Config file for GuidedVAE 
    """
    guided_dims: List[PositiveInt]
    eta: float = 10.0
    gamma: float = 1000.0
    elbo_scheduler: dict = Field(default_factory=lambda: {
                                 'beta': lambda x: 1.0, 'eta': lambda x: 1.0, 'gamma': lambda x: 1.0})


class JointConfig(GuidedConfig):
    """
    Config file for JointVAE
    """
    disc_dim: PositiveInt = 2
