from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Union, Optional, Callable

# borrowing conventions from https://github.com/clementchadebec/benchmark_VAE


class ModelOutput(OrderedDict):
    """
    Wrapper providing a nice __repr__ for printing various model outputs to console
    """

    def __repr__(self):
        '''
        format fields nicely for print on train/eval epoch ends 
        '''
        ps = ""
        for i, (k, v) in enumerate(self.items()):
            ps += f'{k}: {v:>4f}'

            if i != len(self) - 1:
                ps += " | "

        return ps


# RECONFIGURE ALL OF THE STUFF BELOW TO USE PYDANTIC
# FROM PYDANTIC IMPORT BASE

# some configs for vae's of interest
@dataclass
class BaseConfig:
    input_dim: Union[int, None] = None
    latent_dim: int = 10
    device: str = "cpu"


@dataclass
class GuideConfig(BaseConfig):
    guided_dim: int = 1
    beta: float = 1.0
    eta: float = 10.0
    gamma: float = 1000.0
    elbo_scheduler: dict = field(default_factory=lambda: {
                                 'beta': lambda x: 1.0, 'eta': lambda x: 1.0, 'gamma': lambda x: 1.0})
    # need to come up with a more adaptable way to do this :( --> wrapper?


# https://stackoverflow.com/questions/72630488/valueerror-mutable-default-class-dict-for-field-headers-is-not-allowed-use
