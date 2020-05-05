import torch.nn as nn
from torch import Tensor

from ..concrete_dropout import ConcreteDropout as CDr


def concrete_regulariser(model: nn.Module) -> nn.Module:

    """Adds ConcreteDropout regularisation functionality to a nn.Module.

    Parameters
    ----------
    model : nn.Module
        Model for which to calculate the ConcreteDropout regularisation.

    Returns
    -------
    model : nn.Module
        Model with additional functionality.
    """

    def regularisation(self) -> Tensor:

        """Calculates ConcreteDropout regularisation for each module.

        The total ConcreteDropout can be calculated by iterating through
        each module in the model and accumulating the regularisation for
        each compatible layer.

        Returns
        -------
        Tensor
            Total ConcreteDropout regularisation.
        """

        total_regularisation = 0
        for module in filter(lambda x: isinstance(x, CDr), self.modules()):
            total_regularisation += module.regularisation

        return total_regularisation

    setattr(model, 'regularisation', regularisation)

    return model
