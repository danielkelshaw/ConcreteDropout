import torch
import torch.nn as nn
from torch import Tensor

from condrop.concrete_dropout import ConcreteDropout


class TestConcreteDropout:

    def test_forward(self):

        to_pass = torch.ones(5)

        cd = ConcreteDropout(
            weight_regulariser=1e-6,
            dropout_regulariser=1e-3
        )

        linear_layer = nn.Linear(5, 3)
        ret_tensor = cd(to_pass, linear_layer)

        assert isinstance(ret_tensor, torch.Tensor)
        assert ret_tensor.numel() == 3

        assert isinstance(cd.regularisation, Tensor)
        assert cd.regularisation.numel() == 1
