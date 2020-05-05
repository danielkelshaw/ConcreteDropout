import pytest
import torch
import torch.nn as nn
from torch import Tensor

from condrop import ConcreteDropout
from condrop import concrete_regulariser


class TestConcreteRegulariser:

    @pytest.fixture
    def network(self):

        @concrete_regulariser
        class Network(nn.Module):

            def __init__(self):
                super().__init__()

                weight_regulariser = 1e-6
                dropout_regulariser = 1e-3

                self.fc1 = nn.Linear(5, 10)
                self.fc2 = nn.Linear(10, 10)
                self.fc3 = nn.Linear(10, 1)

                self.relu = nn.ReLU()

                self.condrop1 = ConcreteDropout(
                    weight_regulariser=weight_regulariser,
                    dropout_regulariser=dropout_regulariser
                )

                self.condrop2 = ConcreteDropout(
                    weight_regulariser=weight_regulariser,
                    dropout_regulariser=dropout_regulariser
                )

            def forward(self, x):

                x = self.condrop1(x, nn.Sequential(self.fc1, self.relu))
                x = self.condrop2(x, nn.Sequential(self.fc2, self.relu))
                x = self.fc3(x)

                return x

        return Network

    def test_regularisation(self, network):

        to_feed = torch.ones(5)
        model = network()

        output = model(to_feed)
        regularisation = model.regularisation()

        assert isinstance(output, Tensor)
        assert output.numel() == 1

        assert isinstance(regularisation, Tensor)
        assert regularisation.numel() == 1
