import argparse
import csv

import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch import Tensor
from torchvision import datasets, transforms

import condrop
from condrop import ConcreteDropout


@condrop.concrete_regulariser
class MLPConcreteDropout(nn.Module):

    def __init__(self, n_hidden: int = 512) -> None:

        super().__init__()

        self.fc1 = nn.Linear(784, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, n_hidden)
        self.fc4 = nn.Linear(n_hidden, 10)

        w, d = 1e-6, 1e-3
        self.cd1 = ConcreteDropout(weight_regulariser=w, dropout_regulariser=d)
        self.cd2 = ConcreteDropout(weight_regulariser=w, dropout_regulariser=d)
        self.cd3 = ConcreteDropout(weight_regulariser=w, dropout_regulariser=d)
        self.cd4 = ConcreteDropout(weight_regulariser=w, dropout_regulariser=d)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: Tensor) -> Tensor:

        x = x.view(-1, 784)

        x = self.cd1(x, nn.Sequential(self.fc1, self.relu))
        x = self.cd2(x, nn.Sequential(self.fc2, self.relu))
        x = self.cd3(x, nn.Sequential(self.fc3, self.relu))
        x = self.cd4(x, nn.Sequential(self.fc4, self.softmax))

        return x


def train(model, trainloader, optimiser, epoch, device):

    train_loss = 0.0

    model.train()
    for batch_idx, (data, labels) in enumerate(trainloader):
        data, labels = data.to(device), labels.to(device)

        optimiser.zero_grad()

        outputs = model(data)
        loss = F.cross_entropy(outputs, labels) + model.regularisation()
        train_loss += loss.item() * data.size(0)

        loss.backward()
        optimiser.step()

    train_loss /= len(trainloader.dataset)
    return train_loss


def test(model, testloader, device):

    correct = 0
    total = 0
    test_loss = 0.0

    model.eval()
    for batch_idx, (data, labels) in enumerate(testloader):
        data, labels = data.to(device), labels.to(device)

        outputs = model(data)
        loss = F.cross_entropy(outputs, labels)
        test_loss += loss.item() * data.size(0)

        preds = outputs.argmax(dim=1, keepdim=True)
        correct += preds.eq(labels.view_as(preds)).sum().item()

    test_loss /= len(testloader.dataset)
    accuracy = correct / len(testloader.dataset)

    return test_loss, accuracy


if __name__ == '__main__':

    # read command line arguments
    parser = argparse.ArgumentParser(
        description='Concrete Dropout - MNIST Classification Example.'
    )

    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--testbatch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=15, metavar='N',
                        help='number of epochs to train (default: 15)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training (default: false)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='batch logging interval (defaul: 10)')

    args = parser.parse_args()

    # set training device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # load / process data
    trainset = datasets.MNIST('./data',
                              train=True,
                              download=True,
                              transform=transform)

    testset = datasets.MNIST('./data',
                             train=False,
                             download=True,
                             transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=args.batch_size,
                                              **kwargs)

    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=args.testbatch_size,
                                             **kwargs)

    # define model / optimizer
    model = MLPConcreteDropout().to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)

    # prepare results file
    rows = ['epoch', 'train_loss', 'test_loss',
            'accuracy', 'dp1', 'dp2', 'dp3', 'dp4']

    with open('results.csv', 'w+', newline="") as f_out:
        writer = csv.writer(f_out, delimiter=',')
        writer.writerow(rows)

    # run training
    min_test_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        train_loss = train(model, trainloader, optimiser, epoch, device)
        test_loss, accuracy = test(model, testloader, device)

        # extract dropout probabilities
        probs = [cd.p.cpu().data.numpy()[0] for cd in filter(
            lambda x: isinstance(x, ConcreteDropout), model.modules()
        )]

        _results = [epoch, train_loss, test_loss, accuracy]
        _results.extend(probs)

        if len(rows) != len(_results):
            raise ValueError('Invalid number of output rows.')

        with open('results.csv', 'a', newline="") as f_out:
            writer = csv.writer(f_out, delimiter=',')
            writer.writerow(_results)
