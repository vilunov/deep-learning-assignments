from typing import Optional

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

import torch
import torch.nn as nn
from torch.distributions import Normal, Uniform
from torch.autograd import Variable

from math import exp


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(4, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 3)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, X):
        X = torch.tanh(self.fc1(X))
        X = torch.tanh(self.fc2(X))
        X = self.fc3(X)
        X = self.softmax(X)
        return X


def fit_net(net: Net, train_X, train_y):
    criterion = nn.CrossEntropyLoss()  # cross entropy loss
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

    prev_loss: Optional[float] = None
    for epoch in range(100000):
        optimizer.zero_grad()
        out = net(train_X)
        loss = criterion(out, train_y)
        loss.backward()
        optimizer.step()

        new_loss = loss.item()
        if epoch % 1000 == 0:
            print("number of epoch", epoch, "loss", new_loss)
        if prev_loss is not None and abs(new_loss - prev_loss) < 1e-8:
            break
        prev_loss = new_loss


def eval_net(net: Net, test_X, test_y):
    predict_out = net(test_X)
    _, predict_y = torch.max(predict_out, 1)
    print("prediction accuracy", accuracy_score(test_y.data, predict_y.data))
    print(
        "macro precision", precision_score(test_y.data, predict_y.data, average="macro")
    )
    print(
        "micro precision", precision_score(test_y.data, predict_y.data, average="micro")
    )
    print("macro recall", recall_score(test_y.data, predict_y.data, average="macro"))
    print("micro recall", recall_score(test_y.data, predict_y.data, average="micro"))


def fit_net_se(
    net: Net,
    train_X,
    train_y,
    neighbor_sigma: float = 0.5,
    temperature_high: float = 1.0,
    temperature_low: float = 0.1,
    anneal_period: int = 10,
    anneal_coefficient: float = 0.99,
):
    criterion = nn.CrossEntropyLoss()  # cross entropy loss -- energy func
    neighbor_gen = Normal(0, neighbor_sigma)
    acceptor = Uniform(0.0, 1.0)
    params = [
        net.fc1.bias.data,
        net.fc1.weight.data,
        net.fc2.bias.data,
        net.fc2.weight.data,
        net.fc3.bias.data,
        net.fc3.weight.data,
    ]
    prev_loss: float = criterion(net(train_X), train_y).item()
    temperature: float = temperature_high
    best_loss: float = prev_loss
    best_weights = [torch.zeros(param.shape) for param in params]

    epoch = 0
    while temperature > temperature_low:
        for _ in range(anneal_period):
            deltas = [neighbor_gen.sample(param.shape) for param in params]
            for param, delta in zip(params, deltas):
                param += delta
            new_loss = criterion(net(train_X), train_y).item()

            # undo movement if not accepted
            if (prev_loss < new_loss) and (
                (exp(-new_loss / temperature) / exp(-prev_loss / temperature))
                < acceptor.sample(torch.Size()).item()
            ):
                for param, delta in zip(params, deltas):
                    param -= delta
            else:
                prev_loss = new_loss

            # save best weights
            if best_loss is None or new_loss < best_loss:
                best_loss = new_loss
                for param, best_param in zip(params, best_weights):
                    best_param[:] = param
        if epoch % 1 == 0:
            print("number of epoch", epoch, "loss", best_loss)
        temperature *= anneal_coefficient
        epoch += 1
    for param, best_param in zip(params, best_weights):
        param[:] = best_param


def load_data():
    # load IRIS dataset
    dataset = pd.read_csv("iris.csv")

    # transform species to numerics
    dataset.loc[dataset.species == "Iris-setosa", "species"] = 0
    dataset.loc[dataset.species == "Iris-versicolor", "species"] = 1
    dataset.loc[dataset.species == "Iris-virginica", "species"] = 2

    train_X, test_X, train_y, test_y = train_test_split(
        dataset[dataset.columns[0:4]].values, dataset.species.values, test_size=0.8,
    )

    train_X = Variable(torch.Tensor(train_X).float())
    test_X = Variable(torch.Tensor(test_X).float())
    train_y = Variable(torch.Tensor(train_y).long())
    test_y = Variable(torch.Tensor(test_y).long())
    return train_X, test_X, train_y, test_y


if __name__ == "__main__":
    train_X, test_X, train_y, test_y = load_data()
    net1 = Net()
    fit_net(net1, train_X, train_y)
    eval_net(net1, test_X, test_y)
    print()
    net2 = Net()
    fit_net_se(net2, train_X, train_y)
    eval_net(net2, test_X, test_y)
