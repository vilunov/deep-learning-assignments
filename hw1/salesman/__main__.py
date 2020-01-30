import numpy as np
import pandas as pd
import torch
from torch.distributions import Uniform

import draw


def read_data():
    data = pd.read_csv("cities.csv").nlargest(30, "Население")

    names = list(data["Город"])
    lat = data["Широта"].to_numpy()
    long = data["Долгота"].to_numpy()

    R = 6378137.0  # radius of Earth in meters
    phi_1 = np.cos(np.radians(lat))
    phi = np.expand_dims(phi_1, 1) * np.expand_dims(phi_1, 0)
    delta_phi = np.radians(np.expand_dims(lat, 1) - np.expand_dims(lat, 0))
    delta_lambda = np.radians(np.expand_dims(long, 1) - np.expand_dims(long, 0))
    a = np.sin(delta_phi / 2.0) ** 2 + np.multiply(phi, np.sin(delta_lambda / 2.0) ** 2)

    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    meters = R * c  # output distance in meters
    km = meters / 1000.0  # output distance in kilometers
    return names, km, lat, long


def calc_length(distances, path):
    return distances[path[:-1], path[1:]].sum()


def fit_anneal(
    distances,
    lat,
    long,
    high_temperature: float = 1000.0,
    low_temperature: float = 0.1,
    anneal_period: int = 150,
    anneal_coefficient: float = 0.75,
):
    acceptor = Uniform(0.0, 1.0)
    n = distances.shape[0]
    path = np.arange(0, n)
    np.random.shuffle(path)
    loss = calc_length(distances, path)
    temperature = high_temperature
    best_loss: float = loss
    best_path = path.copy()

    all_paths = [path.copy()]

    epoch = 0
    while temperature > low_temperature:
        for _ in range(anneal_period):
            i = np.random.randint(0, n)
            j = np.random.randint(0, n - 1)
            if j >= i:
                j += 1
            path[[i, j]] = path[[j, i]]  # swap
            new_loss = calc_length(distances, path)

            if new_loss < best_loss:
                best_loss = new_loss
                best_path = path.copy()

            # undo movement if not accepted
            if (loss < new_loss) and (
                (np.exp(-new_loss / temperature) / np.exp(-loss / temperature))
                < acceptor.sample(torch.Size()).item()
            ):
                path[[i, j]] = path[[j, i]]
            else:
                loss = new_loss
            all_paths.append(best_path)
        if epoch % 1 == 0:
            print("number of epoch", epoch, "loss", best_loss, loss, temperature)
        temperature *= anneal_coefficient
        epoch += 1
    return best_path, all_paths


if __name__ == "__main__":
    names, distances, lat, long = read_data()
    best_path, all_paths = fit_anneal(distances, lat, long)
    draw.animate(all_paths, np.stack([lat, long]).T)
