"""Author: Jan Jedrzejewski"""

from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import json
from typing import List, Any, Dict, Tuple
import pickle


def plot_confusion_matrix(
    conf_matrix: np.ndarray,
    target_labels: np.ndarray,
    title=" ",
) -> None:

    plt.figure(figsize=(5, 4))
    sns.heatmap(
        np.flip(conf_matrix, 0),
        annot=True,
        cmap="YlGnBu",
        fmt=".2f",
        xticklabels=target_labels,
        yticklabels=target_labels[::-1],
    )

    plt.xlabel("Predicted Class")
    plt.ylabel("Orginal Class")
    plt.title(label=title)

    plt.savefig(f"results/{title}.png")
    plt.show()


def get_dataset(fname: str) -> pd.DataFrame:
    format_dataset()
    return pd.read_csv(f"datasets/{fname}.csv", sep=",")


def format_dataset():
    pass


def get_categorical_features(dataset_name) -> List[str]:
    file = open(f"datasets/features.json")
    data = json.load(file)
    return data[dataset_name]


def get_labels(dataset_name) -> List[str]:
    file = open(f"datasets/features.json")
    data = json.load(file)
    return data["labels"][dataset_name]


def calculate_accuracy(orginal: List[Any], predictions: List[Any]) -> float:
    if len(orginal) != len(predictions) or len(orginal) == 0:
        return None

    counter = 0
    for i in range(len(orginal)):
        if orginal[i] == predictions[i]:
            counter += 1

    return round(counter / len(orginal), 2)


def aggregate_results(
    results: Tuple[List[float], List[float], List[Any]]
) -> Tuple[float, Any]:
    return aggregate_accuracy(results[1]), aggregate_conf_matrix(results[2])


def aggregate_accuracy(accuracy_list: List[float]) -> float:
    return round(sum(accuracy_list) / len(accuracy_list), 2)


def aggregate_conf_matrix(conf_matrix_list: List[Any]) -> np.ndarray:
    size = len(conf_matrix_list[0])
    aggregated = np.zeros(shape=(size, size), dtype=int)
    for matrix in conf_matrix_list:
        aggregated += matrix
    return aggregated / size


def format_results(data: Tuple[List[float], List[float], List[Any]]) -> Dict[str, Any]:
    formated = {}
    for i in range(len(data[1])):
        formated[f"{i}"] = [data[1][i], data[2][i]]
    return formated


def save_results(data: Dict[str, Dict], fname: str) -> None:
    with open(f"results/{fname}.pickle", "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def save_results_to_json(data: Dict[str, float], fname: str) -> None:
    with open(f"results/{fname}.json", "w") as handle:
        json.dump(data, handle)
