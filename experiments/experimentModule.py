"""Author: Jan Jedrzejewski"""

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from typing import Dict, Any, List, Tuple

from models.RandomForest import RandomForestClassifierID3SVM
from experiments.functions import (
    get_dataset,
    get_categorical_features,
    calculate_accuracy,
    get_labels,
)


class ExperimentModule:
    def __init__(
        self,
        dataset_name: str,
        number_of_experiments: int,
        test_set_percentage: int,
        forest_parameters: List[int],
        svm_parameters: Dict[str, Any],
    ) -> None:
        self.dataset_name = dataset_name
        self.number_of_experiments = number_of_experiments
        self.test_set_percentage = test_set_percentage
        self.forest_parameters = forest_parameters
        self.svm_parameters = svm_parameters

        self.model: RandomForestClassifierID3SVM
        self.bias = []
        self.accuracy = []
        self.conf_matix = []

    def single_experiment(self) -> None:
        self.model = RandomForestClassifierID3SVM(
            # attributes_per_clf_percent=self.forest_parameters[0],
            attributes_per_clf_percent=int((len(self.categorical_features)-1)**(1/2)),
            rows_per_clf_percent=self.forest_parameters[1],
            clfs_cnt=self.forest_parameters[2],
            svm_percentage=self.forest_parameters[3],
            svm_parameters=self.svm_parameters,
        )
        train_dataset, test_dataset = train_test_split(
            self.dataset, shuffle=True, test_size=self.test_set_percentage / 100
        )

        self.model.fit(train_dataset, self.categorical_features, "target")

        test_orginal_class = test_dataset["target"].to_numpy()
        test_predictions = self.model.predict(
            test_dataset[test_dataset.columns[:-1]]
        ).to_numpy()
        train_orginal_class = train_dataset["target"].to_numpy()
        train_predictions = self.model.predict(
            train_dataset[test_dataset.columns[:-1]]
        ).to_numpy()

        self.bias.append(calculate_accuracy(train_orginal_class, train_predictions))
        self.accuracy.append(calculate_accuracy(test_orginal_class, test_predictions))
        self.conf_matix.append(
            confusion_matrix(test_orginal_class, test_predictions, labels=self.labels)
        )

    def run(self) -> Tuple[List[float], List[float], List[Any]]:
        self.dataset = get_dataset(self.dataset_name)
        self.categorical_features = get_categorical_features(self.dataset_name)
        self.labels = get_labels(self.dataset_name)

        for _ in range(self.number_of_experiments):
            self.single_experiment()

        return (self.bias, self.accuracy, self.conf_matix)
