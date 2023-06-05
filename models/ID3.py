"""Author: Antoni Kowalczuk"""
import numpy as np
import pandas as pd
from typing import List
from copy import deepcopy


class Node:
    def __init__(self, attribute_name=None, class_=None):
        self.attribute_name = attribute_name
        self.nodes = {}
        self.class_ = class_

    def add_node(self, value):
        self.nodes[value] = Node()

    def __str__(self):
        return f"{self.attribute_name}, {self.nodes}, {self.class_}"


class ID3:
    """
    Class implementing the ID3 algorithm adjusted for the continual variables (we split them by deciles)

    Attributes:
        root: root of the tree
        target: target column name
        attributes: attributes lits
        categorical_atttributes: categorical attributes list
        continual_attributes: continual attributers list
        continual_attributes_bin_splits: Dict[column_name, Dict[splits: IntervalIndex, min: float, max: float]] used to split continual attributes into buckets
        classes: classes list
        dataset: dataset
        X: X
        y: y
        U: U
    """

    def __init__(
        self,
        dataset: pd.DataFrame,
        categorical_attributes: List[str],
        target_col_name: str,
    ):
        """
        Arguments:
            dataset: DataFrame
            categorical_attributes: list of categorical attributes in the dataset
            target_col_name: string representing the column name in the dataset with the target variable
        """
        self.root = Node()

        self.target = target_col_name
        self.attributes = np.array(
            [col for col in dataset.columns if col != target_col_name]
        )
        self.categorical_attributes = categorical_attributes
        self.continual_attributes = [
            col for col in self.attributes if col not in self.categorical_attributes
        ]
        self.continual_attributes_bin_splits = dict()

        self.classes = np.unique(dataset.loc[:, target_col_name])
        self.dataset = deepcopy(dataset)

        self.create_bins()
        self.encode_continuals(self.dataset)

        self.X = self.dataset.loc[:, self.attributes]
        self.y = self.dataset.loc[:, target_col_name]
        self.U = deepcopy(self.dataset)

    def create_bins(self):
        """
        This methods populates the continual_attributes_bin_splits by utilising the pd.qcut percentile split function
        """
        for col in self.continual_attributes:
            splits = pd.qcut(
                self.dataset[col].values, q=10, duplicates="drop"
            ).categories
            self.continual_attributes_bin_splits[col] = {
                "splits": splits,
                "min": self.dataset[col].min(),
                "max": self.dataset[col].max(),
            }

    @staticmethod
    def find_encoding(x, splits_data):
        """
        Util "lambda-like" function used for applying the encoding from the continual attributes binning to the DataFrame column
        Usage df[col].apply(lambda x: find_encoding(x, splits_data))
        """
        try:
            return splits_data["splits"].get_loc(x)
        except KeyError as e:
            if x <= splits_data["min"]:
                return 0
            else:
                return len(splits_data["splits"])

    def encode_continuals(self, X):
        """
        Util method to encode all continual attributes in the input DataFrame
        """
        for col in self.continual_attributes:
            splits_data = self.continual_attributes_bin_splits[col]
            X[col] = X[col].apply(lambda x: self.find_encoding(x, splits_data))

    def inf_gain(self, d, U):
        """
        This is the maths behind the ID3.
        U - all dataset
        U_j - set when the attribute d has only value = j
        Entropy: negative sum of the [value count * log of value count for each target value in the set]
        Divided entropy: sum of the [entropies of U_j for all unique values of the attribute d, mutliplied by len(U_j) / len(U)]
        Information Gain: entropy of U - divided entropy by d of U
        """
        entropy = lambda y_: -sum(
            [
                fi * np.log(fi) for fi in np.unique(y_, return_counts=True)[1]
            ]  # entropy for a given set and a given attribute (only target in this case)
        )
        divided_entropy = sum(
            [
                len(U.loc[U[d] == j, d])  # len(U_j)
                / len(U)  # len(U)
                * entropy(U.loc[U[d] == j, self.target])  # entropy of the U_j on target
                for j in np.unique(
                    U[d]
                )  # for each unique value of the attribute d in the set
            ]
        )
        return entropy(U.loc[:, self.target]) - divided_entropy  # inf gain

    def fit(self):
        """
        Entry point to fit the ID3 model using the data passed during initialization of the model
        """
        self.fit_recurr(self.root, self.attributes, self.U)

    def fit_recurr(self, node: Node, D: np.ndarray, U: pd.DataFrame):
        if len(np.unique(U.loc[:, self.target])) == 1:
            node.class_ = np.unique(U.loc[:, self.target])[0]
            return

        values, counts = np.unique(U.loc[:, self.target], return_counts=True)
        node.class_ = values[np.argmax(counts)]
        if len(D) == 0:
            return

        d = D[np.argmax([self.inf_gain(d, U) for d in D])]
        node.attribute_name = d
        for j in np.unique(U[d]):
            # print(D, j, d)
            node.add_node(j)
            # print(node.nodes)
            D_minus = np.array([D_ for D_ in D if D_ != d])
            Uj = U.loc[U[d] == j, np.append(D_minus, self.target)]
            self.fit_recurr(node.nodes[j], D_minus, Uj)
        assert len(node.nodes) == len(np.unique(U[d]))

    def predict(self, X: pd.DataFrame):
        """
        Entry point to use the ID3 model to get predictions on the new data
        """
        new_X = deepcopy(X)
        self.encode_continuals(new_X)
        return X.apply(lambda x: self.predict_single(x), axis=1)

    def predict_single(self, x):
        node = self.root
        while node.attribute_name is not None:
            try:
                node = node.nodes[x.loc[node.attribute_name]]
            except:
                return node.class_
        return node.class_
