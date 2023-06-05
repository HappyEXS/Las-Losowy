"""Author: Antoni Kowalczuk"""
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from models.ID3 import ID3
from typing import Dict, Any, List, Tuple
from copy import deepcopy


class RandomForestClassifierID3SVM:
    """
    Class implementing the Random Forest Classifier with SVM and ID3

    Attributes:
        models: dict of models, structure: {
            idx: {
                'model': model instance,
                'attributes': attributes list on which model was trained and will predict,
                'indices': list of indices of the rows of the train dataframe,
                'model_type': 'SVM' or 'ID3',
                'categorical_attributes': list of categorical attributes the model has been trained on, SVM-specific,
                'categorical_encoder': OrdinalEncoder instance for categorical columns encoding in train and predict step, \
                    SVM-specific, if no categorical attributes are present then this key doesn't exist
            }
        }
        attributes_per_clf_percent: [0; 100] – % of attributes used by each model (so if attributes count = 10 and 60% all models see 6 random attributes)
        rows_per_clf_percent: [0; 100] – % of rows used by each model (so if rows count = 100 and 50% all models see 50 random attributes)
        clfs_cnt: classifiers count
        svm_percentage: [0; 100] – % of SVM models in the classifier (so if classifiers count = 100 and 30% there is 30 SVM instances in the model)
        svm_parameters: dict of SVC class initialization arguments (e.g. {'kernel': 'linear'})
        categorical_features: list of all categorical features in the training dataset, inferred from the self.fit() method
        attributes: list of all features in the training dataset, inferred from the self.fit() method
        target: target feature name, inferred from the self.fit() method
    """

    def __init__(
        self,
        attributes_per_clf_percent: int,
        rows_per_clf_percent: int,
        clfs_cnt: int,
        svm_percentage: int,
        svm_parameters: Dict[str, Any],
    ):
        self.models: Dict[int, Dict[str, Any]]
        self.models = dict()
        self.attributes_per_clf_percent = attributes_per_clf_percent
        self.rows_per_clf_percent = rows_per_clf_percent
        self.clfs_cnt = clfs_cnt
        self.svm_percentage = svm_percentage
        self.svm_parameters = svm_parameters
        self.categorical_features = []
        self.attributes = []
        self.target = ""

    def get_svm_and_id3_indices(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates random indices of the SVM and ID3 models from the range(self.clfs_cnt) values
        """
        mask = np.random.rand(self.clfs_cnt) <= (self.svm_percentage / 100)
        return np.arange(self.clfs_cnt)[mask], np.arange(self.clfs_cnt)[~mask]

    def initialize_svm(self) -> SVC:
        """
        Self-explanatory
        """
        return SVC(**self.svm_parameters)

    def initialize_id3(
        self, train_dataset: pd.DataFrame, attributes: np.ndarray, indices: list
    ) -> ID3:
        """
        Self-explanatory
        """
        categorical_attributes = self.get_categoricals_from_attributes_subset(
            attributes
        ).tolist()
        dataset = train_dataset.loc[indices, np.append(attributes, self.target)]
        return ID3(
            dataset=dataset,
            categorical_attributes=categorical_attributes,
            target_col_name=self.target,
        )

    def initialize_attributes_for_clf(self) -> np.ndarray:
        """
        Draws random attributes for one clf
        """
        if self.attributes_per_clf_percent == 100:
            return np.array(self.attributes)
        attributes = np.unique(
            np.random.choice(
                self.attributes,
                int(len(self.attributes) * (self.attributes_per_clf_percent / 100)),
            )
        )
        if len(attributes) == 0:
            return np.array(np.random.choice(self.attributes)).reshape(-1)
        return attributes

    def get_categoricals_from_attributes_subset(
        self, attributes: np.ndarray
    ) -> np.ndarray:
        """
        Returns the categorical features from the subset of random attributes
        """
        return np.array(
            list(set(attributes).intersection(set(self.categorical_features)))
        )

    def initialize_rows_indices_for_clf(self, indices: list) -> np.ndarray:
        """
        Draws random indices of the rows of the training dataset for one clf
        """
        if self.rows_per_clf_percent == 100:
            return np.array(indices)
        rows_indices = np.unique(
            np.random.choice(
                indices, int(len(indices) * (self.rows_per_clf_percent / 100))
            )
        )
        if len(rows_indices) == 0:
            return np.array(np.random.choice(indices)).reshape(-1)
        return rows_indices

    def initialize_models(self, train_dataset: pd.DataFrame):
        """
        Entrypoint to initialize all models and initialize the self.models dict
        """
        svm_indices, id3_indices = self.get_svm_and_id3_indices()
        for idx in svm_indices:
            self.models[idx] = {
                "model": self.initialize_svm(),
                "attributes": self.initialize_attributes_for_clf(),
                "indices": self.initialize_rows_indices_for_clf(
                    train_dataset.index.tolist()
                ),
                "model_type": "SVM",
            }
            self.models[idx][
                "categorical_attributes"
            ] = self.get_categoricals_from_attributes_subset(
                self.models[idx]["attributes"]
            )

        for idx in id3_indices:
            model_data = {
                "attributes": self.initialize_attributes_for_clf(),
                "indices": self.initialize_rows_indices_for_clf(
                    train_dataset.index.tolist()
                ),
                "model_type": "ID3",
            }
            model = self.initialize_id3(
                train_dataset=deepcopy(
                    train_dataset.loc[
                        model_data["indices"],
                        np.append(model_data["attributes"], self.target),
                    ]
                ),
                attributes=model_data["attributes"],
                indices=model_data["indices"],
            )
            model_data["model"] = model
            self.models[idx] = model_data

    def svm_fit_cat_encoder(self, X: pd.DataFrame, model_data: Dict[str, Any]) -> None:
        """
        Fits OrdinalEncoder for one SVM model on the training data
        If no categorical columns in the data → does nothing
        """
        if len(model_data["categorical_attributes"]) == 0:
            return
        model_data["categorical_encoder"] = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        ).fit(X[model_data["categorical_attributes"]])

    def svm_preprocess_categoricals(
        self, X: pd.DataFrame, model_data: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Transforms the categorical features from X using the fitted OrdinalEncoder of this SVM model instance
        If no categorical columns in the data → just returns untransformed X
        """
        if len(model_data["categorical_attributes"]) == 0:
            return X
        X[model_data["categorical_attributes"]] = model_data[
            "categorical_encoder"
        ].transform(X[model_data["categorical_attributes"]])
        return X

    def get_X_y(
        self, train_dataset: pd.DataFrame, model_data: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Splits train dataset into X and y
        If SVM → fits OrdinalEndcoder, transforms categoricals
        """
        X = train_dataset.loc[model_data["indices"], model_data["attributes"]]
        y = train_dataset.loc[model_data["indices"], self.target]
        if model_data["model_type"] == "SVM":
            self.svm_fit_cat_encoder(X, model_data)
            X = self.svm_preprocess_categoricals(X, model_data)
        return X, y

    def fit_single_model(
        self, model_data: Dict[str, Any], train_dataset: pd.DataFrame
    ) -> None:
        """
        Self-explanatory
        """
        if model_data["model_type"] == "ID3":
            model_data["model"].fit()
        elif model_data["model_type"] == "SVM":
            X, y = self.get_X_y(train_dataset=train_dataset, model_data=model_data)
            model_data["model"].fit(X, y)
        else:
            raise ValueError(
                f"UNEXPECTED MODEL TYPE {model_data['model_type']}, SUPPORTED TYPES: SVM, ID3. EXITTING"
            )

    def fit(
        self,
        train_dataset: pd.DataFrame,
        categorical_features: List[str],
        target_feature: str,
    ) -> None:
        """
        Entrypoint to fit the created Classifier instance using passed training data.
        Initializes all the models and trains them using the training data

        Arguments:
            train_dataset: pd.DataFrame containing all the features and the target column
            categorical_features: list of categorical feature column names. If it's wrong stuff will break
            target_feature: target column name
        """
        self.categorical_features = categorical_features
        self.attributes = [
            col for col in train_dataset.columns if col != target_feature
        ]
        self.target = target_feature
        self.initialize_models(train_dataset=train_dataset)
        for idx, model_data in self.models.items():
            self.fit_single_model(model_data=model_data, train_dataset=train_dataset)

    def predict_single(self, X: pd.DataFrame, model_data: Dict[str, Any]) -> pd.Series:
        """
        Returns predictions of one model on the input data
        """
        if model_data["model_type"] == "SVM":
            X = self.svm_preprocess_categoricals(X, model_data)
            return pd.Series(
                model_data["model"].predict(X[model_data["attributes"]]), index=X.index
            )
        elif model_data["model_type"] == "ID3":
            return model_data["model"].predict(X[model_data["attributes"]])
        else:
            raise ValueError(
                f"UNEXPECTED MODEL TYPE {model_data['model_type']}, SUPPORTED TYPES: SVM, ID3. EXITTING"
            )

    def predict(self, X: pd.DataFrame):
        """
        Entrypoint to get predictions from the classifier

        Arguments:
            X: pd.DataFrame containing all the features
        """
        assert sorted(self.attributes) == sorted(
            X.columns.tolist()
        ), f"FEATURES IN X ARE DIFFERENT THAN IN MODEL, MODEL: {self.attributes}, X: {X.columns.tolist()}, EXITTING"
        assert X.shape[0] != 0, "EMPTY DATAFRAME, EXITTING"
        preds = dict()
        for idx, model_data in self.models.items():
            tmp_X = deepcopy(X)
            preds[idx] = self.predict_single(tmp_X, model_data)
        preds = pd.DataFrame(preds, index=X.index)
        return preds.apply(lambda x: np.unique(x, return_counts=True), axis=1).apply(
            lambda x: x[0][x[1].argmax()]
        )
