import logging
import os
import pickle
from typing import Dict
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from automl.preprocess import preprocess_pipeline, PreprocessingError

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def validate_input_parameters(file_path: str, target_column: str, task: str) -> None:
    """Validate input file existence and task type."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found at: {file_path}")
    if task not in ['classification', 'regression']:
        raise ValueError(f"Invalid task type: {task}. Must be 'classification' or 'regression'.")


class ModelTrainer:
    def __init__(self,
                 model: str,
                 params: dict,
                 file_path: str,
                 target_column: str,
                 task: str = 'classification',
                 preprocessed: bool = False,
                 feature_selection: bool = False,
                 outlier_threshold: float = 3.0,
                 imbalance_threshold: float = 2.0,
                 numeric_strategy: str = 'median',
                 categorical_strategy: str = 'most_frequent',
                 random_state: int = 42,
                 max_categories_for_ohe: int = 50,
                 min_samples_for_resampling: int = 10,
                 hyperparameter_grids: Dict = None,
                 tune_parameters: bool = False,
                 get_best_model: bool = False): 
        """
        Initialize the model trainer with configuration and file parameters.
        """
        validate_input_parameters(file_path, target_column, task)

        self.model_name = model  
        self.params = params
        self.file_path = file_path
        self.target_column = target_column.strip().lower().replace(" ", "_")
        self.task = task
        self.preprocessed = preprocessed
        self.feature_selection = feature_selection
        self.outlier_threshold = outlier_threshold
        self.imbalance_threshold = imbalance_threshold
        self.numeric_strategy = numeric_strategy
        self.categorical_strategy = categorical_strategy
        self.random_state = random_state
        self.max_categories_for_ohe = max_categories_for_ohe
        self.min_samples_for_resampling = min_samples_for_resampling
        self.tune_parameters = tune_parameters
        self.hyperparameter_grids = hyperparameter_grids if hyperparameter_grids is not None else {}
        self.get_best_model = get_best_model 

        try:
            if self.file_path.endswith(".csv"):
                self.data = pd.read_csv(self.file_path)
            elif self.file_path.endswith((".xlsx", ".xls")):
                self.data = pd.read_excel(self.file_path)
            else:
                raise ValueError("Unsupported file format. Use CSV or Excel.")
            logger.info(f"Data loaded with shape: {self.data.shape}")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

        self.preprocess_config = dict(
            file_path=self.file_path,
            feature_selection=self.feature_selection,
            target_column=self.target_column,
            task=self.task,
            outlier_threshold=self.outlier_threshold,
            imbalance_threshold=self.imbalance_threshold,
            numeric_strategy=self.numeric_strategy,
            categorical_strategy=self.categorical_strategy,
            random_state=self.random_state,
            max_categories_for_ohe=self.max_categories_for_ohe,
            min_samples_for_resampling=self.min_samples_for_resampling
        )

    def _tune_model(self, model_instance, param_grid, X_train, y_train, cv=5):
        """Tune a single model using GridSearchCV and return the best estimator."""
        grid = GridSearchCV(
            model_instance,
            param_grid,
            cv=cv,
            scoring='accuracy' if self.task == 'classification' else 'neg_mean_squared_error'
        )
        grid.fit(X_train, y_train)
        logger.info(f"Best parameters for {model_instance.__class__.__name__}: {grid.best_params_}")
        return grid.best_estimator_

    def train(self) -> dict:
        """
        Execute the full pipeline: preprocessing, train/test split, model training (and tuning if enabled),
        and then saving the model.
        """
        if not self.preprocessed:
            try:
                result = preprocess_pipeline(**self.preprocess_config)
                logger.info("Preprocessing completed successfully!")
                self.data = result['data']
                logger.info(f"Data after preprocessing has shape: {self.data.shape}")
            except PreprocessingError as e:
                logger.error(f"Preprocessing failed: {e}")
                raise

        self.data.columns = (self.data.columns.str.strip()
                            .str.lower()
                            .str.replace(r'[^\w]', '_', regex=True))
        if self.target_column not in self.data.columns:
            raise ValueError(f"Target column '{self.target_column}' missing after preprocessing.")

        X = self.data.drop(columns=[self.target_column])
        y = self.data[self.target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=self.random_state)

        results = {}
        best_model = None
        best_model_name = None

        if self.task == 'classification':
            from sklearn.linear_model import LogisticRegression
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.svm import SVC

            models = {
                'LogisticRegression': LogisticRegression(**self.params.get('LogisticRegression', {})),
                'RandomForestClassifier': RandomForestClassifier(**self.params.get('RandomForestClassifier', {})),
                'SVC': SVC(**self.params.get('SVC', {}))
            }
        elif self.task == 'regression':
            from sklearn.linear_model import LinearRegression
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.svm import SVR

            models = {
                'LinearRegression': LinearRegression(**self.params.get('LinearRegression', {})),
                'RandomForestRegressor': RandomForestRegressor(**self.params.get('RandomForestRegressor', {})),
                'SVR': SVR(**self.params.get('SVR', {}))
            }
        else:
            raise ValueError("Task must be either 'classification' or 'regression'.")

        if self.get_best_model:
            if self.task == 'classification':
                best_score = -float("inf")  # higher is better
                for name, model_instance in models.items():
                    if self.tune_parameters and name in self.hyperparameter_grids:
                        model_instance = self._tune_model(model_instance, self.hyperparameter_grids[name], X_train, y_train)
                    model = model_instance.fit(X_train, y_train)
                    predictions = model.predict(X_test)
                    accuracy = accuracy_score(y_test, predictions)
                    report = classification_report(y_test, predictions)
                    results[name] = {'accuracy': accuracy, 'report': report}
                    logger.info(f"{name} Accuracy: {accuracy:.4f}")
                    if accuracy > best_score:
                        best_score = accuracy
                        best_model = model
                        best_model_name = name

            elif self.task == 'regression':
                best_score = float("inf")  # lower MSE is better
                for name, model_instance in models.items():
                    if self.tune_parameters and name in self.hyperparameter_grids:
                        model_instance = self._tune_model(model_instance, self.hyperparameter_grids[name], X_train, y_train)
                    model = model_instance.fit(X_train, y_train)
                    predictions = model.predict(X_test)
                    mse = mean_squared_error(y_test, predictions)
                    r2 = r2_score(y_test, predictions)
                    results[name] = {'mse': mse, 'r2': r2}
                    logger.info(f"{name} MSE: {mse:.4f}, R2 Score: {r2:.4f}")
                    if mse < best_score:
                        best_score = mse
                        best_model = model
                        best_model_name = name
        else:
            if self.model_name not in models:
                raise ValueError(f"Invalid model name '{self.model_name}' for {self.task}. Valid options: {list(models.keys())}")
            model_instance = models[self.model_name]
            if self.tune_parameters and self.model_name in self.hyperparameter_grids:
                model_instance = self._tune_model(model_instance, self.hyperparameter_grids[self.model_name], X_train, y_train)
            model = model_instance.fit(X_train, y_train)
            if self.task == 'classification':
                predictions = model.predict(X_test)
                accuracy = accuracy_score(y_test, predictions)
                report = classification_report(y_test, predictions)
                results[self.model_name] = {'accuracy': accuracy, 'report': report}
                logger.info(f"{self.model_name} Accuracy: {accuracy:.4f}")
            elif self.task == 'regression':
                predictions = model.predict(X_test)
                mse = mean_squared_error(y_test, predictions)
                r2 = r2_score(y_test, predictions)
                results[self.model_name] = {'mse': mse, 'r2': r2}
                logger.info(f"{self.model_name} MSE: {mse:.4f}, R2 Score: {r2:.4f}")
            best_model = model
            best_model_name = self.model_name

        def plot_learning_curve(estimator, X, y, cv=5, scoring='accuracy', title="Learning Curve"):
            """
            Generate and display a learning curve plot.
            """
            train_sizes, train_scores, test_scores = learning_curve(
                estimator,
                X,
                y,
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
                train_sizes=np.linspace(0.1, 1.0, 10)
            )
            train_scores_mean = np.mean(train_scores, axis=1)
            test_scores_mean = np.mean(test_scores, axis=1)
            
            plt.figure(figsize=(8, 6))
            plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
            plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
            plt.title(title)
            plt.xlabel("Number of Training Examples")
            plt.ylabel(scoring)
            plt.legend(loc="best")
            plt.grid(True)
            # plt.show()  

        plot = plot_learning_curve(best_model, X_train, y_train, cv=5, scoring='accuracy',
                                   title=f"Learning Curve for {best_model.__class__.__name__}")

        if self.get_best_model:
            model_file = f"{best_model_name}_model.pkl"
        else:
            model_file = f"{best_model_name}_model.pkl"
        try:
            with open(model_file, 'wb') as file:
                pickle.dump(best_model, file)
            logger.info(f"Model '{best_model_name}' saved to {model_file}")
        except Exception as e:
            logger.error(f"Failed to save the model '{best_model_name}': {e}")

        return {"results": results, "learning_curve_plot": plot, 'model' : best_model_name}

