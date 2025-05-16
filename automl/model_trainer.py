"""Imports for Model Trainer"""
import logging
import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from automl.Processor import PreprocessingError, DataProcessor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, ElasticNet
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from xgboost import XGBClassifier, XGBRegressor 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score , r2_score, f1_score, mean_squared_error
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import cross_val_score
import warnings

# Configure logging and warnings
warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""Creating the model trainer class"""
class ModelTrainer:
    def __init__(self,
                 file_path: str,
                 model_name: str,
                 target_column: str,
                 task: str = 'classification',
                 preprocessed: bool = False,
                 train_to_test_ratio: float = 0.33,
                 save_file: bool = False,
                 scale_method: str = 'standard',
                 numeric_strategy: str = 'median',
                 categorical_strategy: str = 'most_frequent',
                 max_categorical_cardinality: int = 50,
                 custom_hyperparams: dict = None,
                 use_optuna: bool = False,
                 optuna_trials: int = 50,
                 optuna_timeout: int = 3600,
                 optuna_metrics: list = ['default']):
        

        """Initialize the training pipeline with enhanced validation"""
        self.file_path = file_path
        self.target_column = target_column
        self.model_name = model_name
        self.task = task
        self.train_to_test_ratio = train_to_test_ratio
        self.save_file = save_file
        self.scale_method = scale_method
        self.numeric_strategy = numeric_strategy
        self.categorical_strategy = categorical_strategy
        self.max_categories = max_categorical_cardinality
        self.custom_hyperparams = custom_hyperparams or {}
        self.best_model = None
        self.preprocessed = preprocessed
        self.metadata = {}
        self.random_state = 42
        self.feature_names = []
        self.use_optuna = use_optuna
        self.optuna_trials = optuna_trials
        self.optuna_timeout = optuna_timeout
        self.optuna_metrics = optuna_metrics


    """Validation function for validating all the input parameters"""
    def _validate_parameters(self) -> None:
        """
        Validating all the input configuration according to their Input types, Data types 
        and validating supported model names
        """
        # Parameter existence check
        if not hasattr(self, 'file_path'):
            raise AttributeError("Missing required parameter: file_path")
        if not hasattr(self, 'target_column'):
            raise AttributeError("Missing required parameter: target_column")

        # Type and value validation
        if not isinstance(self.file_path, str):
            raise TypeError(f"file_path must be string, got {type(self.file_path)}")
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Data file not found: {self.file_path}")
        if not self.file_path.lower().endswith(('.csv', '.parquet', '.feather')):
            raise ValueError("Supported formats: CSV, Parquet, Feather")

        if not isinstance(self.target_column, str):
            raise TypeError(f"target_column must be string, got {type(self.target_column)}")
        
         # Train-test split validation
        if not isinstance(self.train_to_test_ratio, (float, int)):
            raise TypeError("train_to_test_ratio must be numeric")
        if not (0.01 <= self.train_to_test_ratio <= 0.99):
            raise ValueError("train_to_test_ratio must be between 0.01-0.99")

        # Boolean flag validation
        if not isinstance(self.save_file, bool):
            raise TypeError("save_file must be boolean")

        # Scaling method validation
        valid_scalers = ['standard', 'minmax', 'maxabs', 'robust', None]
        if self.scale_method not in valid_scalers:
            raise ValueError(f"Invalid scale_method: {self.scale_method}")

        # Imputation strategies
        num_strategies = ['mean', 'median', 'most_frequent', 'constant']
        if self.numeric_strategy not in num_strategies:
            raise ValueError(f"Invalid numeric_strategy: {self.numeric_strategy}")

        cat_strategies = ['most_frequent', 'constant']
        if self.categorical_strategy not in cat_strategies:
            raise ValueError(f"Invalid categorical_strategy: {self.categorical_strategy}")

        # Cardinality check
        if not isinstance(self.max_categories, int):
            raise TypeError("max_categorical_cardinality must be integer")
        if not (1 <= self.max_categories <= 1000):
            raise ValueError("max_categorical_cardinality must be 1-1000")

        # Existing validations
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Data file not found: {self.file_path}")
            
        valid_tasks = ['classification', 'regression']
        if self.task not in valid_tasks:
            raise ValueError(f"Invalid task: {self.task}. Valid options: {valid_tasks}")

        # Model validation
        valid_models = {
            'classification': ['RandomForest', 'XGBoost', 'LogisticRegression', 
                              'SVM', 'DecisionTree', 'KNN', 'GradientBoosting'],
            'regression': ['LinearRegression', 'RandomForest', 'XGBoost',
                          'SVR', 'ElasticNet', 'GradientBoosting', 'KNN']
        }
        if self.model_name not in valid_models[self.task]:
            raise ValueError(f"Invalid model '{self.model_name}' for {self.task} task")

        # Hyperparameter type validation
        model_class = self._get_model_class()
        valid_params = model_class().get_params().keys()
        
        for param in self.custom_hyperparams:
            if param not in valid_params:
                raise ValueError(f"Invalid hyperparameter '{param}' for {self.model_name}")
                
        # Add specific hyperparam value checks here if needed
    
    def _optuna_objective(self, trial, X, y, default_params: dict) -> float:
        """Optuna optimization objective function"""
        try:
            params = self._define_optuna_search_space(trial, default_params)
            model = self._get_model_class()(**params)
            
            if 'default' in self.optuna_metrics:
                scoring = 'accuracy' if self.task == 'classification' else 'r2'
            else:
                scoring = self.optuna_metrics[0]

            scores = cross_val_score(
                model, X, y,
                cv=5,
                scoring=scoring,
                n_jobs=-1
            )
            return np.mean(scores)
            
        except Exception as e:
            logger.error(f"Trial failed: {str(e)}")
            return float('-inf') if self.task == 'classification' else float('inf')


    def _create_optuna_study(self) -> optuna.Study:
        """Initialize and configuring Optuna study method"""
        direction = 'maximize' if self.task == 'classification' else 'minimize'
        sampler = TPESampler(seed=self.random_state)
        return optuna.create_study(direction=direction, sampler=sampler)
    
    def _define_optuna_search_space(self, trial, default_params: dict) -> dict:
        """Defining hyperparameter search space for different available models"""
        params = default_params.copy()
        
        # RandomForest
        if self.model_name == 'RandomForest':
            params.update({
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
            })
        
        # XGBoost
        elif self.model_name == 'XGBoost':
            params.update({
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'gamma': trial.suggest_float('gamma', 0.0, 1.0)
            })
        
        # Logistic Regression
        elif self.model_name == 'LogisticRegression':
            params.update({
                'C': trial.suggest_float('C', 1e-4, 1e4, log=True),
                'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
                'solver': trial.suggest_categorical('solver', ['liblinear', 'saga']),
                'max_iter': trial.suggest_int('max_iter', 100, 1000)
            })
        
        elif self.model_name == 'SVM':
            kernel = trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf'])
            svm_params = {
                'C': trial.suggest_float('C', 1e-4, 1e4, log=True),
                'kernel': kernel,
                'degree': trial.suggest_int('degree', 2, 5) if kernel == 'poly' else 3
            }
            
            # SVR-specific parameter
            if self.model_name == 'SVR':
                svm_params['epsilon'] = trial.suggest_float('epsilon', 0.01, 1.0)
            
            params.update(svm_params)
                        
        # Decision Tree
        elif self.model_name == 'DecisionTree':
            params.update({
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
                'criterion': trial.suggest_categorical('criterion', 
                    ['gini', 'entropy'] if self.task == 'classification' 
                    else ['squared_error', 'friedman_mse'])
            })
        
        # K-Nearest Neighbors
        elif self.model_name == 'KNN':
            params.update({
                'n_neighbors': trial.suggest_int('n_neighbors', 1, 30),
                'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
                'p': trial.suggest_int('p', 1, 2)
            })
        
        # Gradient Boosting
        elif self.model_name == 'GradientBoosting':
            params.update({
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
            })
        
        # ElasticNet Regression
        elif self.model_name == 'ElasticNet':
            params.update({
                'alpha': trial.suggest_float('alpha', 1e-4, 1e4, log=True),
                'l1_ratio': trial.suggest_float('l1_ratio', 0.0, 1.0),
                'max_iter': trial.suggest_int('max_iter', 100, 1000),
                'tol': trial.suggest_float('tol', 1e-5, 1e-3, log=True)
            })
        
        # Linear Regression
        elif self.model_name == 'LinearRegression':
            params.update({
                'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False]),
                'positive': trial.suggest_categorical('positive', [True, False])
            })
        
        return params

    def _get_model_class(self):
        """Get model class with hyperparameter validation"""
        model_map = {
            'classification': {
                'RandomForest': RandomForestClassifier,
                'XGBoost': XGBClassifier,
                'LogisticRegression': LogisticRegression,
                'SVM': SVC,
                'DecisionTree': DecisionTreeClassifier,
                'KNN': KNeighborsClassifier,
                'GradientBoosting': GradientBoostingClassifier
            },
            'regression': {
                'RandomForest': RandomForestRegressor,
                'XGBoost': XGBRegressor,
                'LinearRegression': LinearRegression,
                'SVR': SVR,
                'ElasticNet': ElasticNet,
                'GradientBoosting': GradientBoostingRegressor,
                'KNN': KNeighborsRegressor
            }
        }
        return model_map[self.task][self.model_name]

    def _initialize_model(self):
        """Initialize model with custom hyperparameters"""
        model_class = self._get_model_class()
        return model_class(**self.custom_hyperparams, random_state=self.random_state)

    def load_data(self) -> pd.DataFrame:
        """
        Load data from supported file formats with validation
        
        Supported formats(as of now):
        - CSV (.csv)
        - Excel (.xlsx, .xls)
        
        :return: Loaded DataFrame
        """
        loaders = {
            '.csv': pd.read_csv,
            '.xlsx': pd.read_excel
        }
        try:
            ext = os.path.splitext(self.file_path)[1].lower()
            if ext not in loaders:
                raise ValueError(f"Unsupported file format: {ext}")
            
            self.df = loaders[ext](self.file_path)
            if self.target_column not in self.df.columns:
                raise ValueError(f"Target column {self.target_column} not found")
            else:
                return self.df
            
        except Exception as e:
            logger.error(f"Data loading failed: {str(e)}")
            raise PreprocessingError(f"Data loading failed: {str(e)}", "DATA_LOAD") from e
        

    def load_and_preprocess_data(self) -> pd.DataFrame:
        """
        Load and preprocess data using DataProcessor
        """
        # Creating an instance of DataProcessor class
        processor = DataProcessor(
            file_path=self.file_path,
            target_column=self.target_column,
            save_file=self.save_file,  
            scale_method=self.scale_method,
            numeric_strategy=self.numeric_strategy,
            categorical_strategy=self.categorical_strategy,
            max_categorical_cardinality=self.max_categories,
            safe_mode=True
        )

        """
        Initializing the processing and updating the self.df with preprocessed data,
        also saving the preprocessed data( if save_data is set True)
        """
        try:
            processed_df = processor.fit_transform()
            metadata = processor.get_metadata()
            print(f"Saved to: {metadata.get('save_path', 'No save path')}")
            self.df = processed_df
            return self.df
        except PreprocessingError as e:
            print(f"Processing failed: {e}")
            raise

    def train(self):
        """
        Creating the complete training pipeline in one class instance:
        1. Validating the parameters 
        2. Loading and preprocessing the data if the data is unprocessed
        3. Training the desired model according to the given task 
        4. Storing the model and metadata
        
        """
        try:
            self._validate_parameters()
            if self.preprocessed == False:
                logger.info('Data is not processed, Processing the unprocessed data...')
                df = self.load_and_preprocess_data()
            else:
                df = self.load_data()
            X = df.drop(columns=[self.target_column])
            y = df[self.target_column]
            self.feature_names = X.columns.tolist()

            # Split data with stratification
            stratify = y if self.task == 'classification' else None
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.train_to_test_ratio,
                stratify=stratify,
                random_state=self.random_state
            )

            # Hyperparameter optimization
            best_params = self.custom_hyperparams.copy()
            if self.use_optuna:
                logger.info("Starting hyperparameter optimization with Optuna...")
                study = self._create_optuna_study()
                study.optimize(
                    lambda trial: self._optuna_objective(trial, X_train, y_train, best_params),
                    n_trials=self.optuna_trials,
                    timeout=self.optuna_timeout
                )
                logger.info(f"Best trial value: {study.best_trial.value:.4f}")
                best_params.update(study.best_trial.params)

            # Final model training
            model = self._get_model_class()(**best_params)
            model.fit(X_train, y_train)
            
            # Calculate predictions and metrics
            train_preds = model.predict(X_train)
            test_preds = model.predict(X_test)


            # Store metadata
            self.best_model = model
            self.metadata = {
                'hyperparameters': model.get_params(),
                'features': self.feature_names,
                'optuna_params': best_params if self.use_optuna else None,
                'metrics': {}
            }

            # Calculate metrics
            if self.task == 'classification':
                self.metadata['metrics'].update({
                    'train_accuracy': accuracy_score(y_train, train_preds),
                    'test_accuracy': accuracy_score(y_test, test_preds),
                    'train_f1': f1_score(y_train, train_preds, average='weighted'),
                    'test_f1': f1_score(y_test, test_preds, average='weighted')
                })
            else:
                self.metadata['metrics'].update({
                    'train_r2': r2_score(y_train, train_preds),
                    'test_r2': r2_score(y_test, test_preds),
                    'train_rmse': np.sqrt(mean_squared_error(y_train, train_preds)),
                    'test_rmse': np.sqrt(mean_squared_error(y_test, test_preds))
                })
            
            # Log results
            logger.info("Training results:")
            for metric, value in self.metadata['metrics'].items():
                logger.info(f"  - {metric}: {value:.4f}")

            return self.best_model, self.metadata

        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise

    def save_model(self, path='saved_models'):
        """Saving model with metadata"""
        os.makedirs(path, exist_ok=True)
        model_path = os.path.join(path, 'model.pkl')
        
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': self.best_model,
                'metadata': self.metadata
            }, f)
        
        logger.info(f"Model saved to {model_path}")
        return model_path

    def plot_feature_importance(self, top_n=20):
        """Feature importance visualization"""
        if not hasattr(self.best_model, 'feature_importances_'):
            logger.warning("Model doesn't support feature importance")
            return

        importances = self.best_model.feature_importances_
        indices = np.argsort(importances)[-top_n:]
        
        plt.figure(figsize=(10, 8))
        plt.title("Feature Importances")
        plt.barh(range(top_n), importances[indices], align='center')
        plt.yticks(range(top_n), [self.feature_names[i] for i in indices])
        plt.xlabel("Relative Importance")
        plt.tight_layout()
        return plt
    

"""Testing starts from here"""
if __name__ == '__main__':
    """testing all the model using for-loops """

    # testing classification models
    all_classification_models = ['RandomForest','LogisticRegression','DecisionTree','SVM', 'KNN', 'GradientBoosting', 'XGBoost']

    # Initializing the for-loop for classification model testing
    for models in all_classification_models:
        trainer = ModelTrainer(
        file_path='test_data/Crop_recommendation.csv',
        model_name=models,
        target_column='label',
        task='classification',
        use_optuna=True,
        optuna_trials=10
        )
        model, metadata = trainer.train()
        trainer.save_model()
        print(f"-----Training of Model {models} is completed without error------")


    # initialzing the for-loop for testing all the regression models
    all_regression_models = ['RandomForest','LinearRegression','ElasticNet','SVR', 'KNN', 'GradientBoosting', 'XGBoost']

    # Initializing the for-loop for regression model testing
    for models in all_regression_models:
        trainer = ModelTrainer(
        file_path='test_data/Student_Performance.csv',
        model_name=models,
        target_column='performance_index',
        task='regression',
        use_optuna=True,
        optuna_trials=10
        )
        model, metadata = trainer.train()
        trainer.save_model()
        print(f"-----Training of Model {models} is completed without error------")

    # trainer.plot_feature_importance().show()