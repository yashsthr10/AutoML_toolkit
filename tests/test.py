from automl.model_trainer import ModelTrainer
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

if __name__ == '__main__':
    params={
            'RandomForestClassifier': {'n_estimators': 50, 'random_state': 42},
            'LogisticRegression': {'max_iter': 100},
            'SVC': {'kernel': 'rbf'},
            'LinearRegression': {},
            'RandomForestRegressor': {'n_estimators': 50, 'random_state': 42},
            'SVR': {}
        }
    hyperparameter_grids={
            'LogisticRegression': {
            'C': [0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear']  
            },
            'RandomForestClassifier': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
            },
            'SVC': {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto'],
            'kernel': ['linear', 'rbf']
            }
        }
    print("\n=== Testing: User Input Model Only (SVC) ===")
    trainer_user_model = ModelTrainer(
        model='SVC',
        params=params,
        file_path='D:/Resume/PROJECT_3/Backend/automl_toolkit/tests/test_data/drug200.csv',
        target_column='Drug',
        task='classification',
        preprocessed=False,
        get_best_model=False,
        tune_parameters=True,
        hyperparameter_grids=hyperparameter_grids
    )
    results_user_model = trainer_user_model.train()
    print("User Input Model Only Results:")
    print(results_user_model['results'])

    # --- Test 2: Automatically select the best model ---
    print("\n=== Testing: Get Best Model ===")
    trainer_best_model = ModelTrainer(
        model='SVC', 
        params=params,
        file_path='D:/Resume/PROJECT_3/Backend/automl_toolkit/tests/test_data/drug200.csv',
        target_column='Drug',
        task='classification',
        preprocessed=False,
        get_best_model=True,
        tune_parameters=True,
        hyperparameter_grids=hyperparameter_grids
    )
    results_best_model = trainer_best_model.train()
    print("Best Model Selection Results:")
    print(results_best_model['results'])