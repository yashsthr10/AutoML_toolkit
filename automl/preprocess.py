import os
import logging
from typing import Dict, Union

import pandas as pd
import re
from automl.automl_pipeline import DataPreprocessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PreprocessingError(Exception):
    """Custom exception for errors during preprocessing."""
    pass


def validate_input_parameters(file_path: str, target_column: str, task: str) -> None:
    """Ensure the file exists and that the task is supported."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found at: {file_path}")
    if task not in ['classification', 'regression']:
        raise ValueError(f"Invalid task type: {task}. Must be 'classification' or 'regression'.")


def preprocess_pipeline(
    file_path: str,
    feature_selection: bool = True,
    target_column: str = 'target_column',
    task: str = 'classification',
    outlier_threshold: float = 3.0,
    imbalance_threshold: float = 2.0,
    numeric_strategy: str = 'median',
    categorical_strategy: str = 'most_frequent',
    random_state: int = 42,
    max_categories_for_ohe: int = 50,
    min_samples_for_resampling: int = 10,
    **kwargs
) -> Dict[str, Union[pd.DataFrame, dict]]:
    """
    Execute a robust automated preprocessing pipeline.
    Returns a dictionary with keys 'data' (processed DataFrame) and 'metadata'.
    """
    try:
        validate_input_parameters(file_path, target_column, task)
        logger.info(f"Loading data from {file_path}")

        processor = DataPreprocessor(
            file_path=file_path,
            scale_method='standard',
            strategy_for_num=numeric_strategy,
            strategy_for_cat=categorical_strategy,
            feature_selection=feature_selection
        )
        data = processor.load_data()

        data = data.replace(r'^\s*$', pd.NA, regex=True)
        original_shape = data.shape
        logger.info(f"Original data shape: {original_shape}")

        # target_column = target_column.strip().lower().replace(r'[^\w]', '_', regex=True)


        target_column = re.sub(r'[^\w]', '_', target_column.strip().lower())


        data.columns = data.columns.str.strip().str.lower().str.replace(r'[^\w]', '_', regex=True)

        target_column_std = target_column.strip().lower().replace(" ", "_")

        if data.empty:
            raise PreprocessingError("Input data is empty after loading.")
        if target_column not in data.columns:
            raise PreprocessingError(f"Target column '{target_column}' not found in dataset.")

        logger.info("Handling missing values...")
        processor.df = data.copy()
        processor.handle_missing_values()

        logger.info("Removing duplicates...")
        processor.remove_duplicates()
        logger.info(f"Data shape after deduplication: {processor.df.shape}")

        logger.info("Handling outliers...")
        pre_outlier_count = processor.df.shape[0]
        processor.handle_outliers(threshold=outlier_threshold)
        logger.info(f"Removed {pre_outlier_count - processor.df.shape[0]} outlier(s)")

        if processor.df.empty:
            raise PreprocessingError("Empty dataframe after cleaning steps")

        processor.df.columns = (processor.df.columns.str.strip()
                                .str.lower()
                                .str.replace(r'[^\w]', '_', regex=True))
        target_column_std = target_column.strip().lower().replace(" ", "_")
        if target_column_std not in processor.df.columns:
            raise PreprocessingError(f"Standardized target column '{target_column_std}' not found in data.")

        if task == 'classification':
            if processor.df[target_column_std].dtype == 'object':
                logger.info("Encoding target variable as categorical codes...")
                processor.df[target_column_std] = processor.df[target_column_std].astype('category').cat.codes
            elif processor.df[target_column_std].nunique() == 2:
                logger.info("Binary target detected; converting to int.")
                processor.df[target_column_std] = processor.df[target_column_std].astype(int)

        categorical_cols = processor.df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        if target_column_std in categorical_cols:
            categorical_cols.remove(target_column_std)
        if categorical_cols:
            logger.info(f"Processing {len(categorical_cols)} categorical feature(s)...")
            for col in categorical_cols:
                nunique = processor.df[col].nunique()
                if nunique > max_categories_for_ohe:
                    logger.warning(f"Column '{col}' has {nunique} unique values; dropping it.")
                    processor.df.drop(columns=[col], inplace=True)
                elif nunique == 1:
                    logger.warning(f"Removing constant column '{col}'.")
                    processor.df.drop(columns=[col], inplace=True)
            logger.info("Encoding categorical features...")
            processor.encode_categorical()
        else:
            logger.info("No categorical features to encode.")

        if task == 'classification':
            logger.info("Checking class balance...")
            class_counts = processor.df[target_column_std].value_counts()
            imbalance_ratio = class_counts.max() / class_counts.min()
            logger.info(f"Class distribution:\n{class_counts}")
            logger.info(f"Imbalance ratio: {imbalance_ratio:.2f}:1")
            if imbalance_ratio > imbalance_threshold and class_counts.min() >= min_samples_for_resampling:
                logger.info("Applying SMOTE to handle class imbalance...")
                X = processor.df.drop(columns=[target_column_std])
                y = processor.df[target_column_std]
                X_resampled, y_resampled = processor.handle_class_imbalance(X, y, target_column_std)
                processor.df = pd.concat(
                    [pd.DataFrame(X_resampled, columns=X.columns).reset_index(drop=True),
                     pd.DataFrame(y_resampled, columns=[target_column_std]).reset_index(drop=True)],
                    axis=1
                )
            elif class_counts.min() < min_samples_for_resampling:
                logger.warning("Insufficient samples in minority class for resampling.")

        X = processor.df.drop(columns=[target_column_std])
        y = processor.df[target_column_std]
        processor.df = X.copy()
        X_scaled = processor.scale_features().reset_index(drop=True)
        processor.df = pd.concat([X_scaled, y.reset_index(drop=True)], axis=1)

        if feature_selection:
            logger.info("Performing feature selection...")
            pre_sel_cols = processor.df.shape[1]
            processor.apply_feature_selection()
            post_sel_cols = processor.df.shape[1]
            logger.info(f"Features reduced from {pre_sel_cols} to {post_sel_cols}.")

        if processor.df.isna().sum().sum() > 0:
            raise PreprocessingError("Missing values detected in final dataset.")

        final_shape = processor.df.shape
        logger.info(f"Final data shape: {final_shape}")
        metadata = {
            'original_shape': original_shape,
            'final_shape': final_shape,
            'columns_removed': original_shape[1] - final_shape[1],
            'rows_removed': original_shape[0] - final_shape[0]
        }
        
        return {'data': processor.df, 'metadata': metadata}

    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        raise PreprocessingError(f"Critical error in preprocessing: {e}") from e
