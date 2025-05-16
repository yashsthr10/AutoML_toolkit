"""Imports for Data Processor"""
import os
import re
import logging
import warnings
from typing import Dict
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from scipy.stats import zscore
from imblearn.over_sampling import SMOTE
import json
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator


# Configure logging and warnings
warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PreprocessingError(Exception):
    """Custom exception for preprocessing failures with detailed context"""
    def __init__(self, message: str, component: str = None):
        self.component = component
        super().__init__(f"{f'[{component}] ' if component else ''}{message}")


class DataProcessor:
    """
    initializing the constructor with necessary input parameters for flexible data processing.
    """
    
    def __init__(
        self,
        file_path: str,
        target_column: str,
        save_file: bool = False,
        scale_method: str = 'standard',
        numeric_strategy: str = 'median',
        categorical_strategy: str = 'most_frequent',
        max_categorical_cardinality: int = 50,
        safe_mode: bool = True
    ):
        """
        Initialize data processor with validation checks
        
        :param file_path: Path to input data file
        :param target_column: Name of target variable column
        :param save_file: Saving the processed data as a file 
        :param scale_method: Scaling method ('standard' or 'minmax')
        :param numeric_strategy: Strategy for numeric imputation
        :param categorical_strategy: Strategy for categorical imputation
        :param max_categorical_cardinality: Max unique values for categorical features
        :param safe_mode: Enable/disable destructive operations
        """
        self._validate_initial_params(file_path, target_column, scale_method)
        
        self.file_path = file_path
        self.original_target = target_column
        self.save_file = save_file
        self.scale_method = scale_method
        self.numeric_strategy = numeric_strategy
        self.categorical_strategy = categorical_strategy
        self.max_categories = max_categorical_cardinality
        self.safe_mode = safe_mode
        self.df = pd.DataFrame()
        self.target_column = self._standardize_name(target_column)
        self.scaler = self._init_scaler()
        self.encoders = {}
        self.metadata = {
            'original_shape': None,
            'processed_shape': None,
            'steps_applied': [],
            'warnings': []
        }

    def _validate_initial_params(self, file_path: str, target_column: str, scale_method: str):
        """Validate core initialization parameters"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
        if not isinstance(target_column, str) or len(target_column.strip()) == 0:
            raise ValueError("Target column must be a non-empty string")
        if scale_method not in ['standard', 'minmax']:
            raise ValueError(f"Invalid scale method: {scale_method}")

    def _standardize_name(self, name: str) -> str:
        """Standardize column names to snake_case format"""
        return re.sub(r'[^\w]+', '_', name.strip().lower()).strip('_')

    def _init_scaler(self):
        """Initialize appropriate feature scaler"""
        return StandardScaler() if self.scale_method == 'standard' else MinMaxScaler()

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
            '.xlsx': pd.read_excel,
            '.xls': pd.read_excel,
        }
        
        try:
            ext = os.path.splitext(self.file_path)[1].lower()
            if ext not in loaders:
                raise ValueError(f"Unsupported file format: {ext}")
            
            self.df = loaders[ext](self.file_path)
            self._post_load_processing()
            return self.df
            
        except Exception as e:
            logger.error(f"Data loading failed: {str(e)}")
            raise PreprocessingError(f"Data loading failed: {str(e)}", "DATA_LOAD") from e

    def _post_load_processing(self):
        """Performing initial data validation and standardization"""
        if self.df.empty:
            raise PreprocessingError("Loaded data is empty", "DATA_VALIDATION")
        
        # Standardize column names
        self.df.columns = [self._standardize_name(col) for col in self.df.columns]
        
        # Validate target column exists
        if self.target_column not in self.df.columns:
            raise PreprocessingError(
                f"Target column '{self.target_column}' not found in data columns:{self.df.columns}",
                "DATA_VALIDATION"
            )
        
        self.metadata['original_shape'] = self.df.shape
        self._check_data_integrity()

    def _check_data_integrity(self, check_na: bool = True, check_types: bool = True):
        """Perform basic data integrity checks"""
        if check_na and self.df.isna().all(axis=1).any():
            msg = "Rows with all missing values detected"
            if self.safe_mode:
                raise PreprocessingError(msg, "DATA_INTEGRITY")
            self.metadata['warnings'].append(msg)
            logger.warning(msg)

        if check_types:
            numeric_cols = self.df.select_dtypes(include=np.number).columns
            if len(numeric_cols) == 0:
                msg = "No numeric columns detected in dataset"
                self.metadata['warnings'].append(msg)
                logger.warning(msg)

    def handle_missing_values(self) -> pd.DataFrame:
        """
        missing value handling with strategy preservation using SimpleImputer() method
        """
        try:
            na_report = self.df.isna().sum()
            logger.info(f"Missing values pre-imputation:\n{na_report[na_report > 0]}")
            
            # Numeric imputation
            num_cols = self.df.select_dtypes(include=np.number).columns
            if num_cols.any():
                num_imputer = SimpleImputer(strategy=self.numeric_strategy)
                self.df[num_cols] = num_imputer.fit_transform(self.df[num_cols])
                self.metadata['num_imputer'] = num_imputer

            # Categorical imputation
            cat_cols = self.df.select_dtypes(exclude=np.number).columns
            if cat_cols.any():
                cat_imputer = SimpleImputer(strategy=self.categorical_strategy)
                self.df[cat_cols] = cat_imputer.fit_transform(self.df[cat_cols])
                self.metadata['cat_imputer'] = cat_imputer

            post_na = self.df.isna().sum().sum()
            if post_na > 0:
                msg = f"{post_na} missing values remain after imputation"
                if self.safe_mode:
                    raise PreprocessingError(msg, "MISSING_VALUES")
                self.metadata['warnings'].append(msg)
                logger.warning(msg)

            self.metadata['steps_applied'].append('missing_value_handling')
            return self.df

        except Exception as e:
            logger.error(f"Missing value handling failed: {str(e)}")
            raise PreprocessingError("Missing value handling failed", "MISSING_VALUES") from e

    def remove_duplicates(self) -> pd.DataFrame:
        """duplicate data handling with configurable safety checks"""
        try:
            pre_dedup = len(self.df)
            self.df = self.df.drop_duplicates().reset_index(drop=True)
            removed = pre_dedup - len(self.df)
            
            if removed > 0:
                logger.info(f"Removed {removed} duplicate rows")
                self.metadata['steps_applied'].append('deduplication')
                if removed / pre_dedup > 0.5 and self.safe_mode:
                    msg = "Over 50% of data removed as duplicates"
                    raise PreprocessingError(msg, "DUPLICATES")
            
            return self.df
            
        except Exception as e:
            logger.error("Duplicate removal failed")
            raise PreprocessingError("Duplicate removal failed", "DUPLICATES") from e

    def handle_outliers(self, threshold: float = 3.5) -> pd.DataFrame:
        """
        Robust outlier detection and handling using modified Z-score
        :param threshold: Z-score threshold for outlier detection
        """
        try:
            numeric_cols = self.df.select_dtypes(include=np.number).columns
            if len(numeric_cols) == 0:
                logger.warning("No numeric columns for outlier detection")
                return self.df

            z_scores = zscore(self.df[numeric_cols], nan_policy='omit')
            outliers = (np.abs(z_scores) > threshold).any(axis=1)
            
            if self.safe_mode:
                self.df = self.df[~outliers].reset_index(drop=True)
                logger.info(f"Removed {sum(outliers)} outliers")
            else:
                logger.info(f"Detected {sum(outliers)} outliers (not removed)")
            
            self.metadata['steps_applied'].append('outlier_handling')
            return self.df

        except Exception as e:
            logger.error("Outlier handling failed")
            raise PreprocessingError("Outlier handling failed", "OUTLIERS") from e

    def encode_categorical_features(self) -> pd.DataFrame:
        """categorical encoding with cardinality control"""
        try:
            cat_cols = self.df.select_dtypes(exclude=np.number).columns.tolist()
            if self.target_column in cat_cols:
                cat_cols.remove(self.target_column)

            for col in cat_cols:
                unique_count = self.df[col].nunique()
                
                if unique_count > self.max_categories:
                    if self.safe_mode:
                        self.df.drop(col, axis=1, inplace=True)
                        logger.info(f"Dropped high-cardinality column: {col}")
                    else:
                        logger.warning(f"High-cardinality column: {col} ({unique_count} categories)")
                    continue
                
                encoder = OneHotEncoder(
                    drop='if_binary', 
                    sparse_output=False,
                    handle_unknown='ignore'
                )
                encoded = encoder.fit_transform(self.df[[col]])
                encoded_cols = encoder.get_feature_names_out([col])
                
                self.df = pd.concat([
                    self.df.drop(col, axis=1),
                    pd.DataFrame(encoded, columns=encoded_cols)
                ], axis=1)
                
                self.encoders[col] = encoder

            self.metadata['steps_applied'].append('categorical_encoding')
            return self.df

        except Exception as e:
            logger.error("Categorical encoding failed")
            raise PreprocessingError("Categorical encoding failed", "ENCODING") from e

    def handle_class_imbalance(self, imbalance_ratio: float = 2.0) -> pd.DataFrame:
        """class imbalance handling with auto-detection"""
        try:
            if self.df[self.target_column].nunique() == 1:
                logger.warning("Single class in target column, No model training can be DONE")
                return self.df

            class_counts = self.df[self.target_column].value_counts()
            imbalance = class_counts.max() / class_counts.min()
            
            if imbalance > imbalance_ratio:
                logger.info(f"Addressing class imbalance (ratio: {imbalance:.1f}:1)")
                X = self.df.drop(self.target_column, axis=1)
                y = self.df[self.target_column]
                
                smote = SMOTE(sampling_strategy='auto', random_state=42)
                X_res, y_res = smote.fit_resample(X, y)
                
                self.df = pd.concat([
                    pd.DataFrame(X_res, columns=X.columns),
                    pd.Series(y_res, name=self.target_column)
                ], axis=1)
                
                self.metadata['steps_applied'].append('class_balance')
            
            return self.df

        except Exception as e:
            logger.error("Class balancing failed")
            raise PreprocessingError("Class balancing failed", "CLASS_BALANCE") from e

    def get_metadata(self) -> Dict:
        """Return processing metadata and statistics"""
        self.metadata['processed_shape'] = self.df.shape
        return self.metadata

    def validate_pipeline(self) -> bool:
        """Final validation check before model training"""
        checks = [
            (lambda: not self.df.empty, "Empty dataframe after processing"),
            (lambda: self.target_column in self.df.columns, "Missing target column"),
            (lambda: self.df[self.target_column].notna().all(), "NaN values in target"),
            (lambda: self.df.select_dtypes(exclude=np.number).empty, 
             "Remaining unprocessed categorical features")
        ]
        
        for check, msg in checks:
            if not check():
                if self.safe_mode:
                    raise PreprocessingError(msg, "VALIDATION")
                logger.warning(f"Validation warning: {msg}")
        
        logger.info("Pipeline validation passed")
        return True
    
    def _is_classification_task(self) -> bool:
        """Determine task type based on target characteristics"""
        target = self.df[self.target_column]
        
        # Explicit classification if <10 unique numeric values
        if target.nunique() <= 10 and pd.api.types.is_numeric_dtype(target):
            return True
            
        # Classification if string or categorical type
        return target.dtype in ['object', 'category']

    def _handle_regression_specifics(self):
        """Regression-specific preprocessing steps"""
        logger.info("Applying regression-specific processing")
        
        # Ensure target is numeric
        try:
            self.df[self.target_column] = pd.to_numeric(
                self.df[self.target_column],
                errors='coerce'
            )
            if self.df[self.target_column].isna().any():
                raise PreprocessingError("Regression target contains non-numeric values")
        except Exception as e:
            raise PreprocessingError(f"Regression target conversion failed: {e}")

    def _validate_classification_target(self):
        """Ensuring valid classification target"""
        unique_classes = self.df[self.target_column].nunique()
        if unique_classes < 2:
            raise PreprocessingError("Classification requires at least two classes")
        if unique_classes > 100:
            logger.warning(f"Unusual number of classes ({unique_classes}) for classification")

    def _handle_classification_specifics(self):
        """Enhanced classification handling with label mapping storage"""
        logger.info("Applying classification-specific processing")
        target_col = self.target_column
        
        if self.df[target_col].dtype == 'object':
            # Create categorical mapping metadata
            categories = self.df[target_col].unique().tolist()
            encoded_values = list(range(len(categories)))
            
            self.df[target_col] = (
                self.df[target_col]
                .astype('category')
                .cat.codes
            )
            
            # Store label mapping in metadata
            self.metadata['target_encoding'] = {
                'original_labels': categories,
                'encoded_values': encoded_values,
                'mapping': dict(zip(categories, encoded_values))
            }
            
        self.handle_class_imbalance()
        self._validate_classification_target()

    
    def encode_categorical_features(self) -> pd.DataFrame:
        """categorical encoding with metadata tracking"""
        try:
            cat_cols = self.df.select_dtypes(exclude=np.number).columns.tolist()
            if self.target_column in cat_cols:
                cat_cols.remove(self.target_column)

            encoding_metadata = {}
            
            for col in cat_cols:
                unique_count = self.df[col].nunique()
                
                if unique_count > self.max_categories:
                    if self.safe_mode:
                        self.df.drop(col, axis=1, inplace=True)
                        logger.info(f"Dropped high-cardinality column: {col}")
                    continue
                
                encoder = OneHotEncoder(
                    drop='if_binary', 
                    sparse_output=False,
                    handle_unknown='ignore'
                )
                encoded = encoder.fit_transform(self.df[[col]])
                encoded_cols = encoder.get_feature_names_out([col])
                
                # Store encoding metadata
                encoding_metadata[col] = {
                    'encoder_type': 'onehot',
                    'categories': encoder.categories_[0].tolist(),
                    'encoded_columns': encoded_cols.tolist(),
                    'dropped_category': encoder.drop_idx_ if encoder.drop_idx_ else None
                }
                
                self.df = pd.concat([
                    self.df.drop(col, axis=1),
                    pd.DataFrame(encoded, columns=encoded_cols)
                ], axis=1)
                
                self.encoders[col] = encoder

            # Update metadata with encoding information
            self.metadata['categorical_encodings'] = encoding_metadata
            self.metadata['steps_applied'].append('categorical_encoding')
            return self.df

        except Exception as e:
            logger.error("Categorical encoding failed")
            raise PreprocessingError("Categorical encoding failed", "ENCODING") from e

    def save_preprocessed_data(self, filename: str = None) -> str:
        """
        save method for saving processed data inside the "processed_data" directory
        """
        try:
            output_dir = os.path.join(os.getcwd(), 'processed_data')
            os.makedirs(output_dir, exist_ok=True)
            
            base_name = os.path.basename(self.file_path)
            file_name, _ = os.path.splitext(base_name)
            
            if not filename:
                filename = f"processed_{file_name}.csv"
            
            output_path = os.path.join(output_dir, filename)
            metadata_path = os.path.join(output_dir, f"metadata_{file_name}.json")

            # Save data
            self.df.to_csv(output_path, index=False)

            # Custom serializer for sklearn components
            def convert(o):
                """Handling sklearn estimators and other complex types"""
                if isinstance(o, np.generic):
                    return o.item()
                if isinstance(o, pd.Timestamp):
                    return o.isoformat()
                if hasattr(o, 'dtype') and pd.api.types.is_extension_array_dtype(o):
                    return o.tolist()
                if isinstance(o, BaseEstimator):  # Handle sklearn estimators
                    return {
                        '__class__': o.__class__.__name__,
                        'params': o.get_params()
                    }
                if isinstance(o, ColumnTransformer) or isinstance(o, Pipeline):
                    return str(o)  # Store string representation
                if isinstance(o, (list, tuple)):
                    return [convert(i) for i in o]
                if isinstance(o, dict):
                    return {k: convert(v) for k, v in o.items()}
                return o

            # Create safe metadata copy
            safe_metadata = json.loads(json.dumps(self.metadata, default=convert))
            
            # Save metadata
            with open(metadata_path, 'w') as f:
                json.dump(safe_metadata, f, indent=2, ensure_ascii=False)

            logger.info(f"Data and metadata saved to:\n{output_path}\n{metadata_path}")
            return output_path

        except Exception as e:
            logger.error(f"Failed to save processed data: {str(e)}")
            raise PreprocessingError(f"Data saving failed: {str(e)}", "DATA_SAVE") from e
        
    def scale_features(self) -> pd.DataFrame:
        """Safe feature scaling with validation checks"""
        try:
            # Exclude target column from scaling
            numeric_cols = self.df.select_dtypes(include=np.number).columns.drop(self.target_column, errors='ignore')
            if len(numeric_cols) == 0:
                logger.warning("No numeric columns to scale")
                return self.df

            self.scaler.fit(self.df[numeric_cols])
            self.df[numeric_cols] = self.scaler.transform(self.df[numeric_cols])
            
            self.metadata['steps_applied'].append('feature_scaling')
            return self.df

        except Exception as e:
            logger.error("Feature scaling failed")
            raise PreprocessingError("Feature scaling failed", "SCALING") from e

    def fit_transform(self) -> pd.DataFrame:
        """
        Complete end-to-end preprocessing pipeline execution with manual-saving
        """
        try:
            self.load_data()
            self.handle_missing_values()
            self.remove_duplicates()
            self.handle_outliers()
            self.encode_categorical_features()
            self.scale_features()  # Moved before handling specifics
            if self._is_classification_task():
                self._handle_classification_specifics()
            else:
                self._handle_regression_specifics()
            self.validate_pipeline()

            if self.save_file == True:
                self.save_preprocessed_data()
            
            logger.info("Preprocessing completed successfully")
            return self.df

        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}", exc_info=True)
            raise PreprocessingError(f"Fit-transform error: {str(e)}") from e

# Correct the example block
if __name__ == '__main__':
    processor = DataProcessor(
            file_path='gender.csv',
            target_column='gender',
            scale_method='minmax'
        )

    try:
        processed_df = processor.fit_transform()
        metadata = processor.get_metadata()
        print(f"Saved to: {metadata['save_path']}")
    except PreprocessingError as e:
        print(f"Processing failed: {e}")