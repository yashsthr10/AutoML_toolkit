import logging
import os
import re
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from scipy.stats import zscore
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataPreprocessor:
    def __init__(self,
                 file_path: str,
                 scale_method: str = 'standard',
                 handle_imbalance: bool = True,
                 feature_selection: bool = True,
                 n_components: Optional[int] = None,
                 strategy_for_num: str = 'mean',
                 strategy_for_cat: str = 'most_frequent'):
        """
        Initialize with file path and configuration options.
        """
        self.file_path = file_path
        self.scale_method = scale_method.lower()
        self.handle_imbalance = handle_imbalance
        self.feature_selection = feature_selection
        self.n_components = n_components
        self.strategy_for_num = strategy_for_num
        self.strategy_for_cat = strategy_for_cat
        self.encoders = {}  
        self.df = None

        # Set up scaler based on chosen method.
        if self.scale_method == 'standard':
            self.scaler = StandardScaler()
        elif self.scale_method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError("Unsupported scale_method. Use 'standard' or 'minmax'.")

    def load_data(self) -> pd.DataFrame:
        """Load dataset from file (CSV or Excel)."""
        try:
            if self.file_path.endswith(".csv"):
                self.df = pd.read_csv(self.file_path)
            elif self.file_path.endswith((".xlsx", ".xls")):
                self.df = pd.read_excel(self.file_path)
            else:
                raise ValueError("Unsupported file format. Use CSV or Excel.")
            logger.info(f"Data loaded with shape: {self.df.shape}")
            return self.df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def handle_missing_values(self) -> pd.DataFrame:
        """
        Impute missing values:
          - Numeric columns using the specified strategy.
          - Categorical columns using the specified strategy.
        """
        df = self.df.copy()
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        if numeric_cols:
            try:
                numeric_imputer = SimpleImputer(strategy=self.strategy_for_num)
                df[numeric_cols] = numeric_imputer.fit_transform(df[numeric_cols])
            except Exception as e:
                logger.error(f"Numeric imputation failed: {e}")
                raise

        if categorical_cols:
            try:
                categorical_imputer = SimpleImputer(strategy=self.strategy_for_cat)
                df[categorical_cols] = categorical_imputer.fit_transform(df[categorical_cols])
            except Exception as e:
                logger.error(f"Categorical imputation failed: {e}")
                raise

        logger.info("Missing values handled.")
        self.df = df
        return df

    def remove_duplicates(self) -> pd.DataFrame:
        """Remove duplicate rows."""
        before = self.df.shape[0]
        self.df = self.df.drop_duplicates()
        after = self.df.shape[0]
        logger.info(f"Removed {before - after} duplicate row(s).")
        return self.df

    def handle_outliers(self, threshold: float = 3) -> pd.DataFrame:
        """
        Remove rows with any numeric value having a z-score (absolute) >= threshold.
        """
        df = self.df.copy()
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        if numeric_cols:
            try:
                z_scores = df[numeric_cols].apply(zscore)
                df = df[(np.abs(z_scores) < threshold).all(axis=1)]
                logger.info(f"Outlier removal complete. New shape: {df.shape}")
            except Exception as e:
                logger.error(f"Outlier handling failed: {e}")
                raise
        else:
            logger.warning("No numeric columns found for outlier detection.")
        self.df = df
        return df

    def encode_categorical(self) -> pd.DataFrame:
        """
        Encode categorical columns:
          - Binary columns with LabelEncoder.
          - Multi-class columns with OneHotEncoder.
        """
        df = self.df.copy()
        cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        for col in cat_cols:
            unique_vals = df[col].nunique()
            try:
                if unique_vals <= 2:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col])
                    self.encoders[col] = le
                else:
                    ohe = OneHotEncoder(sparse_output=False, drop='first')
                    transformed = ohe.fit_transform(df[[col]])
                    ohe_cols = [f"{col}_{i}" for i in range(transformed.shape[1])]
                    df = df.drop(columns=[col])
                    df[ohe_cols] = pd.DataFrame(transformed, index=df.index, columns=ohe_cols)
                    self.encoders[col] = ohe
            except Exception as e:
                logger.error(f"Encoding failed for column {col}: {e}")
                raise
        logger.info("Categorical encoding complete.")
        self.df = df
        return df

    def scale_features(self) -> pd.DataFrame:
        """
        Scale numerical features using the chosen scaler.
        """
        df = self.df.copy()
        num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        if num_cols:
            try:
                scaled = self.scaler.fit_transform(df[num_cols])
                df[num_cols] = pd.DataFrame(scaled, columns=num_cols, index=df.index)
                logger.info("Feature scaling complete.")
            except Exception as e:
                logger.error(f"Scaling failed: {e}")
                raise
        else:
            logger.warning("No numeric columns found for scaling.")
        self.df = df
        return df

    def handle_class_imbalance(self, X: pd.DataFrame, y: pd.Series, target_column: str):
        """
        Use SMOTE to balance the classes.
        """
        if X.isna().sum().sum() > 0:
            logger.warning("Missing values detected in predictors; applying mean imputation.")
            X = pd.DataFrame(SimpleImputer(strategy='mean').fit_transform(X), columns=X.columns)
        if y.isna().sum() > 0:
            logger.warning(f"Missing values in target '{target_column}'; filling with mode.")
            y = y.fillna(y.mode()[0])
        try:
            smote = SMOTE(sampling_strategy='auto', random_state=42)
            X_res, y_res = smote.fit_resample(X, y)
            logger.info("Class imbalance handled using SMOTE.")
        except Exception as e:
            logger.error(f"SMOTE resampling failed: {e}")
            raise
        return X_res, y_res

    def apply_feature_selection(self) -> pd.DataFrame:
        """
        Optionally apply PCA if n_components is specified.
        """
        df = self.df.copy()
        if self.n_components:
            try:
                numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
                pca = PCA(n_components=self.n_components)
                df_pca = pca.fit_transform(df[numeric_cols])
                df_pca = pd.DataFrame(df_pca, index=df.index)
                non_numeric = df.drop(columns=numeric_cols)
                df = pd.concat([df_pca, non_numeric], axis=1)
                logger.info(f"PCA applied: features reduced to {self.n_components} component(s).")
            except Exception as e:
                logger.error(f"PCA feature selection failed: {e}")
                raise
        self.df = df
        return df

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text by lowercasing, removing non-alphanumeric characters,
        tokenizing, and removing stopwords.
        """
        try:
            text = text.lower()
            text = re.sub(r'[^a-z0-9]', ' ', text)
            tokens = word_tokenize(text)
            tokens = [word for word in tokens if word not in stopwords.words('english')]
            return ' '.join(tokens)
        except Exception as e:
            logger.error(f"Text preprocessing failed: {e}")
            return text

    def fit_transform(self, target_column: Optional[str] = None) -> pd.DataFrame:
        """
        Run the full preprocessing pipeline. If a target_column is provided,
        optionally handle class imbalance.
        """
        try:
            self.load_data()
            if self.df.empty:
                raise ValueError("Loaded data is empty.")

            self.handle_missing_values()
            self.remove_duplicates()
            self.handle_outliers()
            self.encode_categorical()
            self.scale_features()

            if self.feature_selection:
                self.apply_feature_selection()

            if self.handle_imbalance and target_column:
                if target_column not in self.df.columns:
                    raise ValueError(f"Target column '{target_column}' not found.")
                X = self.df.drop(columns=[target_column])
                y = self.df[target_column]
                X_res, y_res = self.handle_class_imbalance(X, y, target_column)
                self.df = pd.concat([pd.DataFrame(X_res, columns=X.columns, index=X.index),
                                     pd.Series(y_res, name=target_column, index=X.index)], axis=1)

            if self.df.isna().sum().sum() > 0:
                raise ValueError("Final dataset contains missing values.")

            logger.info(f"Final data shape after preprocessing: {self.df.shape}")
            return self.df
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            raise

    def decode_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Decode labels using stored encoders.
        """
        decoded_df = df.copy()
        for col, encoder in self.encoders.items():
            if col in decoded_df.columns:
                try:
                    decoded_df[col] = encoder.inverse_transform(decoded_df[col])
                except Exception as e:
                    logger.warning(f"Could not decode column {col}: {e}")
        return decoded_df
