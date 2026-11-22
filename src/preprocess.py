import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from dataclasses import dataclass
import pandas as pd
from sklearn.pipeline import Pipeline
from scipy.stats.mstats import trimmed_var
from sklearn.preprocessing import StandardScaler
from src.logging_config import logger
from src.utils import save_obj
import mlflow

@dataclass
class PreprocessConfig:
    """Configuration class containing paths for preprocessed data artifacts."""
    
    preprocessor_path: str = os.path.join("artifacts", "preprocessor.pkl")
    selected_features_path: str = os.path.join("artifacts", "selected_features.pkl")
    transformed_data_path: str = os.path.join("artifacts", "transformed_data.parquet")
    
class Preprocessor:
    """Handles feature selection, scaling, and preprocessing artifact storage."""
    
    def __init__(self):
        self.preprocess_config = PreprocessConfig() 
        self.high_var_columns = None
        self.pipeline = None
        
        
    def _validate_data(self, data: pd.DataFrame):
        """
        Validate the input DataFrame.

        Checks that the input is a non-empty pandas DataFrame,
        contains only numeric features, and has no missing values.
        """

        logger.info("Starting input data validation.")

        if not isinstance(data, pd.DataFrame):
            logger.error("Input data must be a pandas DataFrame.")
            raise TypeError("Input must be a pandas DataFrame.")

        if data.empty:
            logger.error("Input DataFrame is empty.")
            raise ValueError("Input DataFrame is empty.")

        if not all(
            pd.api.types.is_numeric_dtype(dtype)
            for dtype in data.dtypes
        ):
            logger.error("Input DataFrame contains non-numeric features.")
            raise ValueError("All features must be numeric.")

        if data.isnull().any().any():
            logger.error("Input DataFrame contains missing values.")
            raise ValueError("Input contains missing values.")

        logger.info("Input data validation completed successfully.")  
        
        
    def start_preprocessing(self,data: pd.DataFrame, trim_var: bool = True, n_features:int=2):
        """
        Perform feature selection and standardization on training data.

        Args:
            data: Input training DataFrame.
            trim_var: Whether to select features based on trimmed variance.
            n_features: Number of highest-variance features to select.

        Returns:
            pd.DataFrame: Scaled training data.
        """
        
        try:
            
            logger.info("Starting data preprocessing.")
            logger.info(f"Input data shape: {data.shape}")
            
            self._validate_data(data)
            
            if n_features <= 0 or n_features > data.shape[1]:
                logger.error(
                    f"Invalid n_features value: {n_features}"
                )
                raise ValueError(
                    "n_features must be positive and can not exceed "
                    "the number of available columns"
                )
        
            if trim_var:
                
                logger.info(f"Selecting the top {n_features} features using trimmed variance.")
                
                self.high_var_columns = data.apply(trimmed_var, limits=(0.1,0.1)).sort_values().tail(n_features).index.to_list()
            
            else:
                self.high_var_columns = data.columns.to_list()
                
            logger.info(f"Selected features: {self.high_var_columns}")    
            
            data = data[self.high_var_columns]  
                
            self.pipeline = Pipeline(steps=[
                ("scaler", StandardScaler())
            ])
            df_scaled = self.pipeline.fit_transform(data)
            
            os.makedirs(os.path.dirname(self.preprocess_config.transformed_data_path), exist_ok=True)
            save_obj(self.pipeline, self.preprocess_config.preprocessor_path)
            save_obj(self.high_var_columns, self.preprocess_config.selected_features_path)
            df_scaled = pd.DataFrame(df_scaled, columns=data.columns, index = data.index)
            df_scaled.to_parquet(self.preprocess_config.transformed_data_path)
            
            mlflow.log_param("trim_var", trim_var)
            mlflow.log_param("n_features", n_features)
            mlflow.log_param("selected_features", ",".join(self.high_var_columns))
            mlflow.log_artifact(self.preprocess_config.preprocessor_path)
            mlflow.log_artifact(self.preprocess_config.selected_features_path)
            
            logger.info(
                f"Preprocessing completed successfully. "
                f"Transformed data shape: {df_scaled.shape}"
            )
            return df_scaled
        except Exception:
            logger.exception("Error preprocessing data")
            raise
        
    def transform(self, data:pd.DataFrame):
        """
        Transform new data using the previously fitted preprocessing pipeline.

        Args:
            data: Input DataFrame to transform.

        Returns:
            pd.DataFrame: Scaled DataFrame containing selected features.
        """
        try:
            self._validate_data(data)
            if self.pipeline is None or self.high_var_columns is None:
                raise RuntimeError(
                    "Preprocessor must be fitted before transformation."
                )
                
            logger.info(
                f"Selecting previously selected features: "
                f"{self.high_var_columns}"
            )    
            df = data[self.high_var_columns]
            transformed = self.pipeline.transform(df)
            
            logger.info(
                f"Transformation completed successfully. "
                f"Output shape: {transformed.shape}"
            )
            return pd.DataFrame(
                                transformed,
                                columns=self.high_var_columns,
                                index=data.index
                            )
                            
        except Exception:
            logger.exception("Error transforming data")
            raise
        
   
        
      
        
        
        