import os 
import sys
from pathlib import Path
import mlflow
sys.path.append(str(Path(__file__).resolve().parent.parent))

from dataclasses import dataclass
import pandas as pd
from typing import Literal, TypeAlias
from src.logging_config import logger
from src.preprocess import Preprocessor
from src.train import ModelTrainer

@dataclass
class DataConfig:
    """Store the train data path."""
    
    train_data_path = os.path.join("artifacts", "train.csv")

class DataPipeline:
    """Handles dataset loading, filtering, and saving operations."""
    def __init__(self):
        self.data_config = DataConfig()
    
    def read_csv_data(self, path:str | Path,fearful:bool = True):
        """Read a CSV dataset and optionally filter fearful households.

        Args:
            path: Path to the CSV file to be loaded.
            fearful: If True, returns only households with TURNFEAR == 1.
                If False, returns the entire dataset.
                Defaults to True.

        Returns:
            pd.DataFrame: Loaded dataset, optionally filtered to include
                only credit-fearful households.

        Raises:
            Exception: If an error occurs while reading or processing
                the dataset.
        """
        
        try:
            df = pd.read_csv(path)
            
            if fearful:
                df=df[df["TURNFEAR"]==1]
                
            return df

        except Exception:
            logger.exception("Error reading data")
            raise
    
    def save_data(self, data:pd.DataFrame):
        
        """Save the entire dataset as a CSV file without splitting it.

        Args:
            data: Input pandas DataFrame to save.

        Returns:
            The original DataFrame.

        Raises:
            Exception: If an error occurs while saving the dataset.
        """
        try:
            os.makedirs(os.path.dirname(self.data_config.train_data_path),exist_ok=True)

            data.to_csv(self.data_config.train_data_path,index=False,header=True)

            logger.info(
                f"Data successfully saved\n"
                f"Total samples: {len(data)}"
            )

            return data

        except Exception:
            logger.exception("Error saving data")
            raise   

def main():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("scf_household_clustering")
    
    with mlflow.start_run(run_name="kmeans_fearful_households"):
            
        data_pipeline = DataPipeline()
        df = data_pipeline.read_csv_data("./data/raw/SCFP2019.csv", fearful=True)
        mlflow.log_param("n_rows", len(df)
                         )
        data_pipeline.save_data(df)
        preprocessor = Preprocessor()
        df_scaled = preprocessor.start_preprocessing(data= df, trim_var=True, n_features=5)
        model_trainer = ModelTrainer()
        model_trainer.train_model(train_data=df_scaled, n_clusters=4)
    
    
if __name__ == "__main__":
    main()    
        
            
            
                
     
            
            
                   