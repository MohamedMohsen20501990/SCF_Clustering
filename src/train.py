import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd
from dataclasses import dataclass
from src.utils import save_obj
from src.logging_config import logger
from src.schemas import ModelTrainInputs
import mlflow
import mlflow.sklearn



@dataclass
class ModelTrainerConfig:
    """Store the trained KMeans model artifact path."""
    trained_model_path: str = os.path.join("artifacts", "model.pkl")
    
    
class ModelTrainer:
    """Train, evaluate, select, and save the best KMeans model."""
    
    def __init__(self):
        self.model_trainer_config: ModelTrainerConfig = ModelTrainerConfig()
        self.report: dict = {}
        self.models = {}
        self.best_model = None   
    
    def _validate_inputs(self,train_data: pd.DataFrame,n_clusters: int | None,max_clusters: int,) -> None:
        
        """Validate training data and cluster bounds prior to fitting."""
        ModelTrainInputs(train_data=train_data, n_clusters= n_clusters, max_clusters=max_clusters )
        
         
        
    def train_model(self,train_data:pd.DataFrame, n_clusters: int = None, max_clusters: int = 10):
        """
        Train KMeans using a user-defined or automatically selected
        number of clusters.

        Args:
            train_data (pd.DataFrame): Training features.
            n_clusters (int, optional): User-defined number of clusters.
                If None, the optimal number is selected automatically.
            max_clusters (int): Maximum number of clusters to evaluate.

        Returns:
            KMeans: The best trained KMeans model.

        Raises:
            Exception: If an error occurs during model training or saving.
        """
        
        try:
            self._validate_inputs(train_data, n_clusters, max_clusters)
            
            self.report= {}
            self.models = {}
            self.best_model = None   
            
            X = train_data.copy()
            
            if n_clusters is not None:
                logger.info(
                "Training KMeans with user-defined number of clusters: %d",
                n_clusters
            )
                model = KMeans(n_clusters= n_clusters, random_state=42, n_init=10)
                model.fit(X)
                labels = model.labels_
                self.report = {"n_clusters": n_clusters, "inertia score": model.inertia_, "silhouette_score": silhouette_score(X, labels)}
                self.best_model = model
                
                mlflow.log_param("n_clusters", n_clusters)
                mlflow.log_metric("inertia", model.inertia_)
                mlflow.log_metric("silhouette_score", self.report["silhouette_score"])
                
                logger.info(
                    "KMeans training completed successfully. "
                    "Inertia: %.4f | Silhouette Score: %.4f",
                    model.inertia_,
                    self.report["silhouette_score"]
                )
            
            else:
                
                logger.info(
                    "No number of clusters specified. "
                    "Starting automatic cluster selection from 2 to %d.",
                    max_clusters
                )
                
                for k in range(2, max_clusters + 1):
                    model = KMeans(n_clusters=k, random_state=42, n_init=10)
                    model.fit(X)
                    labels = model.labels_
                    key = f"model: {k} clusters"
                    self.report[key] = {"inertia_score" : model.inertia_, "silhouette_score": silhouette_score(X, model.labels_)}
                    self.models[key] = model
                    
                    mlflow.log_metric("inertia", model.inertia_, step=k)
                    mlflow.log_metric("silhouette_score", self.report[key]["silhouette_score"], step=k)
                    
                    logger.info(
                        "K=%d | Inertia: %.4f | Silhouette Score: %.4f",
                        k,
                        model.inertia_,
                        self.report[key]["silhouette_score"]
                    )

                logger.info(
                    "Selecting the best model based on the highest Silhouette Score."
                )
                    
                best_model_clusters,best_model_info = max(
                self.report.items(),
                key=lambda item: item[1]["silhouette_score"]
                )
                self.best_model = self.models[best_model_clusters]
                
                mlflow.log_param("best_k", self.best_model.n_clusters)
                
                logger.info(
                    "Best model selected: %s | Silhouette Score: %.4f",
                    best_model_clusters,
                    best_model_info["silhouette_score"])    
            save_obj(self.best_model, self.model_trainer_config.trained_model_path)
            mlflow.sklearn.log_model(self.best_model, name="model")
            
            logger.info(
                "Best model saved successfully to %s.",
                self.model_trainer_config.trained_model_path
            )

            logger.info("KMeans model training completed successfully.")
            
            return self.best_model
                 
                
            
        except Exception:
            logger.exception("Error training model")
            raise