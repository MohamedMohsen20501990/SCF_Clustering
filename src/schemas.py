from __future__ import annotations

import pandas as pd
from pydantic import BaseModel, ConfigDict, field_validator, model_validator, Field, FiniteFloat


class ModelTrainInputs(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    train_data: pd.DataFrame  # is not an ordinary type for pydantic and that's why i am using arbitary_types_allowes = True
    n_clusters: int | None = None
    max_clusters: int = 10

    @field_validator("train_data", mode="before")
    @classmethod
    def _check_train_data(cls, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            raise ValueError("train_data must be a non-empty pandas DataFrame.")
        if len(df) < 3:
            raise ValueError(
                "At least 3 samples are required to evaluate KMeans silhouette scores."
            )
        return df

    @model_validator(mode="after")
    def _check_cluster_bounds(self) -> ModelTrainInputs:
        n_samples = len(self.train_data)

        # Same logic as before: validate n_clusters if given, else max_clusters
        name, value = (
            ("n_clusters", self.n_clusters)
            if self.n_clusters is not None
            else ("max_clusters", self.max_clusters)
        )
        if value < 2 or value >= n_samples:
            raise ValueError(
                f"{name} must be between 2 and {n_samples - 1} (got {value})."
            )
        return self
    
class ModelPredictInputs(BaseModel):
        model_config = ConfigDict(extra="forbid")
        
        DEBT: list[FiniteFloat] = Field(description="Total household debt (2019 USD).")
        NETWORTH: list[FiniteFloat] = Field(description="Total household net worth (2019 USD).")
        HOUSES: list[FiniteFloat] = Field(description="Total value of the household's primary residence (2019 USD).")
        NFIN: list[FiniteFloat] = Field(description="TTotal value of household non-financial assets (2019 USD).")
        ASSET: list[FiniteFloat] = Field(description=" Total value of household assets (2019 USD).")    