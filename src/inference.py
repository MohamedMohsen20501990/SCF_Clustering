from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import pandas as pd

from src.utils import load_obj
from src.logging_config import logger
from src.schemas import ModelPredictInputs

class InferencePipeline:
    """Run predictions with a trained model on raw user input.

    The pipeline loads a serialized model and the list of selected features,
    validates incoming data with ``ModelPredictInputs``, arranges it into a
    DataFrame with the expected column order, and returns the model prediction.
    """
    
    def __init__(self, model_path: str | Path):
        self.model = load_obj(model_path)
        self.preprocessor = load_obj("artifacts/preprocessor.pkl")
        self.high_var_cols = load_obj("artifacts/selected_features.pkl")
        
    def _prepare_input(self, inputs:ModelPredictInputs):
        """Convert validated inputs into a model-ready DataFrame.

        Args:
            inputs: A validated ``ModelPredictInputs`` instance.

        Returns:
            A float-typed DataFrame whose columns follow the order of the
            selected features used during training.
        """
        
        data = inputs.model_dump()
        missing_features = set(self.high_var_cols) - set(data.keys())

        if missing_features:
            raise ValueError(
                f"Missing features: {missing_features}"
            )
        
        df = pd.DataFrame(data, columns=self.high_var_cols, dtype=float)
        logger.debug(f"Prepared input DataFrame with shape {df.shape}.")
        return df
        
        
    def predict(self, input_data:dict):
        """Validate raw input and return the model's prediction.

        Args:
            input_data: Dictionary mapping feature names to lists of values.

        Returns:
            The predicted class for the first row as an ``int``.
        """
        
        validated = ModelPredictInputs(**input_data)
        df = self._prepare_input(validated)
        df_preprocessed = self.preprocessor.transform(df)
        prediction = self.model.predict(df_preprocessed)
        logger.info(f"Prediction completed: {prediction}")

        
        return prediction.tolist()
            

def main():
    user_input = {
    "DEBT": [2.74878375, 26.22225792, 7.1306009, -0.06051075],
    "NETWORTH": [2.90405301, 6.98601403, 27.30267308, -0.0628921],
    "HOUSES": [2.66685552, 25.11671097, 10.92991402, -0.06273942],
    "NFIN": [2.46081347, 8.46368664, 27.676112, -0.06103904],
    "ASSET": [2.95943364, 7.88662677, 27.09107, -0.06412986]
}
    
    inference_pipeline = InferencePipeline(model_path = "artifacts/model.pkl")
    result = inference_pipeline.predict(input_data= user_input)
    print(result)
    

if __name__ == "__main__":
    main()    