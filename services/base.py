import tensorflow as tf
import numpy as np
from typing import Any, Dict, Optional

class BaseService:
    """
    Base class for machine learning services that handle 
    loading TensorFlow/Keras models and running predictions.
    """

    def __init__(self, model_path: str, custom_objects: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the BaseService by loading a pre-trained Keras model.

        Args:
            model_path (str): Path to the saved model (.h5, .keras, or SavedModel directory).
            custom_objects (dict, optional): Dictionary of custom objects 
                                             required to load the model (e.g., custom layers).
                                             Defaults to None.
        """
        self.model = tf.keras.models.load_model(model_path, custom_objects or {})

    def predict(self, data: np.ndarray) -> float:
        """
        Run inference using the loaded model on the given input data.

        Args:
            data (np.ndarray): Preprocessed input data matching the model's expected shape.

        Returns:
            float: Scalar prediction value from the model.
        """
        prediction: np.ndarray = self.model.predict(data)
        return prediction.tolist()[0][0]  # Always return scalar
