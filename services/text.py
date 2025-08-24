import numpy as np
import tensorflow_hub as hub
from .base import BaseService


class TextService(BaseService):
    """
    A service class for handling text inputs and performing predictions
    using a fine-tuned BERT model or similar transformer-based models.

    Inherits from:
        BaseService: Provides a common interface for loading models 
        and making predictions.
    """

    def __init__(self, model_path: str = "./models/trained_bert.h5") -> None:
        """
        Initialize the TextService with a trained text classification model.

        Args:
            model_path (str): Path to the pre-trained text model (.h5 file).
                              Default is './models/trained_bert.h5'.
        """
        super().__init__(model_path, {"KerasLayer": hub.KerasLayer})

    def preprocess(self, text: str) -> np.ndarray:
        """
        Preprocess text input into a format suitable for model inference.

        Steps:
            - Expand dimensions so the model receives a batch input.

        Args:
            text (str): The input text string.

        Returns:
            np.ndarray: Preprocessed text array with shape (1,).
        """
        return np.expand_dims(text, axis=0)

    def predict_text(self, text: str) -> float:
        """
        Perform prediction on a text input.

        Steps:
            - Preprocess the text into model-compatible format.
            - Run the model prediction.
            - Invert the result (1 - prediction) for consistency with your setup.

        Args:
            text (str): The input text string.

        Returns:
            float: Prediction score (inverted).
        """
        processed: np.ndarray = self.preprocess(text)
        return 1 - super().predict(processed)
