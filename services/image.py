import os
import numpy as np
from PIL import Image
import tensorflow as tf
from .base import BaseService
from utils.file import get_name_without_ext
from typing import Tuple

class ImageService(BaseService):
    """
    A service class for handling image files and performing predictions
    using a pre-trained deep learning model.

    Inherits from:
        BaseService: Provides a common interface for loading models 
        and making predictions.
    """

    def __init__(self, model_path: str = "./models/NewnetV1_trained.h5") -> None:
        """
        Initialize the ImageService with a trained image classification model.

        Args:
            model_path (str): Path to the pre-trained image model (.h5 file).
                              Default is './models/NewnetV1_trained.h5'.
        """
        super().__init__(model_path)

    def convert_image(self, filepath: str) -> Tuple[str, int]:
        """
        Convert an image to JPEG format if necessary and detect grayscale.

        Args:
            filepath (str): Path to the input image file.

        Returns:
            Tuple[str, int]:
                - str: New filename of the converted image (always .jpg).
                - int: 1 if the image is grayscale, 0 otherwise.
        """
        img: Image.Image = Image.open(filepath)
        gray: int = 0

        if img.mode in ["RGBA", "P"]:
            img = img.convert("RGB")
        if img.mode == "L":
            gray = 1

        new_filename: str = get_name_without_ext(os.path.basename(filepath)) + ".jpg"
        new_filepath: str = os.path.join(os.path.dirname(filepath), new_filename)
        img.save(new_filepath, quality=75, optimize=True)

        if filepath != new_filepath:
            os.remove(filepath)

        return new_filename, gray

    def preprocess_image(self, filepath: str, is_gray: int) -> np.ndarray:
        """
        Preprocess an image into a tensor suitable for model input.

        Steps:
            - Decode image (grayscale converted to RGB if needed).
            - Normalize pixel values to [0, 1].
            - Resize to (224, 224).
            - Expand dimensions to create batch size.

        Args:
            filepath (str): Path to the image file (.jpg).
            is_gray (int): Flag indicating if the image is grayscale (1) or not (0).

        Returns:
            np.ndarray: Preprocessed image array with shape (1, 224, 224, 3).
        """
        img = tf.io.read_file(filepath)

        if is_gray:
            img = tf.image.decode_jpeg(img, channels=1)
            img = tf.image.grayscale_to_rgb(img)
        else:
            img = tf.image.decode_jpeg(img, channels=3)

        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, [224, 224])
        return np.expand_dims(img, axis=0)

    def predict_image(self, filepath: str) -> float:
        """
        Perform prediction on an image file.

        Steps:
            - Convert the image to JPEG if required.
            - Preprocess the image into a normalized tensor.
            - Pass preprocessed image to the model for prediction.

        Args:
            filepath (str): Path to the input image file.

        Returns:
            float: Prediction score from the model.
        """
        filename, gray = self.convert_image(filepath)
        processed: np.ndarray = self.preprocess_image(filepath, gray)
        return super().predict(processed)
