import os
import numpy as np
import librosa
import tensorflow as tf
from pydub import AudioSegment
from .base import BaseService
from utils.file import get_name_without_ext

class AudioService(BaseService):
    """
    A service class for handling audio files and performing predictions
    using a pre-trained deep learning model.

    Inherits from:
        BaseService: Provides a common interface for loading models 
        and making predictions.
    """

    def __init__(self, model_path: str = "./models/vggish_model_new.h5") -> None:
        """
        Initialize the AudioService with a trained audio classification model.

        Args:
            model_path (str): Path to the pre-trained audio model (.h5 file).
                              Default is './models/vggish_model_new.h5'.
        """
        super().__init__(model_path)

    def convert_audio(self, filepath: str) -> str:
        """
        Convert an audio file (mp3/ogg) to WAV format if necessary.

        Args:
            filepath (str): Path to the input audio file.

        Returns:
            str: Path to the converted (or original) WAV file.

        Raises:
            ValueError: If the file format is not supported.
        """
        name: str = get_name_without_ext(os.path.basename(filepath))
        ext: str = filepath.split(".")[-1].lower()

        if ext == "wav":
            return filepath
        if ext == "mp3":
            audio = AudioSegment.from_mp3(filepath)
        elif ext == "ogg":
            audio = AudioSegment.from_ogg(filepath)
        else:
            raise ValueError("Unsupported audio format")

        new_filepath: str = os.path.join(os.path.dirname(filepath), f"{name}.wav")
        audio.export(new_filepath, format="wav")
        os.remove(filepath)
        return new_filepath

    def preprocess_audio(self, filepath: str) -> np.ndarray:
        """
        Preprocess an audio file into an MFCC feature tensor suitable for model input.

        Steps:
            - Load waveform using librosa.
            - Extract MFCC features.
            - Resize the MFCC array to (96, 64).
            - Expand dimensions to match model input requirements.

        Args:
            filepath (str): Path to the WAV audio file.

        Returns:
            np.ndarray: Preprocessed audio array with shape (-1, 96, 64, 1).
        """
        waveform, sample_rate = librosa.load(filepath, sr=None)
        mfcc = librosa.feature.mfcc(
            y=waveform, sr=sample_rate,
            n_mfcc=25, n_fft=4096, hop_length=512
        )
        mfcc = tf.image.resize(np.expand_dims(mfcc, -1), (96, 64), method="nearest")
        mfcc = tf.expand_dims(mfcc, axis=-1)
        arr: np.ndarray = np.array([mfcc])
        return arr.reshape(-1, 96, 64, 1)

    def predict_audio(self, filepath: str) -> float:
        """
        Perform prediction on an audio file.

        Steps:
            - Convert file to WAV if required.
            - Preprocess the audio into MFCC features.
            - Pass preprocessed features to the model for prediction.

        Args:
            filepath (str): Path to the input audio file.

        Returns:
            float: Prediction score from the model.
        """
        new_filepath: str = self.convert_audio(filepath)
        processed: np.ndarray = self.preprocess_audio(new_filepath)
        return super().predict(processed)
