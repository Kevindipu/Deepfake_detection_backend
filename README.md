# Multimodal Deepfake Detection Backend

A Flask-based backend API for detecting deepfake content across **text, images, and audio** using deep learning models. The application automatically identifies the uploaded file type, performs the required preprocessing, runs inference using a trained model, and returns a prediction score through a single REST API.

## Features

- **Single `/predict` API** for all supported media types
- **Image Deepfake Detection** using a TensorFlow/Keras model
- **Audio Deepfake Detection** using MFCC feature extraction
- **Text Deepfake Detection** using a fine-tuned BERT model
- Automatic file validation and preprocessing
- CORS enabled for frontend integration

## Supported File Formats

| Media | Formats |
|--------|----------|
| Text | `.txt`, `.docx`, `.pdf` |
| Image | `.jpg`, `.jpeg`, `.png` |
| Audio | `.wav`, `.mp3`, `.ogg` |

## Tech Stack

- **Backend:** Flask
- **Machine Learning:** TensorFlow/Keras
- **NLP:** BERT (TensorFlow Hub)
- **Audio Processing:** Librosa, PyDub
- **Image Processing:** Pillow
- **Document Processing:** python-docx, pdfminer

## Project Structure

```text
Deepfake_detection_backend/
│
├── app.py                  # Flask API entry point
├── config.py               # Configuration and upload paths
├── requirements.txt
├── app.yaml
│
├── services/
│   ├── base.py             # Shared model loading logic
│   ├── image.py            # Image preprocessing + inference
│   ├── audio.py            # Audio preprocessing + inference
│   └── text.py             # Text preprocessing + inference
│
├── utils/
│   ├── file.py             # File validation helpers
│   ├── docx.py             # DOCX to text conversion
│   └── pdf.py              # PDF text extraction
│
├── models/                 # Trained models (not included)
└── static/
    ├── images/
    ├── audio/
    └── text/
```

## How It Works

<pre>
Client Upload
      │
      ▼
  /predict API
      │
      ▼
 File Validation
      │
      ▼
Detect File Type
      │
 ┌────┼────────────┐
 ▼    ▼            ▼
Text Image       Audio
 │     │            │
 ▼     ▼            ▼
Preprocessing
 │     │            │
 ▼     ▼            ▼
ML Model Inference
      │
      ▼
Prediction Score
      │
      ▼
 JSON Response
</pre>

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Kevindipu/Deepfake_detection_backend.git
cd Deepfake_detection_backend
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

**Windows**

```bash
venv\Scripts\activate
```

**Linux/macOS**

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Running the Server

```bash
python app.py
```

The Flask server will start locally (typically on `http://127.0.0.1:5000`).

## API Reference

### Predict Deepfake

**Endpoint**

```http
POST /predict
```

**Content-Type**

```text
multipart/form-data
```

**Body**

| Field | Type | Required |
|--------|------|----------|
| file | File | Yes |

### Example using cURL

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -F "file=@sample.jpg"
```

### Example Response

```json
{
  "prediction": 87.42
}
```

The returned value represents the model's prediction score as a percentage.

## Preprocessing Pipeline

### Image

- Converts images to JPEG when required
- Handles grayscale and RGB images
- Resizes to **224×224**
- Normalizes pixel values

### Audio

- Converts MP3/OGG to WAV
- Extracts **25 MFCC** features
- Resizes feature maps to **96×64**
- Creates a model-ready tensor

### Text

- Accepts plain text or extracts text from PDF and DOCX files
- Formats text for BERT inference

## Configuration

`config.py` defines upload locations and supported file extensions.

```python
UPLOAD_FOLDER = "./static"

IMAGE_FOLDER = "./static/images"
AUDIO_FOLDER = "./static/audio"
TEXT_FOLDER = "./static/text"
```

Allowed extensions:

```text
txt, docx, pdf,
jpg, jpeg, png,
mp3, wav, ogg
```

## Model Files

The repository expects pre-trained models inside the `models/` directory.

| Model | Purpose |
|--------|----------|
| `trained_bert.h5` | Text deepfake detection |
| `NewnetV1_trained.h5` | Image deepfake detection |
| `vggish_model_new.h5` | Audio deepfake detection |

> These model files are not included in the repository.

## Future Improvements

- Docker support
- Batch prediction endpoint
- Confidence score calibration
- Request logging
- Model versioning
- Authentication for production deployment

