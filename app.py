from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS
import os

from config import UPLOAD_FOLDER, ALLOWED_EXTENSIONS, IMAGE_FOLDER, AUDIO_FOLDER, TEXT_FOLDER
from utils.file import check_ext
from utils.docx import convert_docx_to_txt
from utils.pdf import convert_pdf_to_text
from services.text import TextService
from services.image import ImageService
from services.audio import AudioService

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Initialize ML services
text_service = TextService()
image_service = ImageService()
audio_service = AudioService()


@app.route("/predict", methods=["POST"])
def predict_deepfake():
    """
    API endpoint for deepfake prediction.

    Accepts a file upload (text, image, or audio), processes it with
    the appropriate model, and returns a prediction score.

    Returns:
        JSON response:
            - {"prediction": float} if successful
            - {"error": str} with status 400 for invalid inputs
    """
    # Validate file
    f = request.files.get("file")
    if not f or not check_ext(f.filename, ALLOWED_EXTENSIONS):
        return jsonify({"error": "Invalid or missing file"}), 400

    filename = secure_filename(f.filename)
    ext = filename.split(".")[-1].lower()

    # ---- Text files (txt, docx, pdf) ----
    if ext in {"txt", "docx", "pdf"}:
        file_path = os.path.join(TEXT_FOLDER, filename)
        f.save(file_path)

        if ext == "docx":
            text = convert_docx_to_txt(file_path)
        elif ext == "pdf":
            text = convert_pdf_to_text(file_path)
        else:  # plain .txt
            with open(file_path, "r", encoding="utf-8", errors="ignore") as fi:
                text = fi.read()

        prediction = text_service.predict_text(text)

    # ---- Image files (jpg, jpeg, png) ----
    elif ext in {"jpg", "jpeg", "png"}:
        file_path = os.path.join(IMAGE_FOLDER, filename)
        f.save(file_path)
        prediction = image_service.predict_image(file_path)

    # ---- Audio files (wav, mp3, ogg) ----
    elif ext in {"ogg", "wav", "mp3"}:
        file_path = os.path.join(AUDIO_FOLDER, filename)
        f.save(file_path)
        prediction = audio_service.predict_audio(file_path)

    else:
        return jsonify({"error": "Unsupported file format"}), 400

    return jsonify({"prediction": prediction * 100})
