"""
Simple Flask server that accepts an audio file and attempts to infer why a baby is
crying.  This prototype uses a very naive rule‑based classifier because a real
model would require training on labelled baby‑cry datasets.  The purpose of
this script is to demonstrate how the audio processing pipeline could be
implemented on a mobile device or backend server.  Audio stays local to the
device when using this code on a phone.

Endpoint:
  POST /classify
    Accepts a multipart/form‑data request with a single file field named
    ``file``.  The uploaded file should be an audio clip (wav, mp3, etc.).
    It returns a JSON object with a predicted reason and some extracted
    features (mean zero‑crossing rate, root‑mean‑square energy and spectral
    centroid).  The classifier in this example uses simple threshold
    heuristics on these features to decide between hunger, burping, belly
    pain and discomfort.

Usage:
  python server.py

This will start a development server at http://localhost:5000.  You can test
the endpoint with curl:

  curl -F file=@path/to/cry.wav http://localhost:5000/classify
"""

import io
from flask import Flask, request, jsonify
import numpy as np

try:
    # Librosa is used for audio feature extraction.  It is a heavy
    # dependency but provides convenient functions for computing
    # zero‑crossing rate, RMS energy and spectral centroid.
    import librosa
except Exception as e:
    raise RuntimeError(
        "librosa is required for this prototype. Install it via pip install librosa"
    )


app = Flask(__name__)


@app.route("/classify", methods=["POST"])
def classify_audio() -> "tuple[dict, int]":
    """Classify the uploaded baby cry audio.

    Returns a JSON response with the predicted reason for the cry and the
    extracted feature values.  If no file is provided or audio cannot be
    processed, returns an error message.
    """
    if "file" not in request.files:
        return jsonify({"error": "Please upload an audio file using the 'file' field."}), 400

    file_storage = request.files["file"]
    if file_storage.filename == "":
        return jsonify({"error": "Empty filename provided."}), 400

    try:
        # Read the file into a bytes buffer
        raw = io.BytesIO(file_storage.read())
        # librosa can load many audio formats directly from file‑like objects.
        y, sr = librosa.load(raw, sr=22050, mono=True)
    except Exception as e:
        return jsonify({"error": f"Unable to read audio: {e}"}), 400

    # Guard against empty audio
    if y.size == 0:
        return jsonify({"error": "The uploaded file contains no audio samples."}), 400

    # Extract features.  These are simple summary statistics over the entire
    # clip.  In a real application you would compute more sophisticated
    # features (MFCCs, mel‑spectrograms, etc.) and feed them into a
    # trained classifier.
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    rmse = float(np.mean(librosa.feature.rms(y)))
    spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))

    # Very naive classification based on arbitrary thresholds.  These values
    # were chosen heuristically for demonstration and do not reflect
    # empirical baby‑cry statistics.  Replace this logic with a proper
    # machine‑learning model trained on labelled cries for meaningful results.
    if spectral_centroid > 3500:
        predicted = "belly_pain"
    elif rmse > 0.05:
        predicted = "hunger"
    elif zcr > 0.1:
        predicted = "burping"
    else:
        predicted = "discomfort"

    return jsonify(
        {
            "reason": predicted,
            "features": {
                "zero_crossing_rate": zcr,
                "rms_energy": rmse,
                "spectral_centroid": spectral_centroid,
            },
        }
    )


if __name__ == "__main__":
    # Running with debug=True is convenient during development but should be
    # disabled in production.
    app.run(debug=True)