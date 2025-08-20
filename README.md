# Baby Cry Classifier Prototype

This repository contains a **very simple prototype** of a baby cry classifier.  It
includes two main components:

1. **Flask server (`server.py`)** – exposes a `/classify` endpoint that accepts
   an uploaded audio clip and returns a predicted reason for the baby’s cry.
   The classifier implemented here is *rule based* and uses basic audio
   features (zero‑crossing rate, root‑mean‑square energy and spectral centroid).
   It is **not accurate** and serves only as an example pipeline.  To build a
   production‑ready system you would need to train a neural network on a
   labelled dataset of baby cries (see the research summary for guidance).

2. **React Native mobile app (`mobile_app`)** – a minimal Expo application that
   records audio from the phone’s microphone and sends it to the Flask server
   for classification.  The app displays the predicted reason returned by the
   server.

## Prerequisites

* **Python 3.9+** and `pip` for the server.
* **Node.js** and **Expo CLI** (`npm install -g expo-cli`) for the mobile app.
* A device or emulator to run the mobile app.

## Setup

### Server

1. Create a virtual environment (recommended):

   ```bash
   cd baby_cry_app
   python3 -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the server:

   ```bash
   python server.py
   ```

   The server will start on port 5000.  You should see output like:

   ```
   * Serving Flask app 'server'
   * Debug mode: on
   * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
   ```

   To classify a recording manually you can use `curl` or an HTTP client:

   ```bash
   curl -F file=@/path/to/audio.wav http://127.0.0.1:5000/classify
   ```

### Mobile app

1. Open a new terminal window and navigate to the mobile directory:

   ```bash
   cd baby_cry_app/mobile_app
   npm install
   ```

2. Edit `App.js` and replace the `SERVER_URL` constant with the IP address of
   your computer on the same network.  For example:

   ```js
   const SERVER_URL = 'http://192.168.0.10:5000/classify';
   ```

3. Start the app with Expo:

   ```bash
   npm start
   ```

   This will open the Expo developer tools in your browser.  Use the QR code
   to launch the app on your phone (you need the Expo Go app installed) or
   choose to run on an Android/iOS emulator.

4. Inside the app, tap **“Start Recording”** to record a few seconds of your
   baby’s cry (or any sound).  When you tap **“Stop Recording,”** the app
   uploads the audio to the server and displays the predicted reason.

## Limitations

* **Accuracy:**  The rule‑based classifier provided here is *not reliable*.
  Without training on real baby cries it cannot meaningfully detect hunger,
  burping, etc.  To achieve accuracy similar to the research (≈96 %) you must
  collect or obtain a labelled dataset and train a deep learning model.  Refer
  to the accompanying research notes (in your conversation) for details on
  feature extraction (MFCC, Mel‑spectrograms) and recommended model
  architectures (CNN, CNN+RNN).
* **Privacy:**  The server processes audio on the machine where it runs.  A
  production mobile app should perform inference *on the device* using a
  library like TensorFlow Lite to avoid transmitting audio off the phone.
* **Permissions:**  The app requests microphone permissions at runtime.  Be sure
  to enable them when prompted.

Feel free to extend this prototype by replacing the classifier in
`server.py` with your own model and improving the mobile UI!