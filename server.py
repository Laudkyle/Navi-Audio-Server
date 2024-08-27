from flask import Flask, request, jsonify
import os
import librosa
import numpy as np
from tensorflow.keras.models import load_model
import sounddevice as sd
import soundfile as sf
from io import BytesIO

# Initialize Flask app
app = Flask(__name__)

# Define the commands and speakers
commands = ['person', 'read']
speakers = ['Barak', 'Joe']  # Add your actual speaker names here

# Load the trained model
model = load_model('Speaker_model.h5')

def preprocess_audio_from_array(audio, sr=16000, n_mfcc=13, fixed_length=2):
    y = librosa.util.fix_length(audio, size=sr * fixed_length)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfccs

def predict_from_audio(audio, model):
    # Preprocess the recorded audio
    mfccs = preprocess_audio_from_array(audio)
    mfccs = mfccs[np.newaxis, ..., np.newaxis]
    
    # Predict using the model
    predictions = model.predict(mfccs)
    command_label = np.argmax(predictions[0])
    speaker_label = np.argmax(predictions[1])
    
    return command_label, speaker_label

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        # Read the audio file
        audio_data = file.read()
        audio, sr = sf.read(BytesIO(audio_data))
        
        # Ensure audio is mono
        if len(audio.shape) > 1:
            audio = audio[:, 0]
        
        # Record a 2-second audio clip
        command_label, speaker_label = predict_from_audio(audio, model)
        
        # Get command and speaker names
        predicted_command = commands[command_label]
        predicted_speaker = speakers[speaker_label]
        
        return jsonify({'command': predicted_command, 'speaker': predicted_speaker}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
