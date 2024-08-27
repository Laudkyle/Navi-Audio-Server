import os
import sounddevice as sd
import soundfile as sf
import requests
from io import BytesIO

# Define the server URL
SERVER_URL = 'http://<server-ip>:5000/predict'  # Replace <server-ip> with your server's IP address

def record_audio(duration=2, sr=16000):
    print("Recording...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()  # Wait until recording is finished
    print("Recording finished.")
    audio = audio.flatten()
    return audio

def send_audio_to_server(audio, sr=16000):
    # Convert audio to BytesIO object
    audio_buffer = BytesIO()
    sf.write(audio_buffer, audio, sr, format='wav')
    audio_buffer.seek(0)
    
    # Send audio file to server
    files = {'file': ('audio.wav', audio_buffer, 'audio/wav')}
    response = requests.post(SERVER_URL, files=files)
    
    # Print server response
    if response.status_code == 200:
        print("Server Response:", response.json())
    else:
        print("Server Error:", response.text)

if __name__ == '__main__':
    # Record audio
    audio = record_audio(duration=2, sr=16000)
    
    # Send audio to the server
    send_audio_to_server(audio, sr=16000)
