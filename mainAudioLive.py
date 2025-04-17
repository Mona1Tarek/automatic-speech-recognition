import pyaudio
from vosk import Model, KaldiRecognizer
import json

# Load the Vosk model
model_path = "/home/mona/voiceCommand/vosk-model-small-en-us-0.15"
model = Model(model_path)

# Set up the microphone
samplerate = 16000
p = pyaudio.PyAudio()

# Open the microphone stream
stream = p.open(format=pyaudio.paInt16, channels=1, rate=samplerate, input=True, frames_per_buffer=4000)
stream.start_stream()

# Set up the recognizer
rec = KaldiRecognizer(model, samplerate)

print("Start speaking...")

# Continuously listen and recognize speech
while True:
    data = stream.read(4000)
    if rec.AcceptWaveform(data):
        result = rec.Result()
        print("Recognized text:", json.loads(result)["text"])

    # Handle partial results, if any
    partial_result = rec.PartialResult()
    partial_data = json.loads(partial_result)
    
    # Check if 'partial' key exists before printing
    if "partial" in partial_data:
        print("Partial result:", partial_data["partial"])

